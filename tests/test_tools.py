"""The offline halves of the pipeline instrumentation: the trace summarizer
and the replay harness's audio handling."""
import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import replay            # noqa: E402
import trace_report      # noqa: E402

TOOLS = Path(__file__).parent.parent / "tools"


def record(**over):
    rec = {"t": 1_800_000_000.0, "uid": 1, "speaker": "you", "chunk_sec": 8.1,
           "split": "soft_max", "spec": "miss", "whisper_queue": 1,
           "partials_skipped": 3, "wait_ms": 40, "in_flight": 2,
           "outcome": "final", "transcribe_ms": 900, "translate_ms": 1500,
           "refine_ms": 700, "chars": 120, "lag_ms": 2500,
           "first_word_lag_ms": 10600}
    rec.update(over)
    return rec


# ------------------------------------------------------------- trace_report


def test_report_attributes_the_delay(capsys):
    trace_report.report([record(uid=1), record(uid=2, split="pause",
                                               chunk_sec=2.0, spec="hit",
                                               first_word_lag_ms=4500)])
    out = capsys.readouterr().out
    assert "2 utterances" in out
    assert "soft_max" in out and "pause" in out
    # The headline: the felt delay broken into measured parts.
    assert "audio accumulating" in out and "translating" in out
    assert "1 used, 1 wasted" in out
    assert "lag, FIRST word" in out


def test_report_survives_a_torn_last_line(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(record()) + "\n" + '{"uid": 2, "chunk')
    assert len(trace_report.load(path)) == 1


def test_report_counts_discards_without_crashing(capsys):
    """Discarded utterances have no translate/lag fields, but they did use
    the Whisper thread — the summary must still include them."""
    trace_report.report([{"t": 1.0, "uid": 1, "outcome": "discard_empty",
                          "split": "pause", "spec": "none",
                          "whisper_queue": 0, "transcribe_ms": 500}])
    out = capsys.readouterr().out
    assert "1 utterances" in out and "1 discarded" in out


def test_percentiles_are_ordered():
    values = list(range(100))
    assert trace_report.pct(values, .5) <= trace_report.pct(values, .9)
    assert trace_report.pct([], .5) == 0.0


@pytest.mark.parametrize("spec,seconds", [("30s", 30), ("10m", 600), ("2h", 7200)])
def test_since_accepts_human_windows(spec, seconds):
    assert trace_report.parse_since(spec) == seconds


def test_report_cli_runs(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(record(uid=i)) for i in range(5)))
    out = subprocess.run([sys.executable, str(TOOLS / "trace_report.py"),
                          "--path", str(path), "--last", "3"],
                         capture_output=True, text=True, check=True).stdout
    assert "3 utterances" in out


# ------------------------------------------------------------------ replay


def test_default_text_has_no_sentence_breaks():
    """`say` pauses at a full stop, and a pause is exactly what this text must
    not give the VAD — the whole point is speech with no natural gaps."""
    assert "." not in replay.DEFAULT_TEXT
    assert "?" not in replay.DEFAULT_TEXT and "!" not in replay.DEFAULT_TEXT
    assert len(replay.DEFAULT_TEXT.split()) > 100   # long enough to force splits


def test_read_pcm_rejects_the_wrong_format(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"\x00" * 400)
    with pytest.raises(SystemExit) as exc:
        replay.read_pcm(path)
    assert "16 kHz" in str(exc.value)


def test_read_pcm_accepts_the_server_format(tmp_path):
    path = tmp_path / "ok.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x01\x02" * 1000)
    assert len(replay.read_pcm(path)) == 2000
