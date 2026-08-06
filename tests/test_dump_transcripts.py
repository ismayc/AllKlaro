"""Tests for the ground-truth transcript dumper.

The dump exists to settle items 4 and 5 by hand-checking real ASR output, so
the thing that must not break is fidelity: the utterances have to come from
the app's own VAD, and a row has to record what Whisper said *and* what the
filters did to it. Whisper itself is stubbed — this asserts the plumbing, not
the model.
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import server as srv
from tools import dump_transcripts as dt


def write_wav(path, pcm, channels=1, rate=srv.SAMPLE_RATE):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype(np.int16).tobytes())


def tone(n_frames, amp=8000):
    t = np.arange(n_frames * srv.FRAME_SAMPLES)
    return (amp * np.sin(2 * np.pi * 440 * t / srv.SAMPLE_RATE)).astype(np.int16)


def quiet(n_frames):
    return np.zeros(n_frames * srv.FRAME_SAMPLES, dtype=np.int16)


def test_frames_yields_whole_vad_frames(tmp_path):
    p = tmp_path / "m.wav"
    write_wav(p, quiet(7))
    got = list(dt.frames(str(p)))
    assert len(got) == 7
    assert all(len(f) == srv.FRAME_SAMPLES for f in got)


def test_frames_mixes_a_stereo_file_down(tmp_path):
    """The recording is one mic, but a stereo file must not be read as
    double-length audio — that would silently halve every duration."""
    p = tmp_path / "s.wav"
    mono = tone(4)
    stereo = np.repeat(mono, 2)              # same signal in both channels
    write_wav(p, stereo, channels=2)
    got = list(dt.frames(str(p)))
    assert len(got) == 4                     # not 8
    assert np.allclose(got[0], mono[:srv.FRAME_SAMPLES], atol=1)


def use_energy_vad(monkeypatch):
    """These fixtures are a 440 Hz tone, which Silero rightly says is not
    speech. The dumper loads Silero to match the server, so pin the energy
    scorer here — this asserts the plumbing, not the VAD model."""
    monkeypatch.setattr(srv, "load_silero", lambda: None)
    monkeypatch.setattr(srv, "make_scorer", lambda: srv.EnergyScorer())


def run_main(tmp_path, monkeypatch, result, language=None):
    audio = tmp_path / "a.wav"
    write_wav(audio, np.concatenate([quiet(10), tone(40), quiet(30)]))
    out = tmp_path / "out.jsonl"
    use_energy_vad(monkeypatch)
    monkeypatch.setattr(srv, "transcribe", lambda a, lang, prompt=None: result)
    argv = ["dump_transcripts.py", "--audio", str(audio), "--out", str(out)]
    if language:
        argv += ["--language", language]
    monkeypatch.setattr(sys, "argv", argv)
    dt.main()
    return [json.loads(l) for l in open(out)]


def test_a_row_records_language_and_both_texts(tmp_path, monkeypatch):
    rows = run_main(tmp_path, monkeypatch, {
        "text": " Hallo Welt.", "language": "de",
        "segments": [{"text": " Hallo Welt.", "compression_ratio": 1.2,
                      "avg_logprob": -0.3, "no_speech_prob": 0.1}]})
    assert len(rows) == 1
    r = rows[0]
    assert r["detected_language"] == "de"
    assert r["raw"] == "Hallo Welt."
    assert r["cleaned"] == "Hallo Welt."
    assert r["dropped"] is False
    assert r["split"] == "pause"
    assert r["dur_sec"] > 0


def test_a_dropped_artifact_is_flagged_with_its_raw_text_kept(tmp_path,
                                                              monkeypatch):
    """The point of the dump is to see what was thrown away, so a filtered
    utterance must keep its raw text rather than vanish from the file."""
    loop = " ni" * 60
    rows = run_main(tmp_path, monkeypatch, {
        "text": loop, "language": "de",
        "segments": [{"text": loop, "compression_ratio": 8.0,
                      "avg_logprob": -0.2, "no_speech_prob": 0.1}]})
    assert len(rows) == 1
    assert rows[0]["dropped"] is True
    assert rows[0]["cleaned"] == ""
    assert rows[0]["raw"].startswith("ni ni")     # evidence survives
    assert rows[0]["max_compression_ratio"] == 8.0


def test_language_flag_is_passed_through_to_whisper(tmp_path, monkeypatch):
    """Auto-detect vs forced `de` is the whole comparison for item 4; if the
    flag were dropped both dumps would be identical and prove nothing."""
    seen = []

    def fake(audio, language, prompt=None):
        seen.append(language)
        return {"text": " Ja.", "language": language or "en", "segments": []}

    use_energy_vad(monkeypatch)
    monkeypatch.setattr(srv, "transcribe", fake)
    audio = tmp_path / "a.wav"
    write_wav(audio, np.concatenate([quiet(10), tone(40), quiet(30)]))
    for lang in (None, "de"):
        argv = ["dump_transcripts.py", "--audio", str(audio),
                "--out", str(tmp_path / "o.jsonl")]
        if lang:
            argv += ["--language", lang]
        monkeypatch.setattr(sys, "argv", argv)
        dt.main()
    assert seen == [None, "de"]


def test_it_runs_as_a_script_not_only_as_an_import():
    """Every other test here imports the module, which pytest makes work by
    putting the repo root on sys.path. Run from the command line that is not
    true, and `import server` failed — invisible to an import-based test."""
    import subprocess
    r = subprocess.run(
        [sys.executable, str(Path(dt.__file__).resolve()), "--help"],
        capture_output=True, text=True, timeout=120,
        cwd=str(Path(dt.__file__).resolve().parent))
    assert r.returncode == 0, r.stderr[-800:]
    assert "--language" in r.stdout


def test_rejects_a_wrong_sample_rate(tmp_path):
    """Whisper and the VAD both assume 16 kHz; a 44.1 kHz file would produce
    plausible-looking nonsense rather than an error."""
    p = tmp_path / "bad.wav"
    write_wav(p, quiet(4), rate=44100)
    with pytest.raises(AssertionError):
        list(dt.frames(str(p)))
