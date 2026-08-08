"""The offline halves of the pipeline instrumentation: the trace summarizer
and the replay harness's audio handling."""
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import capture_refines   # noqa: E402
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


# ---------------------------------------------------------- capture_refines


def ev_utterance(uid, heard, draft, revised=None, target="en", **final_extra):
    """The websocket traffic one translated utterance produces."""
    out = [{"type": "final", "id": uid, "text": heard, "source": "de",
            "target": target, **final_extra}]
    # Ollama's deltas carry their own whitespace; splitting it off here would
    # make the reassembly test pass against a draft no client ever saw.
    out += [{"type": "translation_delta", "id": uid, "target": target,
             "text": piece} for piece in re.findall(r"\S+\s*", draft)]
    out.append({"type": "translation_done", "id": uid})
    out.append({"type": "translation_revised", "id": uid,
                "texts": {} if revised is None else {target: revised}})
    return out


def trace_rec(uid, refine="landed", **over):
    rec = record(uid=uid, refine=refine, refine_wait_ms=4600,
                 refine_changed=False, agreement_changed=False)
    rec.update(over)
    return rec


def test_the_draft_is_reassembled_from_its_deltas():
    """The draft is what the user read first, and it only exists as a stream
    of deltas — comparing a refine against anything else is unfair."""
    events = ev_utterance(1, "Hallo", "he does not have solar")
    assert capture_refines.collect_drafts(events)[(1, "en")] == \
        "he does not have solar"


def test_a_landed_refine_that_changed_the_text_is_reported():
    events = ev_utterance(1, "er hat keinen Solar",
                          "he doesn't have central heating",
                          "he doesn't have solar panels")
    rows = capture_refines.join_refines(
        events, [trace_rec(1, refine_changed=True)])
    assert len(rows) == 1
    assert rows[0]["changed_by"] == "refine"
    assert rows[0]["draft"].endswith("heating")
    assert rows[0]["final"].endswith("panels")


def test_a_declension_fix_is_not_counted_as_a_refine():
    """The join this tool exists for. Both passes deliver their result in the
    same `translation_revised`, so without the server naming which one acted,
    an agreement retry inflates the landed-refine count."""
    events = ev_utterance(1, "das Haus", "in dem Haus", "in das Haus")
    rows = capture_refines.join_refines(
        events, [trace_rec(1, refine="gated", agreement_changed=True)])
    assert rows[0]["changed_by"] == "agreement"
    assert capture_refines.summarize(rows)["changed_by_refine"] == 0
    assert capture_refines.summarize(rows)["changed_by_agreement"] == 1


def test_a_gate_skipped_refine_is_not_read_as_a_fast_one():
    """The specific mistake that produced the bogus "delivery is 39%" figure:
    a refine the gate never ran also sits near 0 ms, so counting "under the
    timeout" as delivery counted skips as successes."""
    events = ev_utterance(1, "hallo", "hello")
    rows = capture_refines.join_refines(
        events, [trace_rec(1, refine="gated", refine_wait_ms=0, refine_ms=3)])
    s = capture_refines.summarize(rows)
    assert s["counts"]["gated"] == 1 and s["counts"]["landed"] == 0
    assert s["attempted"] == 0
    assert s["kill_rate"] is None      # nothing was attempted; not "0% killed"


def test_a_timeout_is_distinguished_from_a_model_error():
    rows = capture_refines.join_refines(
        ev_utterance(1, "a", "b") + ev_utterance(2, "c", "d"),
        [trace_rec(1, refine="timeout"), trace_rec(2, refine="error")])
    s = capture_refines.summarize(rows)
    assert s["counts"]["timeout"] == 1 and s["counts"]["error"] == 1
    # A broken Ollama must not read as a slow one: both are attempts, but
    # only the timeout says anything about the ceiling.
    assert s["attempted"] == 2 and s["kill_rate"] == 0.5


def test_a_merged_card_does_not_count_twice():
    """A fragment merged into its continuation is replaced on screen; the
    replaced uid never gets its own translation and would otherwise look like
    an utterance whose refine vanished."""
    events = ev_utterance(1, "Von Gottbergs hießen die, wo man da als", "x")
    events += ev_utterance(2, "als Küchenbammser gearbeitet hat", "y",
                           replaces=1)
    rows = capture_refines.join_refines(
        events, [trace_rec(1), trace_rec(2)])
    assert [r["uid"] for r in rows] == [2]


def test_discarded_utterances_are_left_out():
    rows = capture_refines.join_refines(
        ev_utterance(1, "hm", "hm"),
        [trace_rec(1, outcome="discard_empty")])
    assert rows == []


def test_the_p50_covers_landed_refines_only():
    """Averaging in the gated rows would drag the median toward zero and make
    the pass look far cheaper than it is."""
    events, recs = [], []
    for uid, (outcome, wait) in enumerate(
            [("landed", 4000), ("landed", 5000), ("landed", 6000),
             ("gated", 0), ("gated", 0)], start=1):
        events += ev_utterance(uid, "x", "y")
        recs.append(trace_rec(uid, refine=outcome, refine_wait_ms=wait))
    assert capture_refines.summarize(
        capture_refines.join_refines(events, recs))["landed_wait_p50"] == 5000


def test_a_punctuation_style_difference_is_not_a_changed_translation():
    """Measured on the real slice: gemma3:12b writes curly apostrophes and
    qwen2.5:7b writes straight ones, so one landed refine in eleven differed
    only by `it's` versus `it’s`. Counting those inflates what the pass
    delivers, and the effect scales with how many contractions appear."""
    events = ev_utterance(1, "auf niedrigem Niveau",
                          "then it's at such a low level.",
                          "then it’s at such a low level.")
    rows = capture_refines.join_refines(
        events, [trace_rec(1, refine_changed=True)])
    assert rows[0]["changed_by"] == "refine"     # the server is not wrong
    assert rows[0]["substantive"] is False       # ...but a reader sees nothing
    s = capture_refines.summarize(rows)
    assert s["substantive"] == 0 and s["typographic_only"] == 1


def test_a_real_rewording_still_counts_as_substantive():
    events = ev_utterance(1, "das war mächtig",
                          "that was quite a bit in between here",
                          "it was pretty intense here at one point")
    rows = capture_refines.join_refines(
        events, [trace_rec(1, refine_changed=True)])
    assert rows[0]["substantive"] is True
    assert capture_refines.summarize(rows)["typographic_only"] == 0


def test_report_prints_the_pairs_to_read(capsys):
    events = ev_utterance(1, "damit es noch in den Skimmer geht",
                          "so it goes into the filter",
                          "so it goes into the skimmer")
    capture_refines.report(capture_refines.join_refines(
        events, [trace_rec(1, refine_changed=True)]))
    out = capsys.readouterr().out
    assert "into the filter" in out and "into the skimmer" in out
    assert "1 utterances translated" in out
