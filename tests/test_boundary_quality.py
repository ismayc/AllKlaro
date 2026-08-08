"""Tests for the boundary-quality measurement.

This tool exists to answer one question honestly -- are the 5 s cap's cuts
worse than real pauses -- and it reached a null. A null is only worth
anything if the rig could have shown a difference, so what is pinned here is
the machinery that could quietly fake one:

  * the merge has to be the server's merge, not a lookalike, or the unit
    being scored is not the card a listener reads;
  * the casing rule is both half of one metric and the whole of the
    `lowercase` merge variant, so scoring that variant with that metric is
    circular and has to be refused rather than reported;
  * the arithmetic that turns counts into a claim.

Ollama is stubbed throughout -- this asserts the plumbing, not the model.
"""
import json
import sys
from pathlib import Path

import pytest

import server as srv
from tools import boundary_quality as bq


def utt(i, text, t_end, dur=2.0, split="soft_max", lang="de"):
    return {"i": i, "cleaned": text, "t_end": t_end, "dur_sec": dur,
            "split": split, "detected_language": lang}


# ------------------------------------------------------- the casing signal

@pytest.mark.parametrize("text,expected", [
    ("die ganze Zeit, weil...", True),      # a continuation
    ("angeschraubt wird.", True),           # verb-final tail of a clause
    ("Der Pool ist ja so ein L.", False),   # a sentence start
    ("Wasservolumen sofort auf.", False),   # German capitalises every noun
    ("...so eine L-Form", True),            # leading punctuation is not casing
    ("", False),
    ("2 grad ab", False),                   # a digit is not a lowercase letter
    ("   ", False),
])
def test_starts_lowercase(text, expected):
    assert bq.starts_lowercase(text) is expected


def test_uncased_transcript_is_the_known_blind_spot():
    """Whisper sometimes emits a chunk with no casing at all, and then a real
    sentence start reads as a continuation. Measured at 2.3% of German cards,
    and it inflates the cap arm rather than the pause arm, so the null holds
    in spite of it. Pinned so nobody 'fixes' it by accident."""
    assert bq.starts_lowercase("morgen für knapp zwei wochen nach hawaii")


# -------------------------------------------------------------- the merge

def test_merge_joins_an_unfinished_utterance_to_its_continuation():
    cards = bq.merge_cards([utt(1, "Also mal gucken, ob der Pool...", 5.0),
                            utt(2, "erreicht.", 7.0)])
    assert len(cards) == 1
    assert cards[0]["text"] == "Also mal gucken, ob der Pool... erreicht."
    assert cards[0]["parts"] == 2


def test_merge_leaves_a_finished_sentence_alone():
    cards = bq.merge_cards([utt(1, "Das macht keinen Sinn.", 5.0),
                            utt(2, "Ja, genau.", 7.0)])
    assert [c["parts"] for c in cards] == [1, 1]


def test_a_gap_longer_than_the_window_is_a_new_card():
    # start of the second is t_end - dur = 8.0, so the gap is 3 s > 2 s.
    cards = bq.merge_cards([utt(1, "Der Vermieter hat gesagt, wenn wir...", 5.0),
                            utt(2, "die Wärme halten wollten.", 10.0)])
    assert len(cards) == 2


def test_a_language_change_is_never_merged():
    cards = bq.merge_cards([utt(1, "und dann gab es...", 5.0),
                            utt(2, "we had a problem.", 7.0, lang="en")])
    assert len(cards) == 2


def test_a_card_stops_growing_at_the_length_cap():
    long_unfinished = "und " * (srv.MERGE_MAX_CHARS // 4 + 1)
    assert not srv.looks_finished(long_unfinished)
    cards = bq.merge_cards([utt(1, long_unfinished, 5.0),
                            utt(2, "weiter.", 7.0)])
    assert len(cards) == 2


def test_a_merged_card_inherits_the_split_that_ended_it():
    cards = bq.merge_cards([utt(1, "ob der Pool...", 5.0, split="soft_max"),
                            utt(2, "erreicht.", 7.0, split="pause")])
    assert cards[0]["split"] == "pause"


def test_empty_transcripts_are_dropped_not_merged():
    cards = bq.merge_cards([utt(1, "Ja, genau.", 5.0), utt(2, "   ", 7.0),
                            utt(3, "Und dann?", 9.0)])
    assert [c["text"] for c in cards] == ["Ja, genau.", "Und dann?"]


def test_merge_uses_the_servers_own_rule(monkeypatch):
    """If the tool ever grows its own copy of looks_finished, the thing being
    measured stops being the thing that ships."""
    monkeypatch.setattr(srv, "looks_finished", lambda text: True)
    cards = bq.merge_cards([utt(1, "ob der Pool...", 5.0),
                            utt(2, "erreicht.", 7.0)])
    assert len(cards) == 2


# --------------------------------------------- the sized-but-unbuilt variant

def test_lowercase_variant_merges_across_a_full_stop():
    """The evidence sits on the CURRENT chunk, which is exactly why the
    punctuation-only rule cannot see it."""
    rows = [utt(1, "da haben sie Kies in dem Bereich des Gartens.", 5.0),
            utt(2, "ausgestreut und mir fiel das auf.", 7.0)]
    assert len(bq.merge_cards(rows)) == 2
    joined = bq.merge_cards(rows, lowercase_continues=True)
    assert len(joined) == 1
    assert joined[0]["text"].endswith("Gartens. ausgestreut und mir fiel das auf.")


def test_lowercase_variant_still_respects_the_gap():
    rows = [utt(1, "in New Mexico.", 5.0), utt(2, "bei Freunden.", 10.0)]
    assert len(bq.merge_cards(rows, lowercase_continues=True)) == 2


# ------------------------------------------------------------- the metrics

def test_ends_metric_excludes_the_casing_rule():
    card = {"label": "COMPLETE", "text": "ausgestreut und mir fiel das auf."}
    assert bq.is_broken(card, "ends") is False
    assert bq.is_broken(card, "broken") is True


def test_broken_metric_still_catches_a_judged_fragment():
    card = {"label": "MISSING_VERB", "text": "Der Vermieter hat gesagt, wenn wir..."}
    assert bq.is_broken(card, "ends") is True
    assert bq.is_broken(card, "broken") is True


def test_scoring_the_lowercase_variant_with_the_casing_metric_is_refused(tmp_path):
    """Otherwise the intervention is measured against itself and always wins."""
    dump = tmp_path / "d.jsonl"
    dump.write_text(json.dumps(utt(1, "Ja, genau.", 5.0)) + "\n")
    rc = bq.main(["--dump", str(dump), "--merge", "lowercase",
                  "--metric", "broken"])
    assert rc == 2


# --------------------------------------------------------------- the judge

class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps({"response": self.payload}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_ollama(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout=0):
        sent["body"] = json.loads(req.data)
        return FakeResponse(sent["reply"])

    monkeypatch.setattr(bq.urllib.request, "urlopen", fake_urlopen)
    return sent


def test_judge_reads_a_json_label(fake_ollama):
    fake_ollama["reply"] = '{"label": "MISSING_VERB", "missing": "wollten"}'
    assert bq.judge("Der Vermieter hat gesagt, wenn wir...") == "MISSING_VERB"


def test_judge_falls_back_to_a_bare_label(fake_ollama):
    fake_ollama["reply"] = "I think this one is COMPLETE, actually."
    assert bq.judge("Ja, genau.") == "COMPLETE"


def test_an_answer_outside_the_label_set_is_not_silently_a_vote(fake_ollama):
    fake_ollama["reply"] = '{"label": "PROBABLY_FINE"}'
    assert bq.judge("Ja, genau.") == "UNPARSED"


def test_judge_sends_the_utterance_and_pins_temperature(fake_ollama):
    fake_ollama["reply"] = '{"label": "COMPLETE"}'
    bq.judge("Der Pool hier ist ja so ein L.", model="m")
    body = fake_ollama["body"]
    assert "Der Pool hier ist ja so ein L." in body["prompt"]
    assert body["model"] == "m"
    assert body["options"]["temperature"] == 0
    assert body["stream"] is False


# ---------------------------------------------------------- the arithmetic

def test_wilson_brackets_the_observed_rate():
    pct, (lo, hi) = bq.wilson(217, 426)
    assert pct == pytest.approx(50.9, abs=0.1)
    assert lo < pct < hi
    assert (lo, hi) == pytest.approx((46.2, 55.7), abs=0.2)


def test_wilson_stays_inside_the_unit_interval_at_the_edges():
    _, (lo, hi) = bq.wilson(0, 30)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert 0 < hi < 100
    assert bq.wilson(0, 0) == (0.0, (0.0, 0.0))


def test_the_measured_null_is_reported_as_a_null():
    """41.1% of 316 against 38.7% of 75 -- the number this tool was built for."""
    assert bq.two_proportion_p(130, 316, 29, 75) > 0.5


def test_a_real_difference_is_not_reported_as_a_null():
    assert bq.two_proportion_p(90, 100, 10, 100) < 0.001


def test_identical_proportions_give_p_one():
    assert bq.two_proportion_p(50, 100, 25, 50) == pytest.approx(1.0)


def test_report_prints_the_minimum_detectable_difference(capsys):
    """A null next to no power claim is how a small effect gets buried."""
    cards = ([{"label": "COMPLETE", "text": "Ja.", "split": "soft_max"}] * 10
             + [{"label": "MISSING_VERB", "text": "weil...", "split": "pause"}] * 10)
    bq.report(cards, "ends")
    out = capsys.readouterr().out
    assert "smallest difference this many cards could detect" in out
    assert "p=" in out
