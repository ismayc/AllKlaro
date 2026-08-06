"""Typed-text language detection: accuracy floors, the specific failures
that prompted the rewrite, and behaviour when py3langid isn't installed."""
import builtins

import pytest

import server
from server import detect_language, detect_language_scored
from fixtures.detect_phrases import PHRASES, TRAPS, cases


def accuracy(data, detect=detect_language):
    """(fraction correct, list of misses) over every phrase/pair combination."""
    misses = [(pair, want, text) for pair, want, text in cases(data)
              if detect(text, pair) != want]
    total = len(cases(data))
    return (total - len(misses)) / total, misses


def test_realistic_phrases_are_detected():
    # Measured 99.0% when this landed; the floor leaves room for a model
    # update to shuffle a case or two without going red for no reason.
    rate, misses = accuracy(PHRASES)
    assert rate >= 0.96, f"{rate:.1%} — misses: {misses}"


def test_cross_language_function_words_do_not_flip_the_answer():
    """Sentences stuffed with the other language's function words. These are
    what stops the word-list vote from being weighted any higher."""
    rate, misses = accuracy(TRAPS)
    assert rate >= 0.93, f"{rate:.1%} — misses: {misses}"


@pytest.mark.parametrize("text", [
    "Happy birthday!", "Merry Christmas!", "Congratulations!", "Well done",
    "Take care", "Sleep well", "Hello there",
])
@pytest.mark.parametrize("pair", [("de", "en"), ("es", "en")])
def test_greetings_without_function_words_are_english(text, pair):
    """The bug this replaced: no listed word in any language meant a 0-0 tie,
    and the tie went to the pair's first language. "Happy birthday!" came out
    as German — in Berlinerisch, if a dialect was selected."""
    assert detect_language(text, pair) == "en"


def test_exclusive_orthography_settles_it_outright():
    for text in ("Die Straße ist gesperrt", "Ich möchte einen Kaffee"):
        assert detect_language_scored(text, ("de", "en")) == ("de", 1.0)
    for text in ("¿Dónde estás?", "¡Feliz cumpleaños!"):
        assert detect_language_scored(text, ("es", "en")) == ("es", 1.0)


def test_umlaut_does_not_prove_german_when_both_candidates_use_one():
    """ü is decisive against English, but Spanish has it too ("pingüino"),
    so in an es-de pair it must not short-circuit the model."""
    _, marked = server._lexical_vote("pingüino", ("es", "de"))
    assert marked is None


@pytest.mark.parametrize("text", ["Bis morgen", "Und sonst?", "Danke dir"])
def test_function_words_overrule_a_hedging_model(text):
    """The model calls each of these English at 0.93–0.96 — confident enough
    to look decided, not confident enough to outrank a German function word.
    This override is the whole reason the word lists survived the rewrite."""
    lang, conf = detect_language_scored(text, ("de", "en"))
    assert lang == "de"
    assert conf < 1.0            # rescued, not certain


def test_confidence_is_lower_when_the_signals_disagree():
    sure, _ = accuracy(PHRASES)
    strong = detect_language_scored("Where are you going tomorrow?", ("de", "en"))
    weak = detect_language_scored("Cheers", ("de", "en"))
    assert strong[0] == weak[0] == "en"
    assert strong[1] > weak[1]
    # The UI only marks a chip "close call" below this.
    assert strong[1] >= server.UNSURE_BELOW


def test_unsure_marking_stays_rare_on_correct_answers():
    """The dashed chip is a nudge, not a nag: if most correct detections were
    flagged it would train the eye to ignore it."""
    flagged = sum(1 for pair, want, text in cases(PHRASES)
                  if detect_language_scored(text, pair)[1] < server.UNSURE_BELOW)
    assert flagged / len(cases(PHRASES)) < 0.15


def test_falls_back_to_word_lists_without_py3langid(monkeypatch):
    """The package is a dependency, but a stale venv must degrade to the old
    behaviour rather than 500 on every typed message."""
    real_import = builtins.__import__

    def no_py3langid(name, *args, **kwargs):
        if name.startswith("py3langid"):
            raise ImportError("py3langid missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_py3langid)
    monkeypatch.setattr(server, "_langid_identifier", None)
    assert server._langid_vote("Guten Morgen", ("de", "en")) == (None, 0.0)
    # Function words and umlauts still carry it most of the way.
    rate, _ = accuracy(PHRASES)
    assert rate >= 0.80
    assert detect_language("Wie geht es dir und der Familie?", ("de", "en")) == "de"


def test_model_failure_does_not_break_detection(monkeypatch, caplog):
    """A corrupt model file should cost accuracy, never the typed message."""
    class Boom:
        def set_languages(self, langs):
            raise RuntimeError("model file is garbage")

    monkeypatch.setattr(server, "_langid_identifier", Boom())
    assert server._langid_vote("Guten Morgen", ("de", "en")) == (None, 0.0)
    assert detect_language("Das ist wirklich gut", ("de", "en")) == "de"


def test_unknown_input_is_deterministic():
    """Nothing to go on: still a stable answer, not a coin flip per call."""
    first = detect_language_scored("Xylophon 4711", ("de", "en"))
    assert first == detect_language_scored("Xylophon 4711", ("de", "en"))
