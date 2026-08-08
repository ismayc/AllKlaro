"""Skipping utterances the listener already understands.

The one lever tried that makes there be *less* work rather than dividing the
same work up differently — and measured on the real slice it bought no
latency at all. Kept because the product argument is separate from the
performance one: translating a sentence a learner understood removes the
reason to practise. Off unless a known-words file exists.
"""
import json

import pytest

import server as srv
from conftest import collect_until, speak


@pytest.fixture
def known(tmp_path, monkeypatch):
    def use(words, max_words=8):
        p = tmp_path / "known.txt"
        p.write_text("# comment\n" + "\n".join(words) + "\n")
        monkeypatch.setattr(srv, "KNOWN_WORDS_PATH", p)
        monkeypatch.setattr(srv, "_known_cache", {"mtime": None,
                                                  "words": frozenset()})
        monkeypatch.setattr(srv, "KNOWN_SKIP_MAX_WORDS", max_words)
        return p
    return use


def test_off_entirely_when_there_is_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(srv, "KNOWN_WORDS_PATH", tmp_path / "nope.txt")
    monkeypatch.setattr(srv, "_known_cache", {"mtime": None,
                                              "words": frozenset()})
    assert srv.is_already_understood("Ja.", "de") is False


def test_a_fully_known_short_sentence_is_skipped(known):
    known(["ja", "das", "ist", "gut"])
    assert srv.is_already_understood("Ja, das ist gut.", "de") is True


def test_one_unknown_word_is_enough_to_translate(known):
    """Conservative on purpose: the two errors are not symmetric. A needless
    translation costs a little time; a skipped one the listener needed costs
    them the sentence."""
    known(["ja", "das", "ist"])
    assert srv.is_already_understood("Ja, das ist Überschwemmung.", "de") is False


def test_a_long_sentence_is_never_skipped_however_familiar(known):
    """A sentence can be built entirely of known words and still say something
    the listener would not assemble in time. The cap is on the sentence, not
    the vocabulary."""
    known(["das", "ist", "auch", "noch", "nicht", "so", "ganz", "wie", "wir",
           "es", "wollten", "gewesen"], max_words=8)
    long = "Das ist auch noch nicht so ganz wie wir es wollten gewesen"
    assert len(long.split()) > 8
    assert srv.is_already_understood(long, "de") is False


def test_only_the_language_being_learned_is_skipped(known):
    """An English utterance is not something a German learner is practising,
    and this conversation is bilingual — English rises from 12% to 60%."""
    known(["yes", "that", "is", "good"])
    assert srv.is_already_understood("Yes that is good", "en") is False


def test_empty_text_is_not_understood(known):
    known(["ja"])
    assert srv.is_already_understood("", "de") is False
    assert srv.is_already_understood("...", "de") is False


def test_the_card_still_appears_with_no_translation(client, stub_transcribe,
                                                    known, monkeypatch):
    """The heard text is never withheld — only the translation. Without the
    card the listener cannot tell a skip from a dropped utterance."""
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    known(["wie", "geht", "es", "dir"])
    with client.websocket_connect("/ws") as ws:
        speak(ws)                       # stub transcribes "Wie geht es dir?"
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    assert final["text"] == "Wie geht es dir?"
    done = next(m for m in msgs if m["type"] == "translation_done")
    assert done["known"] is True and done["translate_ms"] == 0
    assert not any(m["type"] == "translation_delta" for m in msgs), \
        "the model was asked anyway"


def test_the_skip_is_recorded_in_the_trace(client, stub_transcribe, known,
                                           trace_file, monkeypatch):
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    known(["wie", "geht", "es", "dir"])
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        collect_until(ws)
    from conftest import trace_records
    rec = [r for r in trace_records(trace_file) if r.get("uid")][0]
    assert rec["skipped_known"] is True
    assert rec["outcome"] == "final", "a skip is not a discard"


def test_an_unknown_utterance_is_still_translated(client, stub_transcribe,
                                                  known, monkeypatch):
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    known(["hallo"])                     # does not cover the stub's sentence
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    assert any(m["type"] == "translation_delta" for m in msgs)
