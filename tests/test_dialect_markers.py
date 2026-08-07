"""Colouring dialect words in the heard text.

The measurement that shaped this: over 2267 German word tokens of the real
recording the dialect lexicon matched only 14 times, every hit an *ambiguous*
entry ("mehr" ×10, "des" ×4) and none of them Berlinerisch. Whisper normalises
dialect to standard orthography, so the honest design is to mark only
unambiguous forms — dark on the audio path, useful on typed input — rather
than paint ordinary German red to look busy.
"""
import json

import pytest

import server as srv
from conftest import collect_until, speak


def test_unambiguous_berlin_forms_are_marked():
    got = srv.dialect_markers("Ick hab keene Ahnung, wat dit soll", "de")
    assert set(got) == {"ick", "keene", "wat", "dit"}


def test_ordinary_german_is_never_marked():
    """The whole point. "mehr" and "des" are the only lexicon hits in 2267
    real tokens and both are ambiguous — marking them would put red on plain
    speech in a recording with no detectable dialect at all."""
    assert srv.dialect_markers("Ich möchte nicht mehr, des ist genug", "de") == []
    assert srv.dialect_markers("Und wie warm ist es bei dir?", "de") == []


def test_a_selected_dialect_narrows_which_forms_count(monkeypatch):
    """Marking a Rhine-Hessian form for someone listening to a Berliner would
    be a new error, not a feature.

    Tested against a controlled lexicon on purpose: today *every* unambiguous
    entry in dialects.txt is untagged, so this branch is inert on the real
    data. Covering it here means the filter is known-good the day someone
    tags an entry, instead of shipping untested."""
    monkeypatch.setattr(srv, "_dialects_cache", {"mtime": None, "map": {}})
    monkeypatch.setattr(srv, "load_dialects", lambda: {"de": {
        "keene":  ("keine", False, frozenset({"berlin"})),
        "nochemol": ("nochmal", False, frozenset({"hessian", "worms"})),
        "ick":    ("ich", False, None),          # applies to any dialect
    }})
    text = "ick sag keene nochemol"
    assert srv.dialect_markers(text, "de", flavor="berlin") == ["ick", "keene"]
    assert srv.dialect_markers(text, "de", flavor="hessian") == ["ick", "nochemol"]
    assert srv.dialect_markers(text, "de") == ["ick", "keene", "nochemol"]


def test_every_unambiguous_entry_is_currently_untagged():
    """Documents why the filter above is inert, so a future reader does not
    conclude the narrowing is broken. Delete this test when entries get
    tagged — its failure is the signal that the data caught up."""
    lexicon = srv.load_dialects()["de"]
    tagged = [t for t, e in lexicon.items() if not e[1] and e[2]]
    assert tagged == [], f"dialects.txt now tags {tagged} — enable narrowing"


def test_each_word_is_reported_once():
    """The client builds a Set from this, but a repeated token would still be
    wasted bytes on every card."""
    assert srv.dialect_markers("Ick sag ick, und nochmal ick", "de") == ["ick"]


def test_unknown_language_is_not_an_error():
    assert srv.dialect_markers("hello there", "fr") == []


def test_markers_ride_along_with_the_typed_final(client):
    """Typed input is the path this actually fires on, so it is the one worth
    testing end to end."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "de-en"}))
        ws.send_text(json.dumps({"type": "text",
                                 "text": "Ick hab keene Ahnung"}))
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    assert set(final["dialect"]) == {"ick", "keene"}


def test_a_spoken_final_carries_the_field_even_when_empty(client,
                                                          stub_transcribe):
    """Absent and empty must not be the same thing to the client: the field is
    always present so a missing one means a server too old to send it."""
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    assert final["dialect"] == []      # "Wie geht es dir?" is standard German
