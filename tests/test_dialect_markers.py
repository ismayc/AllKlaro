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

    Kept against a controlled lexicon even though the real data is now tagged:
    it pins the *filter*, so a later edit to dialects.txt cannot make this pass
    for the wrong reason."""
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


def test_the_narrowing_is_live_on_the_real_lexicon():
    """The data caught up with the filter: selecting a dialect now changes
    what gets coloured, which it did not while every entry was untagged."""
    text = "Ick hab keene Ahnung, wat dit soll, gell"
    assert srv.dialect_markers(text, "de", "berlin") == ["ick", "keene", "wat", "dit"]
    assert srv.dialect_markers(text, "de", "hessian") == ["gell"]
    # No selection still marks everything unambiguous, whatever dialect.
    assert srv.dialect_markers(text, "de") == ["ick", "keene", "wat", "dit", "gell"]


def test_forms_shared_by_two_dialects_are_marked_under_both():
    """Under-tagging is the silent failure here — a Hessian form tagged only
    [hessian] stops being marked the moment someone picks Wormser Platt, even
    though the word is theirs too."""
    for flavor in ("hessian", "worms"):
        assert srv.dialect_markers("Isch hab ebbes net verstanne", "de",
                                   flavor) == ["isch", "ebbes", "net",
                                               "verstanne"]
    # ...and the Berlin-only reading of the same sentence marks nothing.
    assert srv.dialect_markers("Isch hab ebbes net verstanne", "de",
                               "berlin") == []


def test_wormser_forms_do_not_fire_for_a_frankfurt_speaker():
    """-scht for -st is Rhine Franconian, not Hessisch: the two are separate
    entries in the style selector and must behave that way."""
    text = "bischt du dabber, hoscht du Dorschd"
    assert srv.dialect_markers(text, "de", "worms") == [
        "bischt", "dabber", "hoscht", "dorschd"]
    assert srv.dialect_markers(text, "de", "hessian") == []


def test_enclitic_contractions_are_deliberately_untagged():
    """"haste"/"kannste" are colloquial across the language rather than
    regional, so they stay untagged and keep marking under every selection.
    An untagged entry is the parser's "applies to all", not an oversight."""
    for flavor in ("berlin", "hessian", "worms", ""):
        assert srv.dialect_markers("haste kannste biste", "de", flavor) == [
            "haste", "kannste", "biste"]


def test_every_unambiguous_entry_is_tagged_or_a_known_contraction():
    """The tagging is complete, not partial. A new untagged dialect form would
    be marked under every style — including the ones it does not belong to —
    which is the error the tags exist to prevent."""
    untagged = {t for t, e in srv.load_dialects()["de"].items()
                if not e[1] and not e[2]}
    assert untagged == {"haste", "biste", "isset", "isses", "kannste"}, (
        "new untagged German entries: "
        f"{untagged - {'haste', 'biste', 'isset', 'isses', 'kannste'}}")


def test_no_entry_names_a_dialect_the_selector_does_not_offer():
    """A typo'd tag ("[berlain]") is invisible: the entry simply stops being
    marked under every real selection. Nothing else would catch it."""
    for lang, offered in (("de", srv.FLAVOR_NOTES["de"]),
                          ("es", srv.FLAVOR_NOTES["es"])):
        used = set()
        for _term, (_gloss, _amb, flavors) in srv.load_dialects()[lang].items():
            used |= set(flavors or ())
        assert used <= set(offered), (
            f"dialects.txt [{lang}] tags {used - set(offered)}, "
            f"which the style selector does not offer")


def test_spanish_entries_narrow_the_same_way():
    """The Spanish half had the same gap and the same fix — "mola" is
    Peninsular and glossing it for a Mexican speaker is the mirror error."""
    text = "qué chido, muy guay"
    assert srv.dialect_markers(text, "es", "mexico") == ["chido"]
    assert srv.dialect_markers(text, "es", "barcelona") == ["guay"]
    assert srv.dialect_markers(text, "es") == ["chido", "guay"]


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
