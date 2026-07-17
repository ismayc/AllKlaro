"""Consistency checks between the frontend files — catches wiring mistakes
(JS looking up an element the HTML doesn't define) without a browser."""
import re
from pathlib import Path

STATIC = Path(__file__).parent.parent / "static"
HTML = (STATIC / "index.html").read_text()
JS = (STATIC / "app.js").read_text()


def test_every_js_element_lookup_exists_in_html():
    looked_up = set(re.findall(r'getElementById\("([^"]+)"\)', JS))
    assert looked_up, "no lookups found — did app.js move?"
    defined = set(re.findall(r'id="([^"]+)"', HTML))
    missing = looked_up - defined
    assert not missing, f"app.js references ids missing from index.html: {missing}"


def test_html_references_existing_static_assets():
    for ref in re.findall(r'/static/([\w.]+)', HTML):
        assert (STATIC / ref).exists(), f"index.html references missing {ref}"
    # The audio worklet is loaded from JS, not HTML.
    assert "worklet.js" in JS and (STATIC / "worklet.js").exists()


def test_controls_are_collapsible():
    assert 'id="controls"' in HTML and 'id="controlsBtn"' in HTML
    assert "#controls.hidden" in (STATIC / "style.css").read_text()
    assert "controlsHidden" in JS  # collapsed state persists across reloads


def test_focus_mode_centers_latest_text():
    assert 'id="focusChk"' in HTML
    css = (STATIC / "style.css").read_text()
    assert "#feed.focus" in css          # bottom padding pushes newest to middle
    assert "card partial" in JS          # incoming text becomes an in-feed card
    assert re.search(r'focus:\s*focusChk\.checked', JS)  # persisted setting
    # Every partial/discard path uses the shared clear so no stale live text.
    assert JS.count("clearPartial()") >= 4


def test_conversation_can_be_cleared():
    assert 'id="clearBtn"' in HTML
    # Confirmation guards against an accidental mid-call tap wiping everything.
    assert re.search(r"clearBtn\.onclick[^}]*confirm\(", JS, re.S)
    assert "cards.clear()" in JS            # card state resets, not just the DOM
    assert re.search(r'lastSummary = ""', JS)  # stale summary won't leak into export
    assert "replaceChildren(hint)" in JS    # the initial hint returns to the feed
    # Queued auto-speech from the old conversation stops too.
    assert re.search(r"clearBtn\.onclick[^}]*speechSynthesis\?\.cancel", JS, re.S)


def test_phrases_speak_on_tap_in_their_own_language():
    # Original speaks in the source language, each translation row in its
    # target language; both stop propagation so the big-text view isn't
    # triggered by the same tap.
    assert re.search(r"orig\.onclick.*stopPropagation", JS, re.S)
    assert "speakText(msg.text, msg.source" in JS
    assert re.search(r"row\.onclick[^}]*rows\[t\]\.text, t", JS, re.S)
    # A tap interrupts any queued auto-speech instead of waiting behind it.
    assert "speechSynthesis.cancel()" in JS
