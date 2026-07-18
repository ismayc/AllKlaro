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
    for ref in re.findall(r'/static/([\w.-]+)', HTML):
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


def test_translations_are_editable():
    css = (STATIC / "style.css").read_text()
    assert "edit-btn" in JS and ".edit-btn" in css
    assert "edit-box" in JS and ".edit-box" in css
    # The editor must not trigger tap-to-speak or the big-text overlay.
    assert re.search(r"box\.onclick[^;]*stopPropagation", JS)
    # Saved edits go to the server (persisted) and over the socket (context).
    assert "/api/correction" in JS
    assert re.search(r'type:\s*"correction"', JS)
    # Escape cancels; Enter saves; an edited row is visibly marked.
    assert "Escape" in JS
    assert 'class="edited"' in JS


def test_about_panel_links_to_site_and_repo():
    assert 'id="aboutBtn"' in HTML and 'id="about"' in HTML
    assert "https://ismayc.github.io/AllKlaro/" in HTML
    assert "https://github.com/ismayc/AllKlaro" in HTML
    assert "#about.hidden" in (STATIC / "style.css").read_text()
    # Both the Close button and a backdrop tap dismiss it.
    assert "aboutClose" in JS
    assert re.search(r"e\.target === aboutBox", JS)


def test_phrases_speak_on_tap_in_their_own_language():
    # Original speaks in the source language, each translation row in its
    # target language; both stop propagation so the big-text view isn't
    # triggered by the same tap.
    assert re.search(r"orig\.onclick.*stopPropagation", JS, re.S)
    assert "speakText(msg.text, msg.source" in JS
    assert re.search(r"row\.onclick[^}]*rows\[t\]\.text, t", JS, re.S)
    # A tap interrupts any queued auto-speech instead of waiting behind it.
    assert "speechSynthesis.cancel()" in JS


def test_draft_model_control_is_wired():
    assert 'id="draftModel"' in HTML
    assert "draft_model" in JS                 # reaches the server config
    assert re.search(r"draft:\s*draftSel\.value", JS)  # persisted setting
    # Dropdowns show sizes so the speed/quality tradeoff is visible, and the
    # auto-picked draft skips 3B-class models that paraphrase too freely.
    assert "MIN_DRAFT_BYTES" in JS
    assert re.search(r"GB", JS)
    css = (STATIC / "style.css").read_text()
    # The refining hint and the revision flash are both styled.
    assert ".refining" in css and ".revised" in css
    assert "translation_revised" in JS
    # A refinement never overwrites a translation the user already edited.
    assert re.search(r"translation_revised.*?row\.edited.*?continue", JS, re.S)


def test_typed_input_is_wired():
    assert 'id="typeBar"' in HTML and 'id="typeInput"' in HTML
    assert re.search(r'type:\s*"text"', JS)     # reaches the server
    assert "typeForm.onsubmit" in JS
    # Typing must work without the mic: the socket connects on demand.
    assert re.search(r"typeForm\.onsubmit[\s\S]*?connectWS\(\)", JS)
    assert "#typeBar" in (STATIC / "style.css").read_text()


def test_layout_fits_mobile_viewport():
    css = (STATIC / "style.css").read_text()
    # Mobile Safari's 100vh hides the bottom bars behind its toolbar.
    assert "100dvh" in css
    assert "safe-area-inset-bottom" in css
    assert "viewport-fit=cover" in HTML


def test_launchers_auto_reload_server_code():
    root = Path(__file__).parent.parent
    for name in ("Start AllKlaro.command", "Start AllKlaro (Phone).command"):
        assert "--reload" in (root / name).read_text(), f"{name} lost --reload"


def test_home_screen_web_app_is_configured():
    assert 'rel="apple-touch-icon" href="/static/icon-180.png"' in HTML
    assert (STATIC / "icon-180.png").exists()
    assert 'apple-mobile-web-app-capable' in HTML   # standalone, no browser UI
    assert 'apple-mobile-web-app-title" content="AllKlaro"' in HTML


def test_spanish_speech_uses_latin_american_voice():
    assert '"es-MX"' in JS
    assert '"es-ES"' not in JS
