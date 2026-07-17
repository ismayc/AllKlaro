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
