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


def test_every_message_the_client_sends_is_handled_by_the_server():
    """The websocket contract, checked in the direction a browser would.

    A typo on either side is silent: the server ignores the message and the
    button simply does nothing, which looks like a broken feature rather than
    a broken name. Nothing else in the suite would catch it, because the
    server tests construct their own payloads.
    """
    server_py = (Path(__file__).parent.parent / "server.py").read_text()
    sent = set(re.findall(r'type: "([a-z_]+)"', JS))
    handled = set(re.findall(r'cfg\.get\("type"\) == "([a-z_]+)"', server_py))
    assert sent, "no outbound message types found — did app.js move?"
    assert sent <= handled, \
        f"app.js sends message types the server ignores: {sent - handled}"


def test_every_message_the_client_handles_is_one_the_server_sends():
    server_py = (Path(__file__).parent.parent / "server.py").read_text()
    handled = set(re.findall(r'msg\.type === "([a-z_]+)"', JS))
    sent = set(re.findall(r'"type": "([a-z_]+)"', server_py))
    assert handled, "no inbound message types found — did app.js move?"
    assert handled <= sent, \
        f"app.js handles messages the server never sends: {handled - sent}"


def test_the_improve_tap_sends_what_the_server_reads():
    """The card is re-translated from text the *client* sends back, the same
    way the language-chip flip works, so a card older than the server's
    context window still improves. That only holds if the field names line
    up."""
    call = re.search(r'ws\.send\(JSON\.stringify\(\{ type: "improve"[^)]*\)',
                     JS, re.S)
    assert call, "app.js no longer sends an improve message"
    for field in ("id", "target", "text", "source"):
        assert field in call.group(), f"improve message drops `{field}`"


def test_the_voice_break_is_drawn_from_the_servers_field():
    server_py = (Path(__file__).parent.parent / "server.py").read_text()
    # \b, not a substring test: `msg.voice_changed` contains `msg.voice_change`
    # and would sail past a plain `in` check while reading a field the server
    # never sends.
    assert re.search(r"\bmsg\.voice_change\b", JS), \
        "app.js ignores the voice-change flag"
    assert re.search(r'"voice_change":', server_py), \
        "the server no longer sends voice_change"
    assert '"voice-break"' in JS
    # A break, not an identity: the mark must not claim to name anyone.
    assert "new voice" in JS


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


def test_wrong_language_can_be_corrected_from_the_card():
    """A misdetected card is fixed by tapping its language chip, so the chip
    has to be a real button — and must not also fire tap-to-speak."""
    css = (STATIC / "style.css").read_text()
    assert "sourceChip" in JS and ".lang.flip" in css
    assert re.search(r"btn\.onclick[^}]*stopPropagation\(\);\s*flipCard", JS)
    assert '"retranslate"' in JS            # server redoes it the other way
    # Close calls are marked, and touch has no hover to reveal affordances.
    assert "unsure" in JS and ".lang.flip.unsure" in css
    assert "UNSURE_BELOW" in JS


def test_typed_language_can_be_pinned():
    assert 'id="pinBtn"' in HTML
    assert "#pinBtn" in (STATIC / "style.css").read_text()
    assert re.search(r"pinBtn\.onclick", JS)
    assert "pinnedSource," in JS            # survives a reload
    # The pin only exists for auto modes, and re-checks when the mode changes.
    assert re.search(r"modeSel\.onchange[^;]*renderPin\(\)", JS)


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


def test_translations_are_copyable_without_selecting_text():
    css = (STATIC / "style.css").read_text()
    assert "copy-btn" in JS and ".copy-btn" in css
    # One tap copies via the async clipboard API, with an execCommand
    # fallback for plain-http LAN contexts where that API is unavailable.
    assert "navigator.clipboard?.writeText" in JS
    assert 'document.execCommand("copy")' in JS
    # The tap must not also speak the row or open the big-text overlay.
    assert re.search(r"copyBtn\.onclick[^;]*stopPropagation", JS)
    # Brief visual confirmation, then the button returns to normal.
    assert ".copy-btn.copied" in css
    assert re.search(r'btn\.textContent = "✓"', JS)
    assert re.search(r'btn\.textContent = "📋"', JS)


def test_words_are_lookupable_on_long_press():
    css = (STATIC / "style.css").read_text()
    assert 'id="lookup"' in HTML and 'id="lookupBody"' in HTML
    assert "#lookup.hidden" in css and ".lookup-card" in css
    assert "/api/lookup" in JS
    # Both the original line and finished translation rows get word spans.
    # Matched loosely on the arguments: the heard line also passes the dialect
    # set, so pinning the exact call text made this fail on a change that did
    # not touch lookup at all.
    assert re.search(r"wordSpans\(\s*msg\.text,\s*msg\.source", JS)
    assert "wordSpans(row.text, target)" in JS
    # A long-press must not also fire tap-to-speak or the big-text overlay,
    # and iOS's text-selection callout must not fight the gesture.
    assert "attachLongPress" in JS and "pointerdown" in JS
    assert "suppressClick" in JS
    # A long-press with no follow-up click (common on iOS) must not leave
    # the flag armed to swallow the next genuine tap.
    assert re.search(r'"pointerdown", \(\) => \{ suppressClick = false', JS)
    assert "-webkit-touch-callout" in css
    # Missing lexicons surface the server's how-to-build message, not a crash.
    assert re.search(r"data\.error", JS)


def test_row_buttons_are_finger_visible_on_touch_screens():
    # Touch has no hover-reveal: 12px at 45% opacity made the silver
    # clipboard emoji effectively invisible on iPhone screens.
    #
    # Written against the button classes rather than one exact selector list,
    # so a new row button has to meet the same bar instead of quietly
    # slipping past by not being named here.
    css = (STATIC / "style.css").read_text()
    mobile = css[css.index("@media (max-width: 640px)"):]
    js = (STATIC / "app.js").read_text()
    classes = set(re.findall(r'className = "(copy-btn|edit-btn|improve-btn)"', js))
    assert classes == {"copy-btn", "edit-btn", "improve-btn"}, \
        f"row buttons in app.js not covered by this test: {classes}"
    for cls in sorted(classes):
        rules = [m.group() for m in re.finditer(r"\.[^{}]*\{[^}]*\}", css, re.S)
                 if f".{cls}" in m.group().split("{")[0]]
        sized = [r for r in rules if "font-size" in r]
        assert sized, f"{cls} sets no font-size at all"
        assert re.search(r"font-size:\s*1[6-9]px", sized[0]), f"{cls} too small"
        # Every rule, not just the first: CSS cascades, so a later
        # `.improve-btn { font-size: 11px }` would quietly undo the one above
        # and a check that stops at the first match would never see it.
        for rule in rules:
            base = rule if "@media" not in rule else ""
            for size in re.findall(r"font-size:\s*(\d+)px", base):
                assert int(size) >= 16, f"{cls} shrunk to {size}px by: {rule}"
            for opacity in re.findall(r"opacity:\s*\.(\d)", base):
                # ...except the disabled state, which is meant to look
                # unavailable rather than merely faint.
                if ":disabled" not in rule.split("{")[0]:
                    assert int(opacity) >= 5, f"{cls} too faint: {rule}"
        assert re.search(rf"\.{cls}[^{{}}]*\{{[^}}]*font-size:\s*1[89]px",
                         mobile), f"{cls} not enlarged on phones"


def test_asset_urls_are_cache_busted_and_build_is_visible():
    assert 'href="/static/style.css?v=__BUILD__"' in HTML
    assert 'src="/static/app.js?v=__BUILD__"' in HTML
    # The About panel shows the running build, so a stale phone cache is
    # diagnosable instead of surfacing as "the feature disappeared".
    assert 'id="buildStamp"' in HTML
    assert "buildStamp" in JS and 'searchParams.get("v")' in JS


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


def test_german_flavor_and_paste_are_wired():
    assert 'id="deFlavor"' in HTML
    assert 'value="berlin"' in HTML and 'value="hessian"' in HTML
    assert 'value="worms"' in HTML
    assert re.search(r"de_flavor:\s*flavorSel\.value", JS)  # reaches the server
    assert re.search(r"deFlavor:\s*flavorSel\.value", JS)   # persisted setting
    assert 'id="esFlavor"' in HTML
    assert 'value="mexico"' in HTML and 'value="barcelona"' in HTML
    assert re.search(r"es_flavor:\s*esFlavorSel\.value", JS)
    assert re.search(r"esFlavor:\s*esFlavorSel\.value", JS)
    # The you-form control covers German and Spanish targets.
    assert 'id="address"' in HTML
    for v in ("informal", "formal", "plural"):
        assert f'value="{v}"' in HTML
    assert re.search(r"address:\s*addressSel\.value", JS)
    # Paste-and-translate: one tap from a WhatsApp copy to a translation,
    # falling back to a focused input if clipboard access is refused.
    assert 'id="pasteBtn"' in HTML
    assert "clipboard.readText" in JS
    assert "requestSubmit" in JS
    assert re.search(r"catch[^}]*typeInput\.focus", JS, re.S)


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


def test_type_input_does_not_trigger_ios_auto_zoom():
    # iOS Safari zooms into any focused input under 16px and stays zoomed
    # after the keyboard closes; the type bar and (on phones) the selects
    # must be at least 16px.
    css = (STATIC / "style.css").read_text()
    typebar = re.search(r"#typeBar input \{[^}]*\}", css, re.S).group()
    assert re.search(r"font-size:\s*1[6-9]px", typebar), \
        "type input under 16px re-enables the iOS focus zoom"
    mobile = css[css.index("@media (max-width: 640px)"):]
    assert re.search(r"select \{[^}]*font-size:\s*1[6-9]px", mobile)


def test_launchers_auto_reload_server_code():
    root = Path(__file__).parent.parent
    for name in ("Start AllKlaro.command", "Start AllKlaro (iPhone).command",
                 "Start AllKlaro (Anywhere).command", "allklaroctl"):
        assert "--reload" in (root / name).read_text(), f"{name} lost --reload"


def test_remote_control_script_is_headless_ready():
    ctl = (Path(__file__).parent.parent / "allklaroctl").read_text()
    assert '"/opt/homebrew/bin' in ctl   # non-interactive SSH has a bare PATH
    assert "nohup" in ctl                # server outlives the SSH session
    for verb in ("start", "stop", "restart", "status"):
        assert f"{verb})" in ctl
    # Local .command runs have no menu bar clicker either. (This cannot
    # rescue an SSH start — that arrives through the tunnel, so Tailscale
    # was necessarily already up. See test_readme_documents_vpn_prereq.)
    assert "open -a Tailscale" in ctl


def test_pipeline_overlay_is_wired_and_off_by_default():
    app_js = (STATIC / "app.js").read_text()
    assert 'id="statsChk"' in HTML and 'id="pipeline"' in HTML
    assert 'id="pipeline" class="hidden"' in HTML   # opt-in, not always-on
    # Toggling it must tell the server, or no stats are ever pushed.
    assert "stats: statsChk.checked" in app_js
    assert 'msg.type === "stats"' in app_js
    for field in ("pipeSpeech", "pipeFlight", "pipeQueue", "pipeSkipped",
                  "pipeChunk", "pipeLag"):
        assert f'id="{field}"' in HTML, f"overlay lost {field}"


def test_pipeline_overlay_never_blocks_the_conversation():
    """It floats over the feed, so it must be click-through and opaque enough
    to read against card text underneath."""
    css = (STATIC / "style.css").read_text()
    block = re.search(r"#pipeline \{[^}]*\}", css, re.S).group()
    assert "pointer-events: none" in block
    alpha = float(re.search(r"rgba\(16, 20, 26, \.(\d+)\)", block).group(1)) / 100
    assert alpha >= 0.9, "overlay too transparent — page text bleeds through"


def test_anywhere_mode_serves_loopback_only():
    """Why the LAN Shortcut URL dies the moment you switch to anywhere mode.

    Both anywhere-mode entry points bind uvicorn to 127.0.0.1 and publish
    it through `tailscale serve`, so nothing answers on <mac-ip>:8710 —
    only the phone-mode launcher opens the LAN interface.
    """
    root = Path(__file__).parent.parent
    for name in ("Start AllKlaro (Anywhere).command", "allklaroctl"):
        text = (root / name).read_text()
        assert "--host 127.0.0.1" in text, f"{name} no longer loopback-only"
        assert "serve --bg http://127.0.0.1:8710" in text, \
            f"{name} lost the Tailscale proxy that replaces the LAN address"
    assert "--host 0.0.0.0" in (root / "Start AllKlaro (iPhone).command").read_text()


def _shortcut_section():
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    sect = readme[readme.index("### 📲 iOS Shortcut:"):]
    return sect[:sect.index("## 🌍 Anywhere mode (Tailscale)")]


def test_readme_warns_the_lan_shortcut_url_breaks_in_anywhere_mode():
    """A stale LAN URL breaks every launcher of the shortcut (Back Tap
    included) and reads like a dead server, so the fix must be findable
    from the Shortcut section, not only from the anywhere-mode one."""
    sect = _shortcut_section()
    assert "anywhere mode" in sect         # the URL step points at it
    fix = sect[sect.index("**Troubleshooting:**"):]
    assert "ts.net/api/translate" in fix   # names the URL to switch to
    assert "127.0.0.1" in fix              # and why the LAN one refuses


def _phone_start_section():
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    steps = readme[readme.index("### 🚦 Start the server from your phone"):]
    return steps[:steps.index("## ⚙️ Under the hood")]


def test_readme_documents_vpn_prereq():
    """The shortcut can't heal a down tunnel, so it must connect first itself."""
    steps = _phone_start_section()
    assert "can't repair itself" in steps
    # Tailscale's own Connect action has to run before the SSH action.
    assert steps.index("**Connect**") < steps.index("**Run Script Over SSH**")


def test_readme_ssh_host_is_the_tailscale_ip():
    """Shortcuts' SSH client can't resolve MagicDNS; the 100.x IP can."""
    steps = _phone_start_section()
    assert "tailscale ip -4" in steps
    host = steps[steps.index("**Host:**"):]
    host = host[:host.index("**Port:**")]
    assert "**Tailscale IP**" in host and "100.x.y.z" in host
    assert "Use the number" in host


def test_home_screen_web_app_is_configured():
    assert 'rel="apple-touch-icon" href="/static/icon-180.png"' in HTML
    assert (STATIC / "icon-180.png").exists()
    assert 'apple-mobile-web-app-capable' in HTML   # standalone, no browser UI
    assert 'apple-mobile-web-app-title" content="AllKlaro"' in HTML


def test_spanish_speech_uses_latin_american_voice():
    assert '"es-MX"' in JS
    assert '"es-ES"' not in JS
