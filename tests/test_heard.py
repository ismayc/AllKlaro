"""What was heard reaches the screen before Whisper gets to it.

Measured on a real 71-minute conversation (2026-10-04): a cut chunk waited a
median 6 s for its transcript and 15 s at p90. The only sight of the German in
that gap was the rolling live line, which had moved on, so the words vanished
and came back later inside a card further up. The server now sends a fast
preview of each chunk the moment it is cut, and the page shows it in the place
the card will take. These tests cover the server half and the pure placement
rule; the page itself is checked in a browser.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import server as srv
from conftest import collect_until, speak, trace_records

APP_JS = (Path(__file__).parent.parent / "static" / "app.js").read_text()


def types(msgs):
    return [m["type"] for m in msgs]


def test_a_cut_chunk_is_previewed_before_its_transcript(client, stub_transcribe,
                                                        stub_partial):
    stub_partial.text = "wie geht es dir"
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    heard = [m for m in msgs if m["type"] == "heard"]
    assert heard == [{"type": "heard", "id": 1, "text": "wie geht es dir"}]
    final = next(m for m in msgs if m["type"] == "final")
    assert heard[0]["id"] == final["id"]
    # The preview is for reading only: the card is still Whisper's transcript.
    assert final["text"] == "Wie geht es dir?"


def test_the_preview_decodes_the_whole_chunk_not_a_window(client,
                                                         stub_transcribe,
                                                         stub_partial,
                                                         trace_file):
    """A live partial looks at the last ~6 s. The preview is the chunk."""
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=80)
        collect_until(ws)
    chunk = trace_records(trace_file)[0]["chunk_sec"]
    assert chunk > srv.PARTIAL_WINDOW_FRAMES * srv.FRAME_MS / 1000
    assert max(stub_partial.calls) == pytest.approx(chunk, abs=0.05)


def test_no_fast_model_means_no_preview(client, stub_transcribe):
    """Without Parakeet a preview would be a second job on the Whisper thread
    ahead of the transcript it previews. The card simply arrives as before."""
    assert srv._parakeet_unavailable
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    assert "heard" not in types(msgs)
    assert "final" in types(msgs)


@pytest.mark.parametrize("text", ["", None, "Untertitel der Amara.org-Community"])
def test_an_empty_or_hallucinated_preview_is_not_sent(client, stub_transcribe,
                                                      stub_partial, text):
    """None is `transcribe_partial` saying the fast model went away."""
    stub_partial.text = text
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    assert "heard" not in types(msgs)
    assert "final" in types(msgs)


def test_a_failing_preview_never_costs_the_card(client, stub_transcribe,
                                                monkeypatch, caplog):
    def boom(audio):
        raise RuntimeError("metal went away")

    monkeypatch.setattr(srv, "transcribe_partial", boom)
    monkeypatch.setattr(srv, "_parakeet_unavailable", False)
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    assert "heard" not in types(msgs)
    assert "translation_done" in types(msgs)
    assert "preview transcription failed" in caplog.text


def test_a_live_partial_says_which_utterance_it_belongs_to(client,
                                                           stub_transcribe,
                                                           stub_partial,
                                                           monkeypatch):
    """`after` is the id of the last chunk cut before the partial's window was
    taken. The page drops a partial that lands after its own chunk was cut,
    since those words hold a card by then."""
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 0.0)
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=40)
        first = collect_until(ws)
        speak(ws, speech_chunks=40)
        second = collect_until(ws)
    assert {m["after"] for m in first if m["type"] == "partial"} == {0}
    late = [m["after"] for m in second if m["type"] == "partial"]
    assert late and set(late) <= {0, 1} and 1 in late


# ------------------------------------------------------- the placement rule


def extract(name: str) -> str:
    m = re.search(rf"^function {name}\(.*?^}}$", APP_JS, re.S | re.M)
    assert m, f"{name}() not found in app.js — was it renamed or indented?"
    return m.group(0)


def place_for(msg, cards, held):
    """Run the real app.js placeFor() under node. Elements are stood in for
    by strings, which is all the rule needs."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    script = f"""
{extract("placeFor")}
const i = JSON.parse(require("fs").readFileSync(0, "utf8"));
const m = (o) => new Map(Object.entries(o).map(([k, v]) => [Number(k), v]));
console.log(JSON.stringify(placeFor(i.msg, m(i.cards), m(i.held))));
"""
    out = subprocess.run([node, "-e", script], text=True, capture_output=True,
                         input=json.dumps({"msg": msg, "cards": cards,
                                           "held": held}), timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_a_card_lands_in_the_place_held_since_its_chunk_was_cut():
    assert place_for({"id": 7}, {}, {"7": "held7", "8": "held8"}) == {
        "anchor": "held7", "spare": None}


def test_a_merged_card_takes_over_the_card_it_replaces():
    """The reader's eyes are on the old card. The merged one goes there, and
    the place held for the incoming chunk is the one given up."""
    assert place_for({"id": 7, "replaces": 6}, {"6": "card6"},
                     {"7": "held7"}) == {"anchor": "card6", "spare": "held7"}


def test_a_redone_card_with_no_held_place_still_replaces_in_place():
    """The language-chip flip: a fresh id that no chunk was ever cut for."""
    assert place_for({"id": 9, "replaces": 4}, {"4": "card4"}, {}) == {
        "anchor": "card4", "spare": None}


def test_typed_text_has_no_place_and_goes_to_the_end():
    assert place_for({"id": 3}, {}, {}) == {"anchor": None, "spare": None}


def test_a_merge_whose_target_is_gone_falls_back_to_its_own_place():
    assert place_for({"id": 7, "replaces": 6}, {}, {"7": "held7"}) == {
        "anchor": "held7", "spare": None}


def test_a_new_card_no_longer_blanks_the_live_line():
    """A `final` is for an earlier chunk. Clearing the live line on it wiped
    text describing speech that had not been cut yet."""
    branch = re.search(r'msg\.type === "final"\) \{(.*?)\n  \} else if',
                       APP_JS, re.S)
    assert branch, "the final branch of handleMessage moved"
    assert "clearPartial" not in branch.group(1)
    assert "newCard(msg)" in branch.group(1)


def test_the_heard_text_is_not_small_print():
    css = (Path(__file__).parent.parent / "static" / "style.css").read_text()
    rule = re.search(r"^\.card \.orig \{([^}]*)\}", css, re.M)
    assert rule, "the .card .orig rule moved"
    size = re.search(r"font-size:\s*([\d.]+)em", rule.group(1))
    assert size and float(size.group(1)) >= 0.9
    assert "var(--text)" in rule.group(1)
