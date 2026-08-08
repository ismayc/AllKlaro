"""The on-demand "what did they just say?" window.

This is the ✨ tap widened from one card to a stretch of them, and item 12 is
why it exists: after the item 8 merge, 41% of cards are still fragments,
because that merge only reaches across a 2 s gap while a German clause
routinely spans more. Every card on screen can be a fragment while the
passage they belong to is a perfectly good sentence.

So the thing to pin here is that a recap is ONE translation of the joined
passage rather than a re-run of each card, and that it does not contaminate
what follows: it is a second reading of words the conversation already has,
and letting it into `history` would count them twice.
"""
import asyncio
import json

import pytest

import server as srv
from conftest import (collect_until, collect_until_bounded, receive_bounded,
                      speak)

PASSAGE = "Der Vermieter hat gesagt, wenn wir die Wärme halten wollten."


def recap(ws, text=PASSAGE, source="de", target="en", before_uid=None):
    msg = {"type": "recap", "text": text, "source": source, "target": target}
    if before_uid is not None:
        msg["before_uid"] = before_uid
    ws.send_text(json.dumps(msg))


def until_recap(ws, seconds=5.0):
    """Bounded, because "asking always gets an answer" is under test here.
    An unbounded wait would hang the suite instead of failing this file."""
    return collect_until_bounded(ws, "recap", seconds=seconds)


def no_recap_arrives(ws, seconds=0.4):
    """True when nothing comes back at all.

    The mirror image of `until_recap`, and it needs its own helper: the
    bounded collector RAISES when the awaited message never lands, which is
    the correct behaviour when a reply is the guarantee and the wrong one when
    silence is. Draining to the deadline is the only way to assert silence.
    """
    while True:
        try:
            raw = receive_bounded(ws, seconds)
        except TimeoutError:
            return True
        msg = json.loads(raw["text"]) if "text" in raw else raw
        if msg.get("type") == "recap":
            return False


def test_the_window_comes_back_translated_as_one_passage(client, fake_ollama):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        recap(ws)
        reply = until_recap(ws)[-1]
    assert reply["text"] == "Refined translation."
    assert reply["target"] == "en" and reply["source"] == "de"
    assert "error" not in reply
    # The main model, not the draft. A recap is asked for on purpose, so it
    # gets the same model the ✨ tap does.
    assert fake_ollama["chat"]["model"] == "main-model"


def test_the_joined_passage_is_translated_once_not_card_by_card(client,
                                                                fake_ollama):
    """The whole point. Translating each card separately is what the screen
    already did, and it is what produced the fragments."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        recap(ws)
        until_recap(ws)
    sent = [b for b in fake_ollama["all"] if b.get("stream") is False]
    assert len(sent) == 1, f"{len(sent)} model runs for one recap"
    assert PASSAGE in sent[0]["messages"][-1]["content"]


def test_the_heard_passage_comes_back_with_the_translation(client, fake_ollama):
    """The panel shows both halves: the point is to re-read the stretch, and
    the German is what the listener is checking the English against."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        recap(ws)
        reply = until_recap(ws)[-1]
    assert reply["heard"] == PASSAGE


def test_a_recap_is_not_gated_by_the_backlog(client, fake_ollama, monkeypatch):
    """Same reasoning as the ✨ tap: somebody is waiting on purpose, and the
    moment the pipeline is behind is exactly when they missed something."""
    monkeypatch.setattr(srv, "REFINE_MAX_IN_FLIGHT", 0)
    monkeypatch.setattr(srv, "REFINE_MAX_QUEUE", 0)
    monkeypatch.setattr(srv, "whisper_pending", srv.REFINE_MAX_QUEUE + 5)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        recap(ws)
        reply = until_recap(ws)[-1]
    assert reply.get("text") == "Refined translation."


def test_a_second_ask_while_one_runs_says_busy(client, monkeypatch):
    """The windows overlap by construction — a second tap covers almost the
    same seconds — so queueing them would be the same passage racing itself."""
    calls = []

    async def slow(*a, **k):
        calls.append(1)
        await asyncio.sleep(0.25)
        return "Re-read."

    monkeypatch.setattr(srv, "translate_once", slow)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        recap(ws)
        recap(ws)
        replies = [m for m in until_recap(ws) if m["type"] == "recap"]
    assert any(r.get("error") == "busy" for r in replies)
    assert len(calls) == 1, f"{len(calls)} model runs for overlapping asks"


def test_the_guard_is_released_so_a_later_ask_still_works(client, fake_ollama):
    """A single-flight flag that leaks on the happy path disables the button
    for the rest of the conversation."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        recap(ws)
        until_recap(ws)
        recap(ws)
        reply = until_recap(ws)[-1]
    assert reply.get("text") == "Refined translation."


def test_the_guard_is_released_after_a_failure_too(client, monkeypatch):
    calls = []

    async def broken(*a, **k):
        calls.append(1)
        return None

    monkeypatch.setattr(srv, "translate_once", broken)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        recap(ws)
        assert until_recap(ws)[-1]["error"] == "failed"
        recap(ws)
        assert until_recap(ws)[-1]["error"] == "failed"
    assert len(calls) == 2, "the flag stuck after a failure"


def test_a_failed_recap_is_reported_not_swallowed(client, monkeypatch):
    async def broken(*a, **k):
        return None

    monkeypatch.setattr(srv, "translate_once", broken)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        recap(ws)
        reply = until_recap(ws)[-1]
    assert reply["error"] == "failed" and "text" not in reply


def test_a_hung_model_does_not_leave_the_panel_waiting(client, monkeypatch):
    monkeypatch.setattr(srv, "RECAP_TIMEOUT_SEC", 0.05)

    async def never(*a, **k):
        await asyncio.sleep(30)

    monkeypatch.setattr(srv, "translate_once", never)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        recap(ws)
        reply = until_recap(ws)[-1]
    assert reply["error"] == "failed"


def test_a_recap_does_not_steer_what_follows(client, fake_ollama):
    """The words in the window are already in `history` as cards. Adding the
    re-reading would put the same speech in the model's context twice, and the
    second copy is a translation being passed off as something that was said.
    """
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        speak(ws)
        collect_until(ws, stop_types=("translation_revised", "error"))
        recap(ws, text="Etwas ganz Anderes.")
        until_recap(ws)
        fake_ollama["all"].clear()
        ws.send_text(json.dumps({"type": "text", "text": "Und danach?"}))
        collect_until(ws)
    convo = [m["content"] for b in fake_ollama["all"] for m in b["messages"]]
    assert not any("Etwas ganz Anderes." in c for c in convo), \
        "the re-read passage leaked into the conversation context"


def test_context_stops_where_the_passage_starts(client, fake_ollama):
    """Otherwise the model gets the same words twice — once as history and
    once as the thing to translate — and is asked to agree with itself."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        speak(ws)
        collect_until(ws, stop_types=("translation_revised", "error"))
        fake_ollama["all"].clear()
        recap(ws, before_uid=1)          # the window starts at the only card
        until_recap(ws)
    sent = [b for b in fake_ollama["all"] if b.get("stream") is False][-1]
    # Only the system prompt and the passage itself; no card turns between.
    roles = [m["role"] for m in sent["messages"]]
    assert roles.count("user") == 1, f"context leaked in: {roles}"


def test_without_a_before_uid_the_whole_history_is_context(client, fake_ollama):
    """A window that never matched a card still deserves the conversation so
    far — that is the difference between no context and wrong context."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        speak(ws)
        collect_until(ws, stop_types=("translation_revised", "error"))
        fake_ollama["all"].clear()
        recap(ws)
        until_recap(ws)
    sent = [b for b in fake_ollama["all"] if b.get("stream") is False][-1]
    assert len(sent["messages"]) > 2, "the earlier card was dropped as context"


@pytest.mark.parametrize("kwargs", [
    {"source": "en", "target": "en"},        # nothing to translate
    {"target": ""},                          # no direction picked
    {"source": "kl"},                        # not a language this app knows
])
def test_an_impossible_direction_is_ignored(client, kwargs):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        recap(ws, **kwargs)
        assert no_recap_arrives(ws)


def test_an_empty_window_is_never_sent_to_the_model(client, fake_ollama):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        recap(ws, text="   ")
        assert no_recap_arrives(ws)
    assert not [b for b in fake_ollama.get("all", []) if b.get("stream") is False]


def test_a_very_long_window_is_bounded_before_it_reaches_ollama(client,
                                                                fake_ollama):
    """The client bounds the window by cards, but the server cannot trust a
    client — a runaway passage would be a prompt the GPU chokes on."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        recap(ws, text="Das Wasservolumen. " * 500)
        until_recap(ws)
    sent = [b for b in fake_ollama["all"] if b.get("stream") is False][-1]
    assert len(sent["messages"][-1]["content"]) <= srv.RECAP_MAX_CHARS + 400


def test_the_server_does_not_keep_its_own_copy_of_the_window():
    """`recapWindow` in app.js owns it, because only the client knows when a
    card reached the screen. A second copy here would be a number free to
    drift from the one the button actually uses."""
    assert not hasattr(srv, "RECAP_WINDOW_SEC")
    assert not hasattr(srv, "RECAP_MAX_CARDS")
