"""The on-demand "improve this card" tap.

The refine pass is the same idea running automatically, and its two guards —
yield to the backlog, die at the timeout — are what make it land on a minority
of utterances. Both exist because it competes with a card nobody has read yet.
A tap is a person waiting on purpose, so neither applies, and this file exists
mostly to pin that difference: an improve must NOT be gated, and must not be
quietly dropped when the pipeline is busy.
"""
import asyncio
import json

import pytest

import server as srv
from conftest import collect_until, collect_until_bounded, speak


@pytest.fixture
def backlog(monkeypatch):
    """Pretend the Whisper executor is already deep in work."""
    return lambda n: monkeypatch.setattr(srv, "whisper_pending", n)


def improve(ws, uid, text="Wie geht es dir?", source="de", target="en"):
    ws.send_text(json.dumps({"type": "improve", "id": uid, "text": text,
                             "source": source, "target": target}))


def until_improved(ws, seconds=5.0):
    """Messages up to and including the `improved` reply.

    Bounded, because "a tap always gets an answer" is itself one of the
    guarantees under test: if it regresses there is no reply at all, and an
    unbounded wait would hang the suite rather than fail this test.
    """
    return collect_until_bounded(ws, "improved", seconds=seconds)


def test_a_tap_returns_the_main_models_answer(client, fake_ollama):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        improve(ws, 1)
        msgs = until_improved(ws)
    reply = msgs[-1]
    assert reply["id"] == 1 and reply["target"] == "en"
    assert reply["text"] == "Refined translation."
    assert "error" not in reply
    # The main model, not the draft: the whole point is the model the refine
    # pass keeps failing to reach.
    assert fake_ollama["chat"]["model"] == "main-model"
    assert fake_ollama["chat"]["stream"] is False


def test_an_improve_is_not_gated_by_the_backlog(client, fake_ollama,
                                                monkeypatch, backlog):
    """The difference that justifies the feature.

    A refine behind a backlog is skipped, and rightly — nobody asked for it.
    An improve behind the same backlog must still run, or the tap does nothing
    at the exact moment the automatic pass was already failing.
    """
    monkeypatch.setattr(srv, "REFINE_MAX_IN_FLIGHT", 0)
    monkeypatch.setattr(srv, "REFINE_MAX_QUEUE", 0)
    monkeypatch.setattr(srv, "REFINE_MAX_AGE_SEC", -1.0)   # everything is stale
    backlog(srv.REFINE_MAX_QUEUE + 5)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        improve(ws, 1)
        reply = until_improved(ws)[-1]
    assert reply.get("text") == "Refined translation.", \
        "the tap was shed like an automatic refine"


def test_a_second_tap_on_the_same_card_is_a_no_op(client, fake_ollama,
                                                  monkeypatch):
    """Double-tap is the normal way a touch UI misfires, and each tap costs a
    full main-model run on a GPU three things already share."""
    started = asyncio.Event()
    calls = []

    async def slow(*a, **k):
        calls.append(1)
        started.set()
        await asyncio.sleep(0.2)
        return "Improved."

    monkeypatch.setattr(srv, "translate_once", slow)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        improve(ws, 1)
        improve(ws, 1)
        reply = until_improved(ws)[-1]
    assert reply["text"] == "Improved."
    assert len(calls) == 1, f"{len(calls)} main-model runs for one card"


def test_too_many_at_once_says_so_instead_of_queueing_silently(client,
                                                               monkeypatch):
    """A tap that silently does nothing is indistinguishable from a broken
    button, so the cap has to report itself."""
    monkeypatch.setattr(srv, "IMPROVE_MAX_IN_FLIGHT", 1)

    async def slow(*a, **k):
        await asyncio.sleep(0.3)
        return "Improved."

    monkeypatch.setattr(srv, "translate_once", slow)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        improve(ws, 1)
        improve(ws, 2)
        replies = [m for m in until_improved(ws) if m["type"] == "improved"]
    assert replies[-1]["error"] == "busy" and replies[-1]["id"] == 2


def test_a_failed_improve_is_reported_not_swallowed(client, monkeypatch):
    async def broken(*a, **k):
        return None

    monkeypatch.setattr(srv, "translate_once", broken)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        improve(ws, 1)
        reply = until_improved(ws)[-1]
    assert reply["error"] == "failed" and "text" not in reply


def test_a_hung_model_does_not_wedge_the_card_forever(client, monkeypatch):
    """No deadline to *beat* is not the same as no deadline at all: without
    one, a wedged Ollama leaves the button spinning for the rest of the call."""
    monkeypatch.setattr(srv, "IMPROVE_TIMEOUT_SEC", 0.05)

    async def never(*a, **k):
        await asyncio.sleep(30)

    monkeypatch.setattr(srv, "translate_once", never)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        improve(ws, 1)
        reply = until_improved(ws)[-1]
    assert reply["error"] == "failed"


def test_the_improved_text_steers_what_follows(client, fake_ollama):
    """A card still inside the context window should carry its better wording
    into the next utterances, the same way a hand correction does."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        speak(ws)
        collect_until(ws, stop_types=("translation_revised", "error"))
        improve(ws, 1)
        until_improved(ws)
        ws.send_text(json.dumps({"type": "text", "text": "Und danach?"}))
        collect_until(ws)
    convo = [m["content"] for m in fake_ollama["chat"]["messages"]]
    assert any("Refined translation." in c for c in convo)


def test_a_hand_correction_outranks_a_later_improve(client, fake_ollama):
    """The user's own words are the one thing a model result must not
    overwrite — the same rule the refine pass follows."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        speak(ws)
        collect_until(ws, stop_types=("translation_revised", "error"))
        ws.send_text(json.dumps({"type": "correction", "id": 1,
                                 "target": "en", "corrected": "My own words."}))
        improve(ws, 1)
        until_improved(ws)
        ws.send_text(json.dumps({"type": "text", "text": "Und danach?"}))
        collect_until(ws)
    convo = [m["content"] for m in fake_ollama["chat"]["messages"]]
    assert any("My own words." in c for c in convo)
    assert not any("Refined translation." in c for c in convo), \
        "the improve overwrote the user's correction in the live context"


@pytest.mark.parametrize("payload", [
    {"type": "improve", "id": 1, "text": "hi", "source": "de", "target": "zz"},
    {"type": "improve", "id": 1, "text": "hi", "source": "de", "target": "de"},
    {"type": "improve", "id": "one", "text": "hi", "source": "de",
     "target": "en"},
    {"type": "improve", "id": 1, "text": "   ", "source": "de", "target": "en"},
])
def test_a_malformed_tap_is_ignored_not_answered(client, payload):
    """Everything here arrives from the page, so it is untrusted input; a bad
    one must not start a main-model run or crash the session."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        ws.send_text(json.dumps(payload))
        # The session is still alive and still translating.
        ws.send_text(json.dumps({"type": "text", "text": "Hallo"}))
        msgs = collect_until(ws)
    assert any(m["type"] == "final" for m in msgs)
    assert not any(m["type"] == "improved" for m in msgs)


def test_dialect_output_skips_the_declension_guard(client, fake_ollama):
    """The guard assumes standard German and would "correct" Berlinerisch back
    to Hochdeutsch — the same exemption the refine pass has."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "de_flavor": "berlin"})
        improve(ws, 1, text="How are you?", source="en", target="de")
        reply = until_improved(ws)[-1]
    assert reply["text"] == "Refined translation."
