"""The running gist pinned above the feed.

It is a second background job on the Ollama the translator is using, which is
exactly the shape that made the refine pass lose to the translation backlog on
most utterances. So most of what is worth asserting here is about *restraint*:
that it yields to a busy pipeline, that a failure costs nothing, and that it
never grows without bound over a 54-minute call.
"""
import asyncio

import httpx
import pytest

import server as srv
from conftest import SILENCE_CHUNK, collect_until, speak


# ----------------------------------------------------------- the fold prompt


def test_the_fold_carries_the_previous_gist_and_only_the_new_lines():
    """The whole point of folding is a flat cost per refresh: if the previous
    gist were dropped the conversation would restart every minute, and if the
    transcript were resent the prompt would grow all call."""
    msgs = srv.gist_messages("- They are discussing the garden.",
                             ["[DE] Und der Rasen?", "[EN] Forget the lawn."])
    assert len(msgs) == 2 and msgs[0]["role"] == "system"
    body = msgs[1]["content"]
    assert "They are discussing the garden." in body
    assert "Und der Rasen?" in body and "Forget the lawn." in body


def test_an_empty_gist_says_so_rather_than_leaving_a_blank():
    """A bare "Gist so far:" followed by nothing reads as a truncated prompt;
    models fill that silence by inventing a conversation."""
    body = srv.gist_messages("", ["[EN] Hello."])[1]["content"]
    assert "(nothing yet)" in body


def test_language_tags_never_reach_the_panel():
    """The fold is shown "[DE] …" lines, and on long conversations it starts
    copying one through verbatim instead of summarising it. Measured on fold 6
    of a six-fold run; the tags mean nothing to someone reading the panel."""
    assert srv.strip_lang_tags("- [DE] Ich habe gelesen. [EN] Her father too.") \
        == "- Ich habe gelesen. Her father too."
    assert srv.strip_lang_tags("- Nothing to strip here.") \
        == "- Nothing to strip here."


@pytest.mark.asyncio
async def test_the_fold_strips_tags_from_what_the_model_returns(monkeypatch):
    """The guard has to sit on the response, not only in the prompt: asking a
    model not to do something is not the same as it not doing it."""
    transport = httpx.MockTransport(lambda r: httpx.Response(
        200, json={"message": {"content": "- [EN] They agreed to fly."}}))
    monkeypatch.setattr(srv, "ollama_client", lambda: httpx.AsyncClient(
        transport=transport, base_url="http://ollama.test"))
    assert await srv.fold_gist("", ["[EN] x"], "gemma3:12b") \
        == "- They agreed to fly."


def test_the_prompt_defends_the_details_a_fold_would_otherwise_lose():
    """Rolling a summary into a summary drops specifics: without this
    instruction a named destination survived one fold only 50% of the time
    (8/8 with it). The instruction is the fix, so it is worth pinning."""
    low = srv.GIST_PROMPT.lower()
    assert "names" in low and "generalise" in low


def test_the_prompt_tells_the_model_to_ignore_garbled_speech():
    """4 of 259 real utterances are degenerate repetition loops. Summarizing
    them as if they meant something is worse than dropping them."""
    assert "garbled" in srv.GIST_PROMPT or "repeat" in srv.GIST_PROMPT


# ------------------------------------------------------------------ fold_gist


@pytest.mark.asyncio
async def test_fold_returns_none_when_nothing_was_said(monkeypatch):
    """Guards the model call itself: an empty batch must not spend a request."""
    calls = []

    def fail(request):
        calls.append(request)
        return httpx.Response(200, json={"message": {"content": "x"}})

    transport = httpx.MockTransport(fail)
    monkeypatch.setattr(srv, "ollama_client", lambda: httpx.AsyncClient(
        transport=transport, base_url="http://ollama.test"))
    assert await srv.fold_gist("previous", [], "gemma3:12b") is None
    assert calls == []


@pytest.mark.asyncio
async def test_fold_survives_a_refusing_ollama(monkeypatch):
    """A failed gist must leave the old one on screen, not blank the panel or
    raise into the session that is also translating."""
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(refuse)
    monkeypatch.setattr(srv, "ollama_client", lambda: httpx.AsyncClient(
        transport=transport, base_url="http://ollama.test"))
    assert await srv.fold_gist("old gist", ["[EN] Hi."], "gemma3:12b") is None


@pytest.mark.asyncio
async def test_fold_survives_an_error_status(monkeypatch):
    transport = httpx.MockTransport(
        lambda r: httpx.Response(500, text="model not found"))
    monkeypatch.setattr(srv, "ollama_client", lambda: httpx.AsyncClient(
        transport=transport, base_url="http://ollama.test"))
    assert await srv.fold_gist("old", ["[EN] Hi."], "gemma3:12b") is None


@pytest.mark.asyncio
async def test_fold_treats_a_blank_answer_as_failure(monkeypatch):
    """An empty string would otherwise replace a good gist with nothing."""
    transport = httpx.MockTransport(lambda r: httpx.Response(
        200, json={"message": {"content": "   "}}))
    monkeypatch.setattr(srv, "ollama_client", lambda: httpx.AsyncClient(
        transport=transport, base_url="http://ollama.test"))
    assert await srv.fold_gist("old", ["[EN] Hi."], "gemma3:12b") is None


@pytest.mark.asyncio
async def test_fold_returns_the_new_gist(monkeypatch):
    seen = {}

    def handler(request):
        import json
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "• A point"}})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(srv, "ollama_client", lambda: httpx.AsyncClient(
        transport=transport, base_url="http://ollama.test"))
    got = await srv.fold_gist("old", ["[EN] Hi."], "gemma3:12b")
    assert got == "• A point"
    # Non-streaming: the panel is swapped wholesale, so deltas are useless.
    assert seen["stream"] is False


@pytest.mark.asyncio
async def test_fold_disables_thinking_on_reasoning_models(monkeypatch):
    """Without this the gist is the model's chain of thought, not a summary."""
    seen = {}

    def handler(request):
        import json
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "x"}})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(srv, "ollama_client", lambda: httpx.AsyncClient(
        transport=transport, base_url="http://ollama.test"))
    await srv.fold_gist("", ["[EN] Hi."], "qwen3:8b")
    assert seen.get("think") is False
    seen.clear()
    await srv.fold_gist("", ["[EN] Hi."], "gemma3:12b")
    assert "think" not in seen


# --------------------------------------------------------- inside a session


def drain_after_speaking(client, monkeypatch, chunks=60, read=40, config=None):
    """Speak once, then read a fixed number of messages off the socket.

    The gist is sent from a background task, so "did it arrive?" needs a
    stream to read it out of. Turning stats on with a zero interval makes
    every loop iteration emit one, which gives a bounded, non-blocking read:
    asserting a gist is *absent* by waiting for one would hang forever.

    Both halves are returned. The refresh fires from the receive loop as soon
    as an utterance is pending, which is *before* its translation finishes —
    so the gist usually lands ahead of `translation_done`, and a test that
    only inspected what came after would never see it.
    """
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    monkeypatch.setattr(srv, "STATS_INTERVAL_SEC", 0.0)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(config or '{"type": "config", "stats": true}')
        speak(ws)
        msgs = collect_until(ws)
        for _ in range(chunks):      # keep the loop ticking so the check runs
            ws.send_bytes(SILENCE_CHUNK)
        return msgs + [ws.receive_json() for _ in range(read)]


def test_the_gist_reaches_the_client(client, monkeypatch):
    """End to end over the socket, with the interval collapsed so one
    utterance is enough to trigger a refresh."""
    monkeypatch.setattr(srv, "GIST_INTERVAL_SEC", 0.0)
    msgs = drain_after_speaking(client, monkeypatch)
    gists = [m for m in msgs if m["type"] == "gist"]
    assert gists, "no gist ever reached the client"
    assert gists[-1]["text"].strip()


def test_the_gist_stays_off_when_the_client_turns_it_off(client, monkeypatch):
    """The checkbox has to reach the server: a client-side-only toggle would
    still pay for the model call on every refresh."""
    monkeypatch.setattr(srv, "GIST_INTERVAL_SEC", 0.0)
    msgs = drain_after_speaking(
        client, monkeypatch,
        config='{"type": "config", "stats": true, "gist": false}')
    assert not [m for m in msgs if m["type"] == "gist"]


def test_a_busy_pipeline_postpones_the_gist(client, monkeypatch):
    """The lesson of the refine pass: a background job that ignores the
    backlog takes capacity from the card someone is waiting for."""
    monkeypatch.setattr(srv, "GIST_INTERVAL_SEC", 0.0)
    # Nothing is allowed through while anything at all is in flight.
    monkeypatch.setattr(srv, "GIST_MAX_IN_FLIGHT", -1)
    msgs = drain_after_speaking(client, monkeypatch)
    assert not [m for m in msgs if m["type"] == "gist"]


def test_only_one_fold_runs_at_a_time(client, monkeypatch):
    """The batch is only cleared once a fold succeeds, so while one is in
    flight the queue still looks unfolded. Without the busy flag every tick
    would launch another fold of the same lines — piling concurrent requests
    onto the Ollama the translator is waiting on, which is the exact failure
    this feature is supposed to avoid."""
    monkeypatch.setattr(srv, "GIST_INTERVAL_SEC", 0.0)
    live = {"now": 0, "max": 0}

    async def slow(previous, lines, model):
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        await asyncio.sleep(0.05)
        live["now"] -= 1
        return "• still talking"

    monkeypatch.setattr(srv, "fold_gist", slow)
    drain_after_speaking(client, monkeypatch)
    assert live["max"] == 1, f"{live['max']} folds ran at once"


def test_a_whisper_backlog_postpones_the_gist(client, monkeypatch):
    """The other half of the gate. Transcription and translation queue
    independently, and either one being behind means someone is waiting."""
    monkeypatch.setattr(srv, "GIST_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(srv, "GIST_MAX_QUEUE", -1)
    msgs = drain_after_speaking(client, monkeypatch)
    assert not [m for m in msgs if m["type"] == "gist"]


def test_a_permanently_busy_pipeline_still_gets_a_gist_eventually(
        client, monkeypatch):
    """The idle gate alone meant the feature never existed: over 240 s of the
    real recording `in_flight` never fell to the threshold, and not one gist
    refreshed. Past GIST_MAX_STALE_SEC the backlog stops being a veto."""
    monkeypatch.setattr(srv, "GIST_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(srv, "GIST_MAX_IN_FLIGHT", -1)   # never idle
    monkeypatch.setattr(srv, "GIST_MAX_QUEUE", -1)
    monkeypatch.setattr(srv, "GIST_MAX_STALE_SEC", 0.0)  # ...but already stale
    msgs = drain_after_speaking(client, monkeypatch)
    assert [m for m in msgs if m["type"] == "gist"], \
        "a busy pipeline suppressed the gist forever"


def test_the_interval_actually_holds_the_gist_back(client, monkeypatch):
    """With the real interval, a few seconds of test audio must produce
    nothing: otherwise the panel would rewrite itself constantly."""
    msgs = drain_after_speaking(client, monkeypatch)
    assert not [m for m in msgs if m["type"] == "gist"]


# ------------------------------------------------------------------- bounds


def test_the_pending_list_is_bounded():
    """A 54-minute call with a wedged Ollama must not accumulate an unbounded
    transcript in memory; the newest lines still describe the conversation."""
    pending = []
    for i in range(srv.GIST_MAX_PENDING + 50):
        srv.remember_for_gist(pending, i, "de", f"line {i}")
    assert len(pending) == srv.GIST_MAX_PENDING
    assert pending[-1]["uid"] == srv.GIST_MAX_PENDING + 49   # newest kept
    assert pending[0]["uid"] == 50                           # oldest dropped


def test_a_merged_fragment_is_not_counted_twice():
    """Merging rewrites an utterance as "fragment + rest". Both were queued,
    so without the pop the gist reads the opening half of the sentence twice
    and can weight it as if it had been said twice."""
    pending = []
    srv.remember_for_gist(pending, 1, "de", "Ich habe gedacht")
    srv.remember_for_gist(pending, 2, "de", "Ich habe gedacht, es regnet",
                          replaces=1)
    assert [p["text"] for p in pending] == ["Ich habe gedacht, es regnet"]


def test_only_the_merged_fragment_is_dropped():
    """The pop is deliberately scoped to the immediately preceding utterance;
    a stale or mismatched uid must not silently delete someone else's line."""
    pending = []
    srv.remember_for_gist(pending, 1, "de", "Erste Zeile")
    srv.remember_for_gist(pending, 2, "en", "Second line")
    srv.remember_for_gist(pending, 3, "de", "Dritte Zeile", replaces=1)
    assert [p["uid"] for p in pending] == [1, 2, 3]


def test_one_refresh_folds_a_bounded_number_of_lines():
    """After a long stall the backlog can be large; folding all of it would
    make the one refresh that finally runs the most expensive of the call."""
    assert srv.GIST_MAX_LINES < srv.GIST_MAX_PENDING


def test_the_gist_yields_to_the_same_backlog_the_refine_pass_does():
    """Not a style point: these gates are why the gist cannot become a second
    refine pass. If they drifted above the refine thresholds the gist would
    run in conditions where refinement is already considered too expensive."""
    assert srv.GIST_MAX_IN_FLIGHT <= srv.REFINE_MAX_IN_FLIGHT
    assert srv.GIST_MAX_QUEUE <= srv.REFINE_MAX_QUEUE
