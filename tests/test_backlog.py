"""Optional work is shed by backlog depth, so a real utterance never is.

Measured on a real conversation (docs/findings/real-conversation-pace.md): the
pipeline decoded 2.7x more audio than existed — rolling partials and
speculations — putting the one Whisper thread at ~1.11x realtime against a 1.0x
budget. An 11% overload queues without bound. All of the surplus is optional,
so these thresholds buy the budget back without dropping anything a listener
would miss.
"""
import asyncio

import pytest

import server as srv
from conftest import SPEECH_CHUNK, collect_until, speak, trace_records


@pytest.fixture
def backlog(monkeypatch):
    """Pretend the Whisper executor is already deep in work."""
    return lambda n: monkeypatch.setattr(srv, "whisper_pending", n)


def translation_models(fake_ollama):
    """Models actually asked to translate, ignoring the prewarm ping.

    Prewarm loads the refine model ahead of time; it is a one-token request
    and must not be mistaken for a refinement having run.
    """
    return [b["model"] for b in fake_ollama.get("all", [])
            if b.get("options", {}).get("num_predict") != 1]


@pytest.fixture(autouse=True)
def forget_prewarmed():
    srv._prewarmed.clear()
    yield
    srv._prewarmed.clear()


# ------------------------------------------------------------- speculation


def test_speculation_runs_when_the_pipe_is_clear(client, stub_transcribe,
                                                 monkeypatch):
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    assert any(m["type"] == "final" for m in msgs)
    # The speculation is reused as the final, so exactly one decode happens.
    assert len(stub_transcribe.calls) == 1


def test_speculation_is_shed_when_the_queue_is_deep(client, stub_transcribe,
                                                    monkeypatch, backlog):
    """A speculation only pays off if it lands before the real chunk does.

    Behind a queue it cannot, and it doubles the audio the single thread has
    to decode — which is how the backlog got deep in the first place.
    """
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    backlog(srv.SPEC_MAX_QUEUE + 5)
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    assert any(m["type"] == "final" for m in msgs), \
        "the utterance itself must still be transcribed"
    assert len(stub_transcribe.calls) == 1


# ---------------------------------------------------------------- partials


def test_partials_stream_when_the_pipe_is_clear(client, stub_transcribe,
                                                monkeypatch):
    """Control for the test below: partials are what make it feel live."""
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 0.0)
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=40)
        msgs = collect_until(ws)
    assert any(m["type"] == "partial" for m in msgs)


def test_partials_are_shed_when_the_queue_is_deep(client, stub_transcribe,
                                                  monkeypatch, backlog):
    """`busy` only knows about one in-flight handler, not the queue behind it.

    Without this depth check a partial can be admitted in front of a dozen
    waiting finals — the opposite of what a listener wants.
    """
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 0.0)
    backlog(srv.PARTIAL_MAX_QUEUE + 5)
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=40)
        msgs = collect_until(ws)
    assert not any(m["type"] == "partial" for m in msgs)
    assert any(m["type"] == "final" for m in msgs)


def test_partial_window_stays_bounded():
    """Partials re-decode a rolling window; its size is the cost of each one."""
    assert srv.PARTIAL_WINDOW_FRAMES * srv.FRAME_MS / 1000 <= 7.0


# ------------------------------------------------------------------ refine


def test_refine_is_shed_when_the_queue_is_deep(client, stub_transcribe,
                                               fake_ollama, monkeypatch,
                                               backlog):
    """The refine pass improves wording after the card is already readable.

    Behind a backlog it competes for the same GPU as work the user is still
    waiting on, so it is the first thing to go.
    """
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    backlog(srv.REFINE_MAX_QUEUE + 5)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        speak(ws)
        msgs = collect_until(ws, stop_types=("translation_revised", "error"))
    done = [m for m in msgs if m["type"] == "translation_done"]
    assert done and done[0]["refining"] is True
    models = translation_models(fake_ollama)
    # The draft still ran; only the second, slower pass was skipped.
    assert "fast-draft" in models
    assert "main-model" not in models


def test_refine_runs_when_the_pipe_is_clear(client, stub_transcribe,
                                            fake_ollama, monkeypatch):
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        speak(ws)
        collect_until(ws, stop_types=("translation_revised", "error"))
    models = translation_models(fake_ollama)
    assert "fast-draft" in models and "main-model" in models


def test_refine_cannot_stall_the_pipeline(client, stub_transcribe, monkeypatch):
    """A refinement nobody is waiting for must not block the queue behind it.

    Measured: pointing draft and main at two different Ollama models made every
    refine take over two minutes (model swapping), which stalled everything.
    """
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    monkeypatch.setattr(srv, "REFINE_TIMEOUT_SEC", 0.05)

    async def never_returns(*a, **k):
        await asyncio.sleep(30)

    monkeypatch.setattr(srv, "translate_once", never_returns)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        speak(ws)
        msgs = collect_until(ws, stop_types=("translation_revised", "error"))
    # The draft card still landed, and the hung refine was abandoned.
    assert any(m["type"] == "translation_done" for m in msgs)
    assert any(m["type"] == "translation_revised" for m in msgs)


def test_refine_is_shed_on_an_already_stale_utterance(client, stub_transcribe,
                                                      fake_ollama, monkeypatch):
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    monkeypatch.setattr(srv, "REFINE_MAX_AGE_SEC", -1.0)   # everything is stale
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        speak(ws)
        collect_until(ws, stop_types=("translation_revised", "error"))
    models = translation_models(fake_ollama)
    assert "fast-draft" in models and "main-model" not in models


def test_refine_model_is_prewarmed_when_a_draft_is_configured(client,
                                                              fake_ollama):
    """A cold model load takes far longer than REFINE_TIMEOUT_SEC.

    Without prewarming, every refine is aborted mid-load and starts over, so
    the refinement never lands once — measured as refine_ms pinned at exactly
    the timeout while the user silently got draft-quality text forever.
    """
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        speak(ws)
        collect_until(ws, stop_types=("translation_revised", "error"))
    pre = [b for b in fake_ollama.get("all", [])
           if b.get("options", {}).get("num_predict") == 1]
    assert any(b["model"] == "main-model" for b in pre)


def test_no_prewarm_without_a_draft_model(client, fake_ollama):
    """Single-pass sessions never swap models, so there is nothing to prewarm."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model"})
        speak(ws)
        collect_until(ws)
    pre = [b for b in fake_ollama.get("all", [])
           if b.get("options", {}).get("num_predict") == 1]
    assert not pre


# -------------------------------------------------------------- instrument


def test_lag_excludes_the_background_refine(client, stub_transcribe,
                                            trace_file, monkeypatch):
    """lag_ms must be time-to-card, not time-to-refine.

    The refine pass improves text the user can already read. Charging it to the
    lag made one measured run report 125 s for cards that appeared in ~13 s —
    the same class of error as the on-screen card timing only its own chunk.
    """
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    slow = 0.4

    async def slow_refine(*a, **k):
        await asyncio.sleep(slow)
        return "Refined translation."

    monkeypatch.setattr(srv, "translate_once", slow_refine)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "config", "model": "main-model",
                      "draft_model": "fast-draft"})
        speak(ws)
        collect_until(ws, stop_types=("translation_revised", "error"))
    records = [r for r in trace_records(trace_file) if r.get("uid")]
    assert records
    rec = records[0]
    assert rec["refine_ms"] >= slow * 1000 * 0.8, "the refine really was slow"
    assert rec["lag_ms"] < slow * 1000, \
        f"lag_ms {rec['lag_ms']}ms still includes the {rec['refine_ms']}ms refine"





def test_trace_records_what_was_shed(client, stub_transcribe, trace_file,
                                     monkeypatch, backlog):
    """Shedding must be visible, or it looks identical to the app being slow."""
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    backlog(srv.SPEC_MAX_QUEUE + 5)
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        collect_until(ws)
    records = [r for r in trace_records(trace_file) if r.get("uid")]
    assert records
    assert "specs_shed" in records[0] and "refines_shed" in records[0]
    assert records[0]["specs_shed"] >= 1
