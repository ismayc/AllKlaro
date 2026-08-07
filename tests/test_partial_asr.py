"""Live partials run on a second, faster ASR pass — off the Whisper thread.

Partials re-decode a rolling ~6 s window every 2 s. Measured over a real
conversation that was ~36% of everything the single Whisper thread decoded,
against a capacity deficit of only ~11% (docs/findings/real-conversation-pace.md),
so moving them is what buys the budget back *without* shedding anything.

Finals and speculations deliberately stay on Whisper: a speculation becomes
the final transcript when the pause turns out to be real, so it is not
optional work.
"""
import pytest

import server as srv
from conftest import collect_until, speak, trace_records


def test_partials_use_the_fast_model_not_whisper(client, stub_transcribe,
                                                 stub_partial, monkeypatch):
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 0.0)
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=40)
        msgs = collect_until(ws)

    partials = [m for m in msgs if m["type"] == "partial"]
    assert partials, "no partial emitted"
    assert partials[0]["text"] == stub_partial.text
    assert stub_partial.calls, "the fast model was never called"
    # Whisper decoded the final only — no partial-sized jobs on its thread.
    assert len(stub_transcribe.calls) <= 2   # final (+ at most one speculation)


def test_partial_window_is_bounded_on_the_fast_path(client, stub_transcribe,
                                                    stub_partial, monkeypatch):
    """A partial still looks at a window, not the whole utterance so far."""
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 0.0)
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=80)
        collect_until(ws)
    limit = srv.PARTIAL_WINDOW_FRAMES * srv.FRAME_MS / 1000
    assert stub_partial.calls
    assert max(stub_partial.calls) <= limit + 0.1


def _partials_skipped(trace_file, client, monkeypatch, **fixtures):
    """Run one utterance and report how many partials were skipped."""
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(srv, "PARTIAL_MAX_QUEUE", 99)  # isolate from shedding
    # Pin the soft-max cap: this pair is about whether partials get their own
    # worker, and it needs one long utterance held in flight to create the
    # contention. Inheriting the default coupled it to an unrelated tuning
    # constant — when that dropped 8.0 -> 5.0 the chunk was emitted before any
    # partial came due, so the fallback path stopped starving and the test
    # failed without the fast path having changed at all.
    monkeypatch.setattr(srv, "SOFT_MAX_SEC", 8.0)
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=40)
        collect_until(ws)
    return sum(r.get("partials_skipped", 0) for r in trace_records(trace_file))


def test_a_busy_whisper_thread_no_longer_starves_partials(
        client, stub_transcribe, stub_partial, monkeypatch, trace_file):
    """The behavioural win, and the reason this change exists. On the slow
    path every partial that came due while an utterance was in flight was
    skipped outright — exactly when speech is densest. With its own worker
    nothing is starved.

    The companion below runs the identical scenario on the fallback path and
    asserts the opposite, so this pair fails if the fast path stops working.
    """
    assert _partials_skipped(trace_file, client, monkeypatch) == 0


def test_the_fallback_path_still_starves_partials(client, stub_transcribe,
                                                  monkeypatch, trace_file):
    """No `stub_partial`, so this is the old behaviour: same run, and the
    Whisper thread being busy does cost partials."""
    assert srv._parakeet_unavailable
    assert _partials_skipped(trace_file, client, monkeypatch) > 0


def test_deep_backlog_still_sheds_partials(client, stub_transcribe,
                                           stub_partial, monkeypatch):
    """Not for contention any more — a partial describing *now*, under cards
    a minute stale, is worse than no partial."""
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(srv, "PARTIAL_MAX_QUEUE", 0)
    monkeypatch.setattr(srv, "whisper_pending", 5)
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=40)
        msgs = collect_until(ws)
    assert not [m for m in msgs if m["type"] == "partial"]
    assert not stub_partial.calls


def test_falls_back_to_whisper_without_the_fast_model(client, stub_transcribe,
                                                      monkeypatch):
    """No parakeet-mlx (or a failed load): partials still work, just back on
    the Whisper thread with the old contention. Degraded, never broken."""
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 0.0)
    assert srv._parakeet_unavailable          # the autouse default
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=40)
        msgs = collect_until(ws)
    partials = [m for m in msgs if m["type"] == "partial"]
    assert partials
    assert partials[0]["text"] == "Wie geht es dir?"   # the Whisper stub
    # Partial-sized jobs are back on Whisper's thread.
    assert len(stub_transcribe.calls) > 1


def test_a_degenerate_fast_decode_never_reaches_the_screen(
        client, stub_transcribe, stub_partial, monkeypatch):
    """Seen live: Parakeet emitted 390 consecutive `<unk>` and the app held
    them on screen for seconds. The fast path skipped `clean_transcript`
    entirely, so every repetition filter that protects the finals was simply
    not run — and none would have matched anyway, because with no spaces
    between the tokens the whole thing is one 1950-character "word"."""
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 0.0)
    stub_partial.text = "<unk>" * 390
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=40)
        msgs = collect_until(ws)
    assert stub_partial.calls, "the fast model was never called"
    assert not [m for m in msgs if m["type"] == "partial"], \
        "a partial made of unknown-token noise was shown"


def test_stray_unknown_tokens_are_stripped_but_the_speech_kept(
        client, stub_transcribe, stub_partial, monkeypatch):
    """Dropping the whole partial for one bad token would throw away good
    live text; the token is the artifact, not the utterance."""
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 0.0)
    stub_partial.text = "Ich habe <unk> das gesagt"
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=40)
        msgs = collect_until(ws)
    partials = [m for m in msgs if m["type"] == "partial"]
    assert partials, "the whole partial was dropped over one token"
    assert "<unk>" not in partials[0]["text"]
    assert "Ich habe" in partials[0]["text"] and "gesagt" in partials[0]["text"]


def test_a_repetition_loop_without_unknown_tokens_is_caught_too(
        client, stub_transcribe, stub_partial, monkeypatch):
    """The unknown-token strip and the repetition checks are separate
    guarantees. Parakeet can loop on real words — "EP" ×32 and "PPE" ×60 both
    appear in the finals from this recording — and stripping `<unk>` does
    nothing for those."""
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 0.0)
    stub_partial.text = "die Füße starete, " * 6
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=40)
        msgs = collect_until(ws)
    assert not [m for m in msgs if m["type"] == "partial"], \
        "a repetition loop was shown as live text"


def test_clean_partial_keeps_ordinary_live_text_untouched():
    """The filter runs on every partial, so anything it mangles is mangled
    two times a second."""
    assert srv.clean_partial("und dann sind wir nach Hause") == \
        "und dann sind wir nach Hause"
    assert srv.clean_partial("Ja, ja, ja") == "Ja, ja, ja"
    assert srv.clean_partial("<unk>" * 390) == ""


def test_transcribe_partial_reports_no_model_rather_than_raising(monkeypatch):
    """maybe_partial() distinguishes 'no fast model' from a real failure by
    the None return, so the contract matters."""
    monkeypatch.setattr(srv, "load_parakeet", lambda: None)
    assert srv.transcribe_partial(None) is None


def test_a_broken_fast_model_does_not_kill_the_stream(client, stub_transcribe,
                                                      stub_partial, monkeypatch):
    """An exception in the partial pass must not take down the socket or stop
    finals — the card is what the user actually needs."""
    def boom(audio):
        raise RuntimeError("mlx exploded")

    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(srv, "transcribe_partial", boom)
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=40)
        msgs = collect_until(ws)
    assert any(m["type"] == "final" for m in msgs)
    assert any(m["type"] == "translation_done" for m in msgs)


def test_speculations_stay_on_whisper(client, stub_transcribe, stub_partial,
                                      monkeypatch):
    """A speculation's result *becomes* the final transcript, so it is not
    optional work and must never be downgraded to the fast model."""
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 0.0)
    with client.websocket_connect("/ws") as ws:
        speak(ws, speech_chunks=40)
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    # The final text came from Whisper's stub, never the partial stub.
    assert final["text"] == "Wie geht es dir?"
    assert stub_partial.text not in final["text"]
