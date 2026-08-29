"""The branches that only run when something has already gone wrong.

Every one of these is a path the app takes on a bad day: a missing file, an
Ollama that answers with an error or not at all, a model that will not load, a
client that hung up mid-translation. They are the least-exercised code in the
server and the most expensive to get wrong, because each one runs at the exact
moment the normal path has stopped working.

Anything here that cannot be reached through the public surface says so in its
docstring and reaches the branch by substitution instead, so a reader can tell
a real failure mode from defensive redundancy.
"""
import asyncio
import json

import httpx
import numpy as np
import pytest

import server as srv
from conftest import (FakeWS, collect_until, collect_until_bounded,
                      speak)


# --- instrumentation must never break a session ------------------------------

def test_a_trace_write_that_fails_is_swallowed(tmp_path, monkeypatch):
    """The trace is diagnostics. A full disk or an unwritable path must cost a
    log line, not the conversation the app is in the middle of."""
    monkeypatch.setattr(srv, "TRACE_PATH", str(tmp_path))   # a directory
    srv.trace({"uid": 1})                                   # must not raise


# --- files the user is allowed to delete -------------------------------------

def test_a_missing_dialects_file_yields_no_dialects(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "DIALECTS_PATH", tmp_path / "gone.txt")
    srv._dialects_cache.update(mtime=None, map={})
    try:
        assert srv.load_dialects() == {}
    finally:
        srv._dialects_cache.update(mtime=None, map={})


def test_dialect_notes_with_an_empty_lexicon_returns_nothing(monkeypatch):
    """A dialect the file knows nothing about is not an error: the prompt
    simply carries no hint."""
    monkeypatch.setattr(srv, "load_dialects", lambda: {})
    assert srv.dialect_notes("Ick koof mir wat.", "de", "berlin") is None


# --- the grammar guard declining to speak ------------------------------------

def test_wanted_determiner_is_none_for_an_unrecognized_determiner():
    """_wanted_det only knows articles and possessives; anything else has no
    "correct form" to suggest, and inventing one would be worse than silence."""
    assert srv._wanted_det("irgendwelche", ["nom"], "f") is None


def test_a_noun_with_too_many_genders_is_left_alone(noun_forms):
    """Three readings make the hint unreadable ("it must be der or die or
    das"), so the guard stays quiet rather than printing mush. Homographs
    across genders are real: "der See" and "die See" are different words."""
    # All three share the lemma, or _gender_options prefers the exact-lemma
    # subset and the count never reaches three. Nominative-only readings keep
    # the NP incompatible with "zu" + dative, so the check does not exit
    # earlier on a reading that happens to fit.
    noun_forms([("See", "See", "m", "ns"), ("See", "See", "f", "ns"),
                ("See", "See", "n", "ns")])
    assert srv.agreement_issues("Ich gehe zu der See.", "de") == []


def test_spanish_measure_words_are_skipped(output_gender_map):
    """"un poco agua" takes its article from the measure word, not from the
    noun behind it, so the feminine "agua" is not evidence of an error."""
    output_gender_map([("agua", "f"), ("poco", "m")], target="es")
    assert srv.agreement_issues("Quiero un poco agua.", "es") == []


def test_spanish_nouns_missing_from_the_lexicon_are_skipped(output_gender_map):
    """Nothing in the map means nothing to check: an unknown noun must not be
    guessed at from its ending."""
    output_gender_map([("mesa", "f")], target="es")
    assert srv.agreement_issues("Compré el chiribitil ayer.", "es") == []


# --- transcripts that are mostly loop ----------------------------------------

def test_a_segment_that_is_almost_entirely_repetition_is_dropped():
    """Whisper loops on near-silence. Collapsing "ja ja ja ..." leaves a
    fragment too short to be worth a card, and the caller wants None, not a
    two-word card that interrupts the conversation."""
    assert srv._clean_segment("ja, " * 40) is None


# --- Ollama answering badly, or not at all -----------------------------------

def test_safe_send_reports_a_socket_that_has_already_closed():
    class Closed:
        async def send_json(self, payload):
            raise RuntimeError("websocket is closed")

    assert asyncio.run(srv.safe_send(Closed(), {"type": "stats"})) is False


def test_summarize_surfaces_an_ollama_error_status(monkeypatch):
    def respond(request):
        return httpx.Response(500, text="model runner crashed")

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(srv, "ollama_client",
                        lambda: httpx.AsyncClient(transport=transport,
                                                  base_url="http://ollama.test"))
    out = asyncio.run(srv.summarize({"items": [{"text": "Hallo", "source": "de"}],
                                     "model": "gemma3:12b"}))
    assert "Ollama error" in out["error"]


def test_a_thinking_model_is_told_not_to_think(monkeypatch):
    """qwen3 and friends emit a reasoning block that would land in the card.
    Both the summary and the translation path have to switch it off."""
    seen = {}

    def respond(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(srv, "ollama_client",
                        lambda: httpx.AsyncClient(transport=transport,
                                                  base_url="http://ollama.test"))
    asyncio.run(srv.summarize({"items": [{"text": "Hallo", "source": "de"}],
                               "model": "qwen3:8b"}))
    assert seen["think"] is False
    seen.clear()
    asyncio.run(srv.translate_once("Hallo", "de", "en", "qwen3:8b"))
    assert seen["think"] is False


def test_a_prewarm_that_fails_leaves_the_model_unmarked(monkeypatch):
    """Prewarming is an optimization. If it fails the model must drop out of
    _prewarmed so the next call tries again instead of assuming a warm model."""
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(refuse)
    monkeypatch.setattr(srv, "ollama_client",
                        lambda: httpx.AsyncClient(transport=transport,
                                                  base_url="http://ollama.test"))
    srv._prewarmed.discard("gemma3:12b")
    asyncio.run(srv.prewarm_model("gemma3:12b"))
    assert "gemma3:12b" not in srv._prewarmed


def test_ollama_client_builds_a_real_client_against_the_configured_url():
    """The factory exists so tests can swap a transport in; the un-swapped
    version still has to point at the configured Ollama."""
    client = srv.ollama_client()
    try:
        assert str(client.base_url).rstrip("/") == srv.OLLAMA_URL.rstrip("/")
    finally:
        asyncio.run(client.aclose())


# --- streaming a translation into a socket that may go away ------------------

def _stream_transport(lines, monkeypatch):
    def respond(request):
        return httpx.Response(200, text="".join(lines))

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(srv, "ollama_client",
                        lambda: httpx.AsyncClient(transport=transport,
                                                  base_url="http://ollama.test"))


def test_blank_lines_in_the_stream_are_skipped(monkeypatch):
    """Ollama's ndjson carries keep-alive blank lines; json.loads("") would
    end the translation with an exception."""
    _stream_transport([json.dumps({"message": {"content": "Hi"}}) + "\n",
                       "\n",
                       json.dumps({"message": {"content": ""}, "done": True}) + "\n"],
                      monkeypatch)
    ws = FakeWS()
    out = asyncio.run(srv.stream_translation(ws, 1, "Hallo", "de", "en",
                                             "gemma3:12b"))
    assert out == "Hi"


def test_a_client_that_leaves_mid_stream_stops_the_translation(monkeypatch):
    """Nobody is reading the deltas any more, and the tokens still cost GPU
    time that the next conversation needs."""
    _stream_transport([json.dumps({"message": {"content": "Hi"}}) + "\n",
                       json.dumps({"message": {"content": " there"}}) + "\n",
                       json.dumps({"message": {"content": ""}, "done": True}) + "\n"],
                      monkeypatch)

    class Leaves:
        def __init__(self):
            self.sent = []

        async def send_json(self, payload):
            self.sent.append(payload)
            raise RuntimeError("client gone")

    ws = Leaves()
    assert asyncio.run(srv.stream_translation(ws, 1, "Hallo", "de", "en",
                                              "gemma3:12b")) is None
    assert len(ws.sent) == 1          # stopped at the first failed send


def test_an_unexpected_translation_error_reaches_the_card(monkeypatch):
    """Not a ConnectError, which has its own message: anything else must still
    tell the user why the card never filled in."""
    def explode(request):
        raise ValueError("transport exploded")

    transport = httpx.MockTransport(explode)
    monkeypatch.setattr(srv, "ollama_client",
                        lambda: httpx.AsyncClient(transport=transport,
                                                  base_url="http://ollama.test"))
    ws = FakeWS()
    assert asyncio.run(srv.stream_translation(ws, 7, "Hallo", "de", "en",
                                              "gemma3:12b")) is None
    assert ws.sent[-1]["type"] == "error"
    assert "transport exploded" in ws.sent[-1]["message"]


# --- LanguageTool, which is opt-in and usually absent ------------------------

def test_languagetool_absence_is_cached_not_retried(monkeypatch):
    """The import failing is the normal case (it is an extra, and it needs
    Java). It must be remembered, or every card pays the failed import."""
    monkeypatch.setattr(srv, "_lt_tools", {})
    monkeypatch.setitem(__import__("sys").modules, "language_tool_python", None)
    assert srv._lt_tool("de") is None
    assert srv._lt_tools == {"de": None}


def test_a_languagetool_check_that_raises_yields_no_issues(monkeypatch):
    """A crash inside the checker must not fail the translation it was only
    ever advising on."""
    class Angry:
        def check(self, text):
            raise RuntimeError("server died")

    monkeypatch.setattr(srv, "_lt_tools", {"de": Angry()})
    monkeypatch.setattr(srv, "LT_ENABLED", True)
    assert srv.languagetool_issues("Der Haus ist grün.", "de") == []


# --- mode strings the UI should never send, and a browser might --------------

def test_a_malformed_multi_target_mode_falls_back_to_the_default():
    """A duplicated or unknown target in "de-en+xx" is not worth guessing at;
    the default pair is what the app opens with anyway."""
    assert srv.resolve_targets("de-en+en", None) == ("de", ["en"])
    assert srv.resolve_targets("de-en+zz", None) == ("de", ["en"])


def test_a_segment_that_collapses_to_a_stub_is_dropped():
    """Not degenerate by the long-unit test, but collapsing "Und, und, und…"
    leaves nine characters. A card that short is an interruption, not content."""
    assert srv._clean_segment("Und, " * 10) is None


# --- models that will not load ------------------------------------------------

class FakeParakeet:
    preprocessor_config = {"sample_rate": 16000}

    def __init__(self):
        self.calls = []

    def generate(self, mel):
        self.calls.append(mel)
        return [type("R", (), {"text": "  ein Satz  "})()]


@pytest.fixture
def parakeet_reset(monkeypatch):
    """load_parakeet() memoizes into module globals, including the "gave up"
    flag, so every test here has to start from a clean slate. The autouse
    guard in conftest replaces the loader itself to keep the real multi-second
    download out of the suite; these tests are about the loader, so they put
    the genuine one back and substitute the module it imports instead."""
    from conftest import _REAL_LOAD_PARAKEET
    monkeypatch.setattr(srv, "load_parakeet", _REAL_LOAD_PARAKEET)
    monkeypatch.setattr(srv, "_parakeet", None)
    monkeypatch.setattr(srv, "_parakeet_unavailable", False)


def test_a_missing_partial_model_is_remembered_not_retried(parakeet_reset,
                                                           monkeypatch):
    """parakeet_mlx is an optional dependency. When it is absent, partials
    fall back to Whisper, and the failed import must happen once rather than
    on every partial window."""
    import sys
    monkeypatch.setitem(sys.modules, "parakeet_mlx", None)
    assert srv.load_parakeet() is None
    assert srv._parakeet_unavailable is True
    assert srv.load_parakeet() is None            # the remembered fast path


def test_the_partial_model_loads_once_and_is_reused(parakeet_reset,
                                                    monkeypatch):
    import sys
    import types
    loads = []
    module = types.ModuleType("parakeet_mlx")
    module.from_pretrained = lambda repo: (loads.append(repo), FakeParakeet())[1]
    monkeypatch.setitem(sys.modules, "parakeet_mlx", module)
    first = srv.load_parakeet()
    assert first is srv.load_parakeet()           # second call is the fast path
    assert loads == [srv.PARTIAL_ASR_REPO]


def test_transcribe_partial_without_a_model_defers_to_whisper(parakeet_reset,
                                                              monkeypatch):
    monkeypatch.setattr(srv, "load_parakeet", lambda: None)
    assert srv.transcribe_partial(np.zeros(16000, dtype=np.float32)) is None


def test_transcribe_partial_returns_the_models_text(parakeet_reset,
                                                    monkeypatch):
    """The mlx imports live inside the function so the server starts on a
    machine without them; this substitutes both and checks the text is
    stripped rather than passed through raw."""
    import sys
    import types
    model = FakeParakeet()
    monkeypatch.setattr(srv, "load_parakeet", lambda: model)
    mx = types.ModuleType("mlx.core")
    mx.array = lambda a: a
    core_pkg = types.ModuleType("mlx")
    core_pkg.core = mx
    audio_mod = types.ModuleType("parakeet_mlx.audio")
    audio_mod.get_logmel = lambda arr, cfg: ("mel", cfg)
    monkeypatch.setitem(sys.modules, "mlx", core_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "parakeet_mlx.audio", audio_mod)
    assert srv.transcribe_partial(np.zeros(16000, dtype=np.float32)) == "ein Satz"
    assert model.calls == [("mel", model.preprocessor_config)]


def test_transcribe_passes_the_prompt_and_language_to_whisper(monkeypatch):
    """The one place mlx_whisper is called. Stubbed everywhere else in the
    suite, so without this the argument names are never checked against a
    real call."""
    import sys
    import types
    seen = {}
    module = types.ModuleType("mlx_whisper")
    module.transcribe = lambda audio, **kw: (seen.update(kw),
                                             {"text": "Hallo"})[1]
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    out = srv.transcribe(np.zeros(16000, dtype=np.float32), "de",
                         prompt="Vorher gesagt")
    assert out["text"] == "Hallo"
    assert seen["language"] == "de"
    assert seen["initial_prompt"] == "Vorher gesagt"
    assert seen["path_or_hf_repo"] == srv.WHISPER_REPO


# --- the VAD, including the model interface we no longer ship ----------------

class FakeSession:
    """An onnxruntime session double. `v5` picks which input interface it
    declares, which is how SileroScorer decides how to call it."""

    def __init__(self, v5=True, prob=0.9):
        self.v5 = v5
        self.prob = prob
        self.calls = []

    def get_inputs(self):
        names = ["input", "state", "sr"] if self.v5 else ["input", "sr", "h", "c"]
        return [type("I", (), {"name": n})() for n in names]

    def run(self, outputs, feed):
        self.calls.append(feed)
        p = np.array([[self.prob]], dtype=np.float32)
        if self.v5:
            return p, np.zeros((2, 1, 128), dtype=np.float32)
        return (p, np.zeros((2, 1, 64), dtype=np.float32),
                np.zeros((2, 1, 64), dtype=np.float32))


def test_the_older_silero_interface_still_scores_frames():
    """The h/c interface predates the shipped v5 model. It is detected from
    the model's declared inputs, so a user with an older cached onnx file
    still gets neural VAD instead of a crash at the first frame."""
    session = FakeSession(v5=False)
    scorer = srv.SileroScorer(session)
    assert scorer.v5 is False
    assert scorer(np.zeros(srv.FRAME_SAMPLES, dtype=np.int16)) is True
    assert set(session.calls[0]) == {"input", "sr", "h", "c"}


def test_the_energy_vad_is_used_when_silero_is_switched_off(monkeypatch):
    monkeypatch.setattr(srv, "VAD_BACKEND", "energy")
    monkeypatch.setattr(srv, "_silero_session", None)
    assert srv.load_silero() is None
    assert isinstance(srv.make_scorer(), srv.EnergyScorer)


def test_a_silero_that_will_not_load_falls_back_to_energy(monkeypatch):
    """Any failure here is survivable: the energy VAD is worse at real speech
    but the app still runs, which is the whole point of the fallback."""
    import sys
    monkeypatch.setattr(srv, "VAD_BACKEND", "silero")
    monkeypatch.setattr(srv, "_silero_session", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    srv.load_silero()
    assert srv._silero_session is None
    assert isinstance(srv.make_scorer(), srv.EnergyScorer)


def test_silero_is_downloaded_once_then_loaded(tmp_path, monkeypatch):
    """First run on a new machine fetches the onnx file. The smoke-test call
    inside load_silero is what catches a corrupt download, so the fake session
    has to survive being called."""
    import sys
    import types
    session = FakeSession()
    module = types.ModuleType("onnxruntime")
    module.InferenceSession = lambda path, providers: session
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    monkeypatch.setattr(srv, "VAD_BACKEND", "silero")
    monkeypatch.setattr(srv, "_silero_session", None)
    monkeypatch.setattr(srv, "SILERO_PATH", tmp_path / "sub" / "silero.onnx")
    monkeypatch.setattr(srv.httpx, "get",
                        lambda url, **kw: httpx.Response(
                            200, content=b"onnx",
                            request=httpx.Request("GET", url)))
    srv.load_silero()
    assert srv.SILERO_PATH.read_bytes() == b"onnx"
    assert srv._silero_session is session
    assert isinstance(srv.make_scorer(), srv.SileroScorer)
    monkeypatch.setattr(srv, "_silero_session", None)


# --- the agreement retry that does not help ----------------------------------

def test_an_agreement_retry_that_does_not_fix_it_keeps_the_original(monkeypatch):
    """The retry is only accepted if it verifies clean. A model that returns
    something equally wrong must not overwrite a translation the user can
    already read."""
    async def still_wrong(*args, **kwargs):
        return "Der Haus ist grün."

    async def always_issues(text, target):
        return ["Haus is neuter"]

    monkeypatch.setattr(srv, "translate_once", still_wrong)
    monkeypatch.setattr(srv, "_combined_issues", always_issues)
    out = asyncio.run(srv.enforce_agreement("Das Haus ist grün.", "de", "de",
                                            "gemma3:12b", None,
                                            "Der Haus ist grün."))
    assert out == ("Der Haus ist grün.", False)


# --- the socket layer, when the work behind it raises ------------------------

def test_a_transcription_that_raises_tells_the_client(client, stub_transcribe,
                                                      monkeypatch):
    """Whisper failing mid-conversation must produce an error card, not a
    silent gap where the user keeps waiting for a translation."""
    def explode(audio, language, prompt=None):
        raise RuntimeError("mlx blew up")

    monkeypatch.setattr(stub_transcribe, "side_effect", explode, raising=False)
    monkeypatch.setattr(srv, "transcribe", explode)
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    err = [m for m in msgs if m["type"] == "error"]
    assert err and "mlx blew up" in err[0]["message"]


def test_typed_text_that_raises_tells_the_client(client, monkeypatch):
    """Same guarantee on the typed path, which has none of the audio
    machinery and so needs its own handler."""
    def explode(*args, **kwargs):
        raise RuntimeError("translation exploded")

    # run_translations is a closure over the socket, so the failure is
    # injected at the first module-level call inside the same try block.
    monkeypatch.setattr(srv, "dialect_markers", explode)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "text", "text": "Wie geht es dir?"}))
        msgs = collect_until(ws)
    err = [m for m in msgs if m["type"] == "error"]
    assert err and "translation exploded" in err[0]["message"]


def test_a_nonsense_pause_setting_is_ignored(client):
    """The pause length comes from the UI as a number. A string or a null
    must leave the previous value in place rather than crash the session that
    is already running."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en",
                                 "pause_ms": "soon"}))
        speak(ws)
        msgs = collect_until(ws)
    assert any(m["type"] == "final" for m in msgs)


def test_an_improve_tap_that_raises_still_answers(client, monkeypatch):
    """The tap's contract is that it always replies. An exception inside the
    improve call must still produce an `improved` message, or the spinner in
    the card never stops."""
    async def explode(*args, **kwargs):
        raise RuntimeError("improve exploded")

    monkeypatch.setattr(srv, "translate_once", explode)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "improve", "id": 1,
                                 "text": "Wie geht es dir?",
                                 "source": "de", "target": "en"}))
        replies = collect_until_bounded(ws, "improved", seconds=5.0)
    assert [m for m in replies if m["type"] == "improved"]


def test_a_recap_that_raises_reports_failure(client, monkeypatch):
    async def explode(*args, **kwargs):
        raise RuntimeError("recap exploded")

    monkeypatch.setattr(srv, "translate_once", explode)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "recap", "text": "Was war das?",
                                 "source": "de", "target": "en"}))
        replies = collect_until_bounded(ws, "recap", seconds=5.0)
    assert [m for m in replies if m.get("type") == "recap"
            and m.get("error") == "failed"]


def test_two_callers_racing_the_partial_load_only_load_once(parakeet_reset,
                                                            monkeypatch):
    """The lock exists because a second caller must not start a duplicate
    multi-second load. This stands in for the loser of that race: by the time
    it holds the lock the model is already there, and it must return the
    loaded model rather than loading again."""
    import sys
    import types
    loads = []
    module = types.ModuleType("parakeet_mlx")
    module.from_pretrained = lambda repo: (loads.append(repo), FakeParakeet())[1]
    monkeypatch.setitem(sys.modules, "parakeet_mlx", module)
    winner = FakeParakeet()

    class HandOverLock:
        def __enter__(self):
            srv._parakeet = winner      # the other thread finished while we waited

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(srv, "_parakeet_lock", HandOverLock())
    assert srv.load_parakeet() is winner
    assert loads == []                  # nothing loaded a second time


def test_the_partial_model_is_warmed_at_startup(monkeypatch):
    """The first MLX call compiles the graph. Warming it at startup is why the
    first partial of a conversation is not the slow one."""
    from fastapi.testclient import TestClient
    warmed = []
    monkeypatch.setattr(srv, "load_silero", lambda: None)
    monkeypatch.setattr(srv, "transcribe", lambda audio, language=None,
                        prompt=None: {"text": "", "language": "en"})
    monkeypatch.setattr(srv, "load_parakeet", lambda: FakeParakeet())
    monkeypatch.setattr(srv, "transcribe_partial",
                        lambda audio: warmed.append(len(audio)))
    with TestClient(srv.app):
        srv.partial_executor.submit(lambda: None).result()
    assert warmed == [srv.SAMPLE_RATE]


def test_languagetool_is_built_once_per_target(monkeypatch):
    """The opt-in path. Building the tool is expensive (it starts a Java
    process), so it happens on first use and is then reused."""
    import sys
    import types
    built = []
    module = types.ModuleType("language_tool_python")
    module.LanguageTool = lambda lang: (built.append(lang), object())[1]
    monkeypatch.setitem(sys.modules, "language_tool_python", module)
    monkeypatch.setattr(srv, "_lt_tools", {})
    first = srv._lt_tool("de")
    assert first is not None
    assert srv._lt_tool("de") is first
    assert built == [srv.LT_LANGS["de"]]


def test_a_base_that_outlasts_the_chain_wait_is_translated_in_full(
        client, stub_transcribe, fake_ollama, monkeypatch):
    """A merge link waits for its in-flight base so it can stitch, but the
    wait is bounded: if the base is still not finished, the successor has to
    translate itself whole rather than hang on a future that may never
    resolve. The timeout is what stops one slow translate from stalling every
    card behind it."""
    real = srv.stream_translation

    async def slow_first(ws, uid, text, *a, **kw):
        if text.startswith("Ich glaube"):
            await asyncio.sleep(0.5)      # far longer than the wait below
        return await real(ws, uid, text, *a, **kw)

    monkeypatch.setattr(srv, "stream_translation", slow_first)
    monkeypatch.setattr(srv, "CHAIN_WAIT_SEC", 0.01)
    stub_transcribe.result = {"text": "Ich glaube, dass wir das Projekt",
                              "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        stub_transcribe.result = {"text": "nächste Woche abschließen werden.",
                                  "language": "de"}
        speak(ws)
        msgs = collect_until(ws, stop_types=("translation_done",), limit=400)
    finals = [m for m in msgs if m["type"] == "final"]
    assert finals[-1]["replaces"] == finals[0]["id"]
    # No base was ready to replay, so the whole merged card went to the model.
    sent = [m["content"] for b in fake_ollama["all"]
            for m in b["messages"] if m["role"] == "user"]
    assert ("Ich glaube, dass wir das Projekt "
            "nächste Woche abschließen werden.") in sent


def test_a_client_that_vanishes_ends_the_session_quietly(client,
                                                         stub_transcribe):
    """The real drop path: the disconnect message carries no bytes, and that
    is what ends the receive loop. A lost phone must cost nothing louder than
    the session ending."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en",
                                 "stats": True}))
        speak(ws)
        collect_until(ws)
        ws.close()


def test_a_dropped_connection_ends_the_endpoint_without_raising():
    """Defensive redundancy, pinned deliberately.

    The ordinary drop already ends the session without this handler:
    `ws.receive()` returns the disconnect *message* rather than raising
    (Starlette only raises WebSocketDisconnect from receive_text/bytes/json,
    which this loop does not use), and the loop breaks on it having no bytes.
    This covers the case where the call does raise instead, so the endpoint
    ends quietly either way: no traceback, and no error sent into a socket
    that is already gone."""
    from fastapi import WebSocketDisconnect

    class Vanishes:
        def __init__(self):
            self.sent = []
            self.accepted = False

        async def accept(self):
            self.accepted = True

        async def receive(self):
            raise WebSocketDisconnect(1006)

        async def send_json(self, payload):
            self.sent.append(payload)

    ws = Vanishes()
    asyncio.run(srv.ws_endpoint(ws))     # must return, not raise
    assert ws.accepted and ws.sent == []
