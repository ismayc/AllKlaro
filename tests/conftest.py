import json

import anyio
import httpx
import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

import server

# 2048 samples = 128 ms @ 16 kHz, same chunk size the browser worklet sends.
SILENCE_CHUNK = np.zeros(2048, dtype=np.int16).tobytes()
_t = np.arange(2048)
SPEECH_CHUNK = (8000 * np.sin(2 * np.pi * 440 * _t / 16000)).astype(np.int16).tobytes()


class TranscribeStub:
    """Stands in for mlx-whisper; records calls, returns a configurable result."""

    def __init__(self):
        self.result = {"text": "Wie geht es dir?", "language": "de"}
        self.queue = []      # per-call results consumed before self.result
        self.calls = []

    def __call__(self, audio, language, prompt=None):
        self.calls.append({"seconds": len(audio) / 16000,
                           "language": language, "prompt": prompt})
        return dict(self.queue.pop(0)) if self.queue else dict(self.result)


@pytest.fixture
def stub_transcribe(monkeypatch):
    stub = TranscribeStub()
    monkeypatch.setattr(server, "transcribe", stub)
    return stub


class PartialStub:
    """Stands in for the fast partial-pass ASR (Parakeet); records call sizes."""

    def __init__(self):
        self.text = "live partial text"
        self.calls = []

    def __call__(self, audio):
        self.calls.append(round(len(audio) / 16000, 2))
        return self.text


@pytest.fixture(autouse=True)
def no_real_partial_asr(monkeypatch):
    """Tests must never download or load the real Parakeet model. Default is
    the fallback path, so a test that says nothing gets Whisper partials via
    the existing stub; `stub_partial` opts into the fast path."""
    monkeypatch.setattr(server, "_parakeet", None)
    monkeypatch.setattr(server, "_parakeet_unavailable", True)
    monkeypatch.setattr(server, "load_parakeet", lambda: None)


@pytest.fixture
def stub_partial(monkeypatch):
    """Opt into the fast partial path with a doubled model."""
    stub = PartialStub()
    monkeypatch.setattr(server, "transcribe_partial", stub)
    monkeypatch.setattr(server, "_parakeet_unavailable", False)
    return stub


# Captured before any fixture can patch it, so the integration suite can put
# the genuine loader back.
_REAL_LOAD_PARAKEET = server.load_parakeet


@pytest.fixture
def real_partial_asr(no_real_partial_asr, monkeypatch):
    """Integration only: undo the guard and use the real Parakeet model."""
    monkeypatch.setattr(server, "load_parakeet", _REAL_LOAD_PARAKEET)
    monkeypatch.setattr(server, "_parakeet_unavailable", False)


@pytest.fixture(autouse=True)
def no_user_gender_lexicon(tmp_path, monkeypatch):
    """Tests must not see the developer's real ~/.cache lexicons."""
    monkeypatch.setattr(server, "GENDER_LEXICON_PATHS",
                        {t: tmp_path / f"no-lexicon-{t}.tsv"
                         for t in ("de", "es")})
    monkeypatch.setattr(server, "OUTPUT_GENDER_PATHS",
                        {t: tmp_path / f"no-output-{t}.tsv"
                         for t in ("de", "es")})
    monkeypatch.setattr(server, "NOUN_FORMS_PATH", tmp_path / "no-forms.tsv")
    monkeypatch.setattr(server, "WIKTIONARY_PATHS",
                        {t: tmp_path / f"no-wikt-{t}.sqlite"
                         for t in ("de", "en", "es")})
    server._gender_caches.clear()
    server._output_caches.clear()
    server._noun_forms_cache.update(mtime=None, map={})
    server._wikt_conns.clear()
    yield
    server._gender_caches.clear()
    server._output_caches.clear()
    server._noun_forms_cache.update(mtime=None, map={})
    server._wikt_conns.clear()


@pytest.fixture(autouse=True)
def trace_file(tmp_path, monkeypatch):
    """Tests must never append to the developer's real pipeline trace."""
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(server, "TRACE_PATH", str(path))
    return path


def trace_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text().splitlines() if line.strip()]


@pytest.fixture
def output_gender_map(no_user_gender_lexicon, tmp_path, monkeypatch):
    """Write a small output-side gender map: write(entries, target="de")."""
    def write(entries, target="de"):
        path = tmp_path / f"output-{target}.tsv"
        path.write_text("".join(f"{w.lower()}\t{w}\t{g}\n"
                                for w, g in entries))
        server.OUTPUT_GENDER_PATHS[target] = path
        server._output_caches.clear()

    return write


@pytest.fixture
def gender_lexicon(no_user_gender_lexicon, tmp_path, monkeypatch):
    """Write a small per-target lexicon file: write(entries, target="de")."""
    def write(entries, target="de"):
        path = tmp_path / f"genders-{target}.tsv"
        path.write_text("".join(f"{k}\t{w}\t{g}\n" for k, w, g in entries))
        server.GENDER_LEXICON_PATHS[target] = path
        server._gender_caches.clear()

    return write


@pytest.fixture
def noun_forms(no_user_gender_lexicon, tmp_path, monkeypatch):
    """Write a small German noun-paradigm table:
    write([(form, lemma, gender, "ns,as"), ...])."""
    def write(entries):
        path = tmp_path / "noun-forms.tsv"
        path.write_text("".join(f"{f.lower()}\t{l}\t{g}\t{c}\n"
                                for f, l, g, c in entries))
        monkeypatch.setattr(server, "NOUN_FORMS_PATH", path)
        server._noun_forms_cache.update(mtime=None, map={})

    return write


@pytest.fixture
def corrections_file(tmp_path, monkeypatch):
    """Point the corrections store at a scratch file with a clean cache."""
    path = tmp_path / "corrections.jsonl"
    monkeypatch.setattr(server, "CORRECTIONS_PATH", path)
    server._corrections_cache.update(mtime=None, items=[])
    yield path
    server._corrections_cache.update(mtime=None, items=[])


@pytest.fixture
def fake_ollama(monkeypatch):
    """In-process Ollama double; returns a dict recording the last /api/chat body."""
    record = {}
    fake = FastAPI()

    @fake.get("/api/tags")
    async def tags():
        return {"models": [{"name": "gemma3:12b", "size": 8_100_000_000},
                           {"name": "qwen2.5:7b-instruct", "size": 4_700_000_000},
                           {"name": "nomic-embed-text:latest", "size": 274_000_000}]}

    @fake.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        record["chat"] = body
        record.setdefault("all", []).append(body)
        if "missing" in body["model"]:
            return JSONResponse({"error": f"model '{body['model']}' not found"},
                                status_code=404)
        if body.get("stream", True) is False:
            if body["messages"][0]["content"].startswith("You summarize"):
                return JSONResponse({"message": {"content":
                    "**Summary** They planned a project meeting.\n"
                    "**Vocabulary** das Projekt — the project"}, "done": True})
            if body["messages"][0]["content"].startswith("You keep a running"):
                return JSONResponse({"message": {"content":
                    "• They discussed the garden."}, "done": True})
            # Non-streaming translation = the refinement pass.
            return JSONResponse({"message": {"content": "Refined translation."},
                                 "done": True})

        async def gen():
            for piece in ["How ", "are ", "you?"]:
                yield json.dumps({"message": {"content": piece}, "done": False}) + "\n"
            yield json.dumps({"message": {"content": ""}, "done": True}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    transport = httpx.ASGITransport(app=fake)
    monkeypatch.setattr(
        server, "ollama_client",
        lambda: httpx.AsyncClient(transport=transport, base_url="http://ollama.test"))
    return record


@pytest.fixture
def dead_ollama(monkeypatch):
    """Ollama that refuses every connection."""

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(refuse)
    monkeypatch.setattr(
        server, "ollama_client",
        lambda: httpx.AsyncClient(transport=transport, base_url="http://ollama.test"))


@pytest.fixture
def client(stub_transcribe, fake_ollama, monkeypatch):
    # Unit tests use the deterministic energy VAD (synthetic tones aren't
    # "speech" to Silero) and must not download the Silero model.
    monkeypatch.setattr(server, "load_silero", lambda: None)
    monkeypatch.setattr(server, "make_scorer", lambda: server.EnergyScorer())
    with TestClient(server.app) as c:
        # Drain the startup prewarm through the single-worker executor, then
        # forget its call so tests can assert on their own transcriptions.
        server.whisper_executor.submit(lambda: None).result()
        stub_transcribe.calls.clear()
        yield c


class FakeWS:
    """Collects everything the server tries to send over the socket."""

    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


def speak(ws, speech_chunks=10, silence_chunks=8):
    """Simulate one spoken utterance: brief silence, speech, closing silence."""
    for _ in range(3):
        ws.send_bytes(SILENCE_CHUNK)
    for _ in range(speech_chunks):
        ws.send_bytes(SPEECH_CHUNK)
    for _ in range(silence_chunks):
        ws.send_bytes(SILENCE_CHUNK)


def collect_until(ws, stop_types=("translation_done", "discard", "error"), limit=200):
    msgs = []
    for _ in range(limit):
        msg = ws.receive_json()
        msgs.append(msg)
        if msg["type"] in stop_types:
            break
    return msgs


def receive_bounded(ws, seconds=5.0):
    """One message, or TimeoutError — never an indefinite block.

    Some guarantees here are of the form "the server always answers": a tap
    that fails still gets a reply, so the button stops spinning. The obvious
    way to test that is to wait for the reply, and the obvious way is wrong —
    when the guarantee regresses there is no reply, and a plain `receive_json`
    hangs the whole suite instead of failing the one test. A mutation run is
    where that first bites, because breaking the guarantee on purpose is the
    entire point.
    """
    async def _recv():
        with anyio.fail_after(seconds):
            return await ws._send_rx.receive()

    return ws.portal.call(_recv)


def collect_until_bounded(ws, stop_types, limit=60, seconds=5.0):
    """Messages up to and including the first of `stop_types`, on a deadline.

    The bounded twin of `collect_until`. Worth preferring wherever the thing
    under test is whether a message arrives at all: the unbounded version
    turns "it stopped sending" into a hung suite instead of a failed test.
    """
    if isinstance(stop_types, str):
        stop_types = (stop_types,)
    msgs = []
    for _ in range(limit):
        raw = receive_bounded(ws, seconds)
        msgs.append(json.loads(raw["text"]) if "text" in raw else raw)
        if msgs[-1].get("type") in stop_types:
            return msgs
    raise AssertionError(
        f"no {stop_types} in {limit} messages: "
        f"{[m.get('type') for m in msgs]}")
