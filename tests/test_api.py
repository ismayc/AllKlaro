"""/api endpoints and the Ollama translation streamer, error paths included."""
from fastapi.testclient import TestClient

import server
from conftest import FakeWS


def test_models_listed_and_embeddings_filtered(client):
    data = client.get("/api/models").json()
    assert data["default"] == server.DEFAULT_MODEL
    assert "gemma3:12b" in data["models"]
    assert "nomic-embed-text:latest" not in data["models"]
    assert "error" not in data


def test_models_reports_ollama_down(dead_ollama):
    data = TestClient(server.app).get("/api/models").json()
    assert data["models"] == []
    assert "Cannot reach Ollama" in data["error"]


# ------------------------------------------------------------- translation


async def test_translation_streams_and_returns_text(fake_ollama):
    ws = FakeWS()
    result = await server.stream_translation(
        ws, 7, "Wie geht es dir?", "de", "en", "gemma3:12b")
    assert result == "How are you?"
    deltas = [m for m in ws.sent if m["type"] == "translation_delta"]
    assert "".join(m["text"] for m in deltas) == "How are you?"
    assert all(m["id"] == 7 and m["target"] == "en" for m in deltas)
    # The caller owns translation_done (it carries the latency metrics).
    assert not any(m["type"] == "translation_done" for m in ws.sent)


async def test_translation_missing_model_reports_error(fake_ollama):
    ws = FakeWS()
    result = await server.stream_translation(ws, 1, "Hallo", "de", "en", "missing:1b")
    assert result is None
    assert ws.sent[0]["type"] == "error"
    assert "Ollama error" in ws.sent[0]["message"]


async def test_translation_ollama_down_reports_error(dead_ollama):
    ws = FakeWS()
    result = await server.stream_translation(ws, 1, "Hallo", "de", "en", "gemma3:12b")
    assert result is None
    assert ws.sent[-1]["type"] == "error"
    assert "Cannot reach Ollama" in ws.sent[-1]["message"]


async def test_thinking_disabled_for_reasoning_models(fake_ollama):
    await server.stream_translation(FakeWS(), 1, "Hallo", "de", "en", "qwen3:8b")
    assert fake_ollama["chat"]["think"] is False

    await server.stream_translation(FakeWS(), 1, "Hallo", "de", "en", "gemma3:12b")
    assert "think" not in fake_ollama["chat"]


async def test_history_context_reaches_the_model(fake_ollama):
    history = [{"source": "de", "target": "en",
                "text": "Wir brauchen einen neuen Server.",
                "translation": "We need a new server."}]
    await server.stream_translation(
        FakeWS(), 2, "Er sollte schnell sein.", "de", "en", "gemma3:12b", history)
    messages = fake_ollama["chat"]["messages"]
    assert messages[1] == {"role": "user",
                           "content": "Wir brauchen einen neuen Server."}
    assert messages[2] == {"role": "assistant",
                           "content": "We need a new server."}
    assert messages[-1] == {"role": "user", "content": "Er sollte schnell sein."}


# ------------------------------------------------------------------ export


def test_export_renders_markdown(client):
    items = [{"time": "10:00:00", "speaker": "You", "source": "de",
              "target": "en", "text": "Hallo zusammen.",
              "translation": "Hello everyone."}]
    md = client.post("/api/export", json={"items": items}).json()["markdown"]
    assert "**[10:00:00] You (DE):** Hallo zusammen." in md
    assert "→ **(EN):** Hello everyone." in md


def test_export_multi_target_and_summary(client):
    items = [{"time": "10:01:00", "source": "de", "text": "Guten Morgen.",
              "translations": {"en": "Good morning.", "es": "Buenos días."}}]
    md = client.post("/api/export",
                     json={"items": items, "summary": "Short chat."}).json()["markdown"]
    assert "→ **(EN):** Good morning." in md
    assert "→ **(ES):** Buenos días." in md
    assert "## Summary" in md and "Short chat." in md


def test_export_tolerates_garbage(client):
    md = client.post("/api/export",
                     json={"items": "nope", "summary": 42}).json()["markdown"]
    assert md.startswith("# Conversation transcript")


# ---------------------------------------------------------------- summarize


def test_summarize_returns_summary(client, fake_ollama):
    items = [{"source": "de", "text": "Wir treffen uns nächste Woche."},
             {"source": "en", "text": "Sounds good."}]
    data = client.post("/api/summarize",
                       json={"items": items, "model": "gemma3:12b"}).json()
    assert "Summary" in data["summary"]
    body = fake_ollama["chat"]
    assert body["stream"] is False
    assert "Vocabulary" in body["messages"][0]["content"]
    assert "[DE] Wir treffen uns nächste Woche." in body["messages"][1]["content"]


def test_summarize_empty_conversation(client):
    data = client.post("/api/summarize", json={"items": []}).json()
    assert "Nothing to summarize" in data["error"]


def test_summarize_ollama_down(stub_transcribe, dead_ollama):
    data = TestClient(server.app).post(
        "/api/summarize", json={"items": [{"source": "de", "text": "Hallo"}]}).json()
    assert "Cannot reach Ollama" in data["error"]


# ------------------------------------------------------------- corrections


def test_correction_endpoint_saves_and_counts(client, corrections_file):
    payload = {"source": "de", "target": "en", "text": "Guten Morgen.",
               "model_translation": "Good morning.", "corrected": "Morning!"}
    data = client.post("/api/correction", json=payload).json()
    assert data == {"ok": True, "count": 1}
    saved = server.load_corrections()[0]
    assert saved["corrected"] == "Morning!"
    assert saved["model_translation"] == "Good morning."


def test_correction_endpoint_rejects_garbage(client, corrections_file):
    bad = [
        {"source": "fr", "target": "en", "text": "x", "corrected": "y"},
        {"source": "de", "target": "de", "text": "x", "corrected": "y"},
        {"source": "de", "target": "en", "text": "  ", "corrected": "y"},
        {"source": "de", "target": "en", "text": "x", "corrected": ""},
        {"source": "de", "target": "en", "text": "x", "corrected": 3},
    ]
    for payload in bad:
        assert "error" in client.post("/api/correction", json=payload).json()
    assert not corrections_file.exists()      # nothing was written


def test_models_report_sizes_for_draft_suggestion(client):
    data = client.get("/api/models").json()
    assert data["sizes"]["qwen2.5:7b-instruct"] < data["sizes"]["gemma3:12b"]
    assert "nomic-embed-text:latest" not in data["sizes"]
