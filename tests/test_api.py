"""/api endpoints and the Ollama translation streamer, error paths included."""
from fastapi.testclient import TestClient

import server
from conftest import FakeWS


def test_default_model_is_env_overridable():
    # ALLKLARO_MODEL turns the integration suite into a model benchmark.
    import inspect
    src = inspect.getsource(server)
    assert 'os.environ.get("ALLKLARO_MODEL"' in src


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


# ------------------------------------------------------------ cert download


def test_cert_endpoint_serves_certificate(client, tmp_path, monkeypatch):
    pem = tmp_path / "cert.pem"
    pem.write_text("-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")
    monkeypatch.setattr(server, "CERT_PATH", pem)
    r = client.get("/cert")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-x509-ca-cert")
    assert b"BEGIN CERTIFICATE" in r.content


def test_cert_endpoint_404_without_phone_mode(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "CERT_PATH", tmp_path / "missing.pem")
    assert client.get("/cert").status_code == 404

def test_static_assets_must_be_revalidated_not_cached(client):
    # iOS home-screen apps replay stale cached JS for days otherwise —
    # shipped features would silently vanish on the phone after updates.
    for path in ("/", "/static/app.js", "/static/style.css"):
        assert client.get(path).headers.get("cache-control") == "no-cache", path

def test_index_stamps_asset_versions_for_cache_busting(client):
    # Browsers revalidate the HTML on navigation but can keep subresources
    # cached (seen on iOS: new HTML, old CSS/JS). A changed ?v= forces a
    # fresh CSS/JS fetch as soon as the new HTML lands.
    import re
    html = client.get("/").text
    assert "__BUILD__" not in html
    assert re.search(r'/static/app\.js\?v=\d+', html)
    assert re.search(r'/static/style\.css\?v=\d+', html)

# ---------------------------------------------------- /api/translate (Shortcut)


def test_api_translate_stateless(client, fake_ollama):
    r = client.post("/api/translate",
                    json={"text": "Wie geht es dir und der Familie?"}).json()
    assert r["source"] == "de" and r["target"] == "en"
    assert r["translation"] == "Refined translation."
    assert r["translations"] == {"en": "Refined translation."}
    # Stateless: just the system prompt and the sentence, no history turns.
    assert len(fake_ollama["chat"]["messages"]) == 2


def test_api_translate_forced_mode_carries_flavor(client, fake_ollama):
    r = client.post("/api/translate",
                    json={"text": "See you tomorrow!", "mode": "en-de",
                          "de_flavor": "berlin"}).json()
    assert (r["source"], r["target"]) == ("en", "de")
    assert "Berlinerisch" in fake_ollama["chat"]["messages"][0]["content"]


def test_api_translate_multi_target_mode(client, fake_ollama):
    r = client.post("/api/translate",
                    json={"text": "Hello there!", "mode": "en-de+es"}).json()
    assert set(r["translations"]) == {"de", "es"}
    assert r["target"] == "de"


def test_api_translate_rejects_empty_text(client):
    assert "error" in client.post("/api/translate", json={"text": "   "}).json()
    assert "error" in client.post("/api/translate", json={}).json()


def test_api_translate_reports_ollama_down(dead_ollama):
    r = TestClient(server.app).post(
        "/api/translate", json={"text": "Hallo, wie geht's dir denn?"}).json()
    assert "available" in r["error"]


def test_api_translate_display_field_is_captioned(client, fake_ollama):
    r = client.post("/api/translate",
                    json={"text": "Wie geht es dir und der Familie?"}).json()
    assert r["display"] == "🗣️ AllKlaro (DE → EN):\nRefined translation."


def test_api_translate_reports_detection_confidence(client, fake_ollama):
    """The share-sheet Shortcut gets no card to tap, so it gets the number
    instead and can re-ask with an explicit source when it looks shaky."""
    sure = client.post("/api/translate",
                       json={"text": "Wie geht es dir und der Familie?"}).json()
    assert sure["source"] == "de" and sure["confidence"] > 0.9
    # A forced direction detects nothing, so there is nothing to report.
    forced = client.post("/api/translate",
                         json={"text": "Guten Morgen", "mode": "de-en"}).json()
    assert forced["confidence"] is None


def test_api_translate_source_override(client, fake_ollama):
    plain = client.post("/api/translate", json={"text": "Happy birthday!"}).json()
    assert plain["source"] == "en"           # detection gets it right now
    pinned = client.post("/api/translate",
                         json={"text": "Happy birthday!", "source": "de"}).json()
    assert (pinned["source"], pinned["target"]) == ("de", "en")
    assert pinned["confidence"] is None      # pinned, not guessed
    # A source outside the mode's pair is ignored rather than obeyed.
    off_pair = client.post("/api/translate",
                           json={"text": "Happy birthday!", "source": "es"}).json()
    assert off_pair["source"] == "en"


def test_api_translate_get_variant_for_browser_smoke_tests(client, fake_ollama):
    r = client.get("/api/translate",
                   params={"text": "Wie geht es dir und der Familie?"}).json()
    assert r["source"] == "de" and r["translation"] == "Refined translation."
    assert "error" in client.get("/api/translate").json()


def test_api_translate_pins_the_address_form(client, fake_ollama):
    client.post("/api/translate", json={"text": "Can you help me tomorrow?",
                                        "mode": "en-es",
                                        "address": "plural"})
    assert '"ustedes"' in fake_ollama["chat"]["messages"][0]["content"]


def test_api_translate_spanish_flavor(client, fake_ollama):
    client.post("/api/translate", json={"text": "That's cool!",
                                        "mode": "en-es",
                                        "es_flavor": "barcelona"})
    assert "Barcelona" in fake_ollama["chat"]["messages"][0]["content"]
