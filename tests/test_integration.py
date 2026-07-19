"""Slow tests against the real Whisper model and a live Ollama server.

Run with:  RUN_INTEGRATION=1 uv run pytest -m integration -v
"""
import json
import os
import subprocess
import wave

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from conftest import FakeWS, collect_until

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("RUN_INTEGRATION"),
                       reason="set RUN_INTEGRATION=1 to run integration tests"),
]

GERMAN_TEXT = ("Guten Tag! Ich wollte fragen, ob wir uns nächste Woche "
               "treffen können, um das Projekt zu besprechen.")
SPANISH_TEXT = ("¡Buenos días! Quería preguntar si podemos reunirnos la "
                "próxima semana para hablar del proyecto.")
SPANISH_VOICE = "Flo (Spanish (Spain))"


def tts_wav(tmp_path, text, voice):
    path = tmp_path / f"{voice}.wav"
    result = subprocess.run(
        ["say", "-v", voice, "-o", str(path), "--data-format=LEI16@16000", text],
        capture_output=True)
    if result.returncode != 0:
        pytest.skip(f"macOS TTS voice '{voice}' unavailable")
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000
        return w.readframes(w.getnframes())


def ollama_up():
    try:
        httpx.get(f"{server.OLLAMA_URL}/api/tags", timeout=3)
        return True
    except httpx.ConnectError:
        return False


def test_real_whisper_transcribes_german(tmp_path):
    pcm = tts_wav(tmp_path, GERMAN_TEXT, "Anna")
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    result = server.transcribe(audio, language=None)
    assert result["language"] == "de"
    assert "projekt" in result["text"].lower()
    assert "woche" in result["text"].lower()


def test_real_whisper_transcribes_spanish(tmp_path):
    pcm = tts_wav(tmp_path, SPANISH_TEXT, SPANISH_VOICE)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    result = server.transcribe(audio, language=None)
    assert result["language"] == "es"
    assert "proyecto" in result["text"].lower()
    assert "semana" in result["text"].lower()


async def test_real_ollama_translates_spanish():
    if not ollama_up():
        pytest.skip("Ollama not running")
    text = await server.stream_translation(
        FakeWS(), 1, "¡Buenos días! ¿Cómo está usted hoy?", "es", "en",
        server.DEFAULT_MODEL)
    assert "morning" in text.lower()


async def test_real_ollama_translates_german():
    if not ollama_up():
        pytest.skip("Ollama not running")
    text = await server.stream_translation(
        FakeWS(), 1, "Guten Morgen! Wie geht es Ihnen heute?", "de", "en",
        server.DEFAULT_MODEL)
    assert "morning" in text.lower()


async def test_real_ollama_dictionary_gender_overrides_generic_rule(
        gender_lexicon):
    if not ollama_up():
        pytest.skip("Ollama not running")
    # The generic -a rule says feminine, but dict.cc says der Caipirinha —
    # the per-word dictionary note must win.
    gender_lexicon([("caipirinha", "Caipirinha", "m")])
    text = (await server.stream_translation(
        FakeWS(), 1, "I'd like a caipirinha, please.", "en", "de",
        server.DEFAULT_MODEL)).lower()
    assert "einen caipirinha" in text and "eine caipirinha" not in text


async def test_real_ollama_spanish_gender_note_followed(gender_lexicon):
    if not ollama_up():
        pytest.skip("Ollama not running")
    # "mapa" is masculine despite -a; the es-lexicon note must be obeyed.
    gender_lexicon([("map", "mapa", "m")], target="es")
    text = (await server.stream_translation(
        FakeWS(), 1, "The map is very old.", "en", "es",
        server.DEFAULT_MODEL)).lower()
    assert "el mapa" in text and "la mapa" not in text


async def test_real_translate_once_refinement_pass():
    if not ollama_up():
        pytest.skip("Ollama not running")
    text = await server.translate_once(
        "Guten Morgen! Wie geht es Ihnen heute?", "de", "en",
        server.DEFAULT_MODEL)
    assert text and "morning" in text.lower()


async def test_real_ollama_gender_agreement_for_loanword_drinks(
        corrections_file, tmp_path, monkeypatch):
    if not ollama_up():
        pytest.skip("Ollama not running")
    # No glossary, no corrections: the GRAMMAR_NOTES rule alone must get
    # "die Margarita" (feminine) right — gemma3:12b says "der" without it.
    monkeypatch.setattr(server, "GLOSSARY_PATH", tmp_path / "no-glossary.txt")
    server._glossary_cache.update(mtime=None, lines=[])
    checks = [("I'd like a margarita, please.",
               "eine margarita", "einen margarita"),
              ("This margarita is really good.",
               "diese margarita", "dieser margarita")]
    for sentence, want, reject in checks:
        text = (await server.stream_translation(
            FakeWS(), 1, sentence, "en", "de", server.DEFAULT_MODEL)).lower()
        assert want in text and reject not in text, f"{sentence!r} -> {text!r}"
    server._glossary_cache.update(mtime=None, lines=[])


async def test_real_ollama_correction_steers_translation(corrections_file):
    if not ollama_up():
        pytest.skip("Ollama not running")
    # The user once corrected "Meeting" -> "stand-up"; a later sentence about
    # the Meeting should pick up that preferred wording via retrieval.
    server.save_correction({
        "source": "de", "target": "en",
        "text": "Das Meeting wurde auf Donnerstag verschoben.",
        "corrected": "The stand-up was moved to Thursday.",
        "model_translation": "The meeting was moved to Thursday."})
    text = await server.stream_translation(
        FakeWS(), 1, "Wann beginnt das Meeting?", "de", "en",
        server.DEFAULT_MODEL)
    assert "stand-up" in text.lower()


async def test_real_ollama_context_resolves_pronoun():
    if not ollama_up():
        pytest.skip("Ollama not running")
    history = [{"source": "de", "target": "en",
                "text": "Meine Schwester kommt morgen zu Besuch.",
                "translation": "My sister is visiting tomorrow."}]
    text = await server.stream_translation(
        FakeWS(), 2, "Sie bringt ihren Hund mit.", "de", "en",
        server.DEFAULT_MODEL, history)
    # With context, "Sie" must resolve to she (the sister), not formal you.
    assert "she" in text.lower()


def test_real_silero_vad_on_real_speech(tmp_path):
    server.load_silero()
    if server._silero_session is None:
        pytest.skip("Silero VAD unavailable")
    pcm = tts_wav(tmp_path, GERMAN_TEXT, "Anna")
    samples = np.frombuffer(pcm, dtype=np.int16)
    pad = np.zeros(16000, dtype=np.int16)  # 1 s of trailing silence
    samples = np.concatenate([pad, samples, pad])

    vad = server.VadSession(server.SileroScorer(server._silero_session))
    utterances = []
    for i in range(0, len(samples) - server.FRAME_SAMPLES,
                   server.FRAME_SAMPLES):
        utt = vad.feed(samples[i:i + server.FRAME_SAMPLES])
        if utt is not None:
            utterances.append(utt)
    assert utterances, "Silero VAD found no speech in real TTS audio"

    silent_vad = server.VadSession(server.SileroScorer(server._silero_session))
    for i in range(100):
        assert silent_vad.feed(np.zeros(server.FRAME_SAMPLES,
                                        dtype=np.int16)) is None
    assert not silent_vad.in_speech


async def test_real_summarize():
    if not ollama_up():
        pytest.skip("Ollama not running")
    from fastapi.testclient import TestClient as TC
    items = [
        {"source": "de", "text": "Können wir uns Dienstag um drei treffen?"},
        {"source": "en", "text": "Yes, Tuesday at three works for me."},
        {"source": "de", "text": "Perfekt, ich reserviere einen Raum."},
    ]
    data = TC(server.app).post(
        "/api/summarize", json={"items": items,
                                "model": server.DEFAULT_MODEL}).json()
    assert "error" not in data
    assert "tuesday" in data["summary"].lower()


def test_real_end_to_end_websocket(tmp_path):
    if not ollama_up():
        pytest.skip("Ollama not running")
    pcm = tts_wav(tmp_path, GERMAN_TEXT, "Anna")
    silence = bytes(4096)
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "config", "mode": "auto",
                                     "model": server.DEFAULT_MODEL}))
            for _ in range(8):
                ws.send_bytes(silence)
            for i in range(0, len(pcm), 4096):
                ws.send_bytes(pcm[i:i + 4096])
            for _ in range(12):
                ws.send_bytes(silence)
            msgs = collect_until(ws)

    final = next(m for m in msgs if m["type"] == "final")
    assert final["source"] == "de"
    assert "woche" in final["text"].lower()
    translated = "".join(m["text"] for m in msgs
                         if m["type"] == "translation_delta").lower()
    assert "week" in translated


async def test_real_agreement_guard_fixes_declension(output_gender_map):
    if not ollama_up():
        pytest.skip("Ollama not running")
    output_gender_map([("Margarita", "f")])
    final, changed = await server.enforce_agreement(
        "I'd like a margarita, please.", "en", "de", server.DEFAULT_MODEL,
        [], "Ich hätte gerne einen Margarita, bitte.")
    assert changed and "eine Margarita" in final


async def test_real_dialect_mishearing_negation_preserved():
    if not ollama_up():
        pytest.skip("Ollama not running")
    # Whisper heard Hessian "net verstanne" as "nett verstarne"; without the
    # dialect note gemma inverts the meaning to "understood nicely".
    text = (await server.stream_translation(
        FakeWS(), 1,
        "Ich hab des nett verstarne, kannste des nochemol saache?",
        "de", "en", server.DEFAULT_MODEL)).lower()
    assert "not" in text or "didn't" in text or "did not" in text
    assert "nicely" not in text and "nice" not in text


async def test_real_ollama_berlin_flavor_produces_dialect():
    if not ollama_up():
        pytest.skip("Ollama not running")
    text = await server.stream_translation(
        FakeWS(), 1, "I can't make it today, let's meet tomorrow instead.",
        "en", "de", server.DEFAULT_MODEL, flavor="berlin")
    # At least one unmistakable Berlin marker (verified live 2026-07-18:
    # gemma3:12b wrote "dit heite ... morjen up'm Markt").
    assert any(m in text.lower() for m in ("ick", "dit ", "wat ", "ooch",
                                           "nüscht", "morjen", "jut"))


async def test_real_ollama_hessian_flavor_produces_dialect():
    if not ollama_up():
        pytest.skip("Ollama not running")
    text = await server.stream_translation(
        FakeWS(), 1, "I can't make it today, let's meet tomorrow instead.",
        "en", "de", server.DEFAULT_MODEL, flavor="hessian")
    assert any(m in text.lower() for m in (" net ", "net,", "net.", "gell",
                                           "isch ", "aach", "ebbes"))


async def test_real_api_translate_endpoint():
    if not ollama_up():
        pytest.skip("Ollama not running")
    r = await server.translate_api(
        {"text": "Kannste morjen ooch bei uns vorbeikommen, dit wär jut."})
    assert r["source"] == "de" and r["target"] == "en"
    assert "tomorrow" in r["translation"].lower()


async def test_real_ollama_worms_flavor_produces_dialect():
    if not ollama_up():
        pytest.skip("Ollama not running")
    text = await server.stream_translation(
        FakeWS(), 1, "Are you coming to the wine festival tomorrow?",
        "en", "de", server.DEFAULT_MODEL, flavor="worms")
    assert any(m in text.lower() for m in ("woi", "morje", "kummst", "aach",
                                           " net", "bischt", "hoscht", "alla",
                                           "gell", "nää"))


async def test_real_ollama_address_forms_are_followed():
    if not ollama_up():
        pytest.skip("Ollama not running")
    text = "Can you send me the photos when you have time?"
    formal = await server.stream_translation(
        FakeWS(), 1, text, "en", "de", server.DEFAULT_MODEL, address="formal")
    assert "Sie" in formal and " du " not in f" {formal} "
    informal = await server.stream_translation(
        FakeWS(), 1, text, "en", "de", server.DEFAULT_MODEL, address="informal")
    assert "du" in informal.lower() and "Sie" not in informal
    plural_es = await server.stream_translation(
        FakeWS(), 1, text, "en", "es", server.DEFAULT_MODEL, address="plural")
    assert "puedes" not in plural_es.lower()


async def test_real_ollama_mexican_flavor_produces_mexican_spanish():
    if not ollama_up():
        pytest.skip("Ollama not running")
    text = await server.stream_translation(
        FakeWS(), 1, "That's really cool! Call me on my cell phone later.",
        "en", "es", server.DEFAULT_MODEL, flavor="mexico")
    low = text.lower()
    assert "celular" in low or "chido" in low or "padre" in low
    assert "móvil" not in low and "vosotros" not in low


async def test_real_ollama_barcelona_flavor_produces_peninsular_spanish():
    if not ollama_up():
        pytest.skip("Ollama not running")
    text = await server.stream_translation(
        FakeWS(), 1,
        "Okay, can you all send me the photos on my phone later?",
        "en", "es", server.DEFAULT_MODEL, flavor="barcelona",
        address="plural")
    low = text.lower()
    assert "podéis" in low or "enviadme" in low or "móvil" in low or "vale" in low
    assert "ustedes" not in low and "celular" not in low
