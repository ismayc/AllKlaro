"""Slow tests against the real Whisper model and a live Ollama server.

Run with:  RUN_INTEGRATION=1 uv run pytest -m integration -v
"""
import json
import os
import re
import subprocess
import time
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


def test_real_partial_asr_transcribes_german_and_spanish(tmp_path,
                                                         real_partial_asr):
    """The fast partial pass against the real Parakeet model. Partials are
    transient, so the bar is 'recognisably the right words in the right
    language', not Whisper-grade — but it has to actually handle German
    umlauts and Spanish accents, which is why both are checked."""
    if server.load_parakeet() is None:
        pytest.skip("parakeet-mlx / model unavailable")
    for text, voice, must in ((GERMAN_TEXT, "Anna", ("projekt", "woche")),
                              (SPANISH_TEXT, SPANISH_VOICE,
                               ("proyecto", "semana"))):
        pcm = tts_wav(tmp_path, text, voice)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        got = server.transcribe_partial(audio)
        assert got is not None, "fast path returned None with a model loaded"
        for word in must:
            assert word in got.lower(), f"{word!r} missing from {got!r}"


def test_real_partial_asr_is_much_faster_than_whisper(tmp_path,
                                                      real_partial_asr):
    """The entire justification for the second model. Measured 884 ms vs
    71 ms on a 6 s German window (2026-08-06); asserting only 3x leaves
    room for a slower machine or a cold-ish graph."""
    if server.load_parakeet() is None:
        pytest.skip("parakeet-mlx / model unavailable")
    pcm = tts_wav(tmp_path, GERMAN_TEXT, "Anna")
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    window = audio[:16000 * 6]

    server.transcribe_partial(window)          # warm the MLX graph
    server.transcribe(window, language=None)
    t = time.perf_counter(); server.transcribe_partial(window)
    fast = time.perf_counter() - t
    t = time.perf_counter(); server.transcribe(window, language=None)
    slow = time.perf_counter() - t
    assert fast * 3 < slow, f"partial pass {fast:.3f}s vs whisper {slow:.3f}s"


def test_reused_speculation_transcribes_the_same_words(tmp_path):
    """The soundness condition for reusing a speculation on a soft_max split.

    That split emits `speech[:split_at]`, while the speculation decoded the
    same audio plus the rest of its silence run — (EARLY - MICRO) frames,
    128 ms. Reuse is only honest if that tail cannot change the words, which
    is a claim about a real model, not about arithmetic: the unit tests can
    only check the lengths line up. Punctuation and case are normalised away
    because a trailing pause legitimately moves a full stop, and a card is
    not wrong for ending in "." instead of "!".
    """
    pcm = tts_wav(tmp_path, GERMAN_TEXT, "Anna")
    chunk = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    pad = ((server.EARLY_SILENCE_FRAMES - server.MICRO_PAUSE_FRAMES)
           * server.FRAME_SAMPLES)
    speculated = np.concatenate([chunk, np.zeros(pad, dtype=np.float32)])

    def words(audio):
        text = server.clean_transcript(server.transcribe(audio, language="de"))
        return re.findall(r"\w+", text.lower())

    assert words(speculated) == words(chunk)


NEGATED = ("didn't", "did not", "not ", "n't ", "no ")


async def test_real_model_uncrosses_a_mis_heard_negation():
    """The failure this whole dialect path exists for, against the real model.

    Hessian "net verstanne" (nicht verstanden) is transcribed as "nett
    verstarne", which gemma then renders as "I understood that nicely" — the
    exact opposite of what was said. The gloss that fixes it was already in
    dialects.txt and unreachable: it is an ambiguous entry, and those were
    only offered alongside an unambiguous marker that speech never supplies.
    """
    text = "Ich habe das nett verstarne."
    plain = await server.translate_once(text, "de", "en", server.DEFAULT_MODEL)
    hinted = await server.translate_once(text, "de", "en", server.DEFAULT_MODEL,
                                         heard_flavor="hessian")
    assert plain and hinted
    assert not any(n in plain.lower() for n in NEGATED), \
        f"expected the uncorrected inversion, got {plain!r}"
    assert any(n in hinted.lower() for n in NEGATED), \
        f"negation not recovered: {hinted!r}"


async def test_real_model_does_not_invent_a_negation():
    """The opposite failure. "nett" really can just mean nice, so the hint is
    hedged — an unhedged gloss would invert this one instead of fixing it."""
    hinted = await server.translate_once("Das war nett von dir.", "de", "en",
                                         server.DEFAULT_MODEL, heard_flavor="hessian")
    assert hinted and not any(n in hinted.lower() for n in NEGATED), \
        f"hint inverted a sentence that was already correct: {hinted!r}"


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


async def test_real_ollama_folds_a_gist_from_a_bilingual_exchange():
    """The stub cannot tell a summary from any other string. This checks the
    thing that actually matters: a real model, given lines in two languages,
    answers in English about what was said rather than translating them."""
    if not ollama_up():
        pytest.skip("Ollama not running")
    gist = await server.fold_gist("", [
        "[DE] Wir müssen den Garten bewässern, der Rasen ist völlig trocken.",
        "[EN] Forget the lawn, I only care about the palm and the bushes.",
        "[DE] Gut, dann sammeln wir das Wasser über Nacht in der Tonne.",
    ], server.DEFAULT_MODEL)
    assert gist
    low = gist.lower()
    assert any(w in low for w in ("water", "garden", "lawn", "palm")), gist
    # An English gist, not a translation of the German lines.
    assert "bewässern" not in low and "völlig" not in low


async def test_real_ollama_gist_carries_earlier_context_forward():
    """The fold is only worth doing if the previous gist survives it: this is
    what keeps a 54-minute call from restarting every minute."""
    if not ollama_up():
        pytest.skip("Ollama not running")
    gist = await server.fold_gist(
        "• They agreed to fly to Hawaii in March.",
        ["[EN] So the overnight flight is the cheaper one.",
         "[DE] Ja, dann nehmen wir den Nachtflug."],
        server.DEFAULT_MODEL)
    assert gist
    low = gist.lower()
    # Without the "keep concrete details" instruction in GIST_PROMPT this
    # retained the destination only 4 times in 8; with it, 8 in 8. Left as a
    # hard assertion deliberately — if it starts flaking again, the fold has
    # regressed to generalising specifics away, which is the failure mode.
    assert "hawaii" in low, f"earlier context dropped: {gist}"
    assert "night" in low or "overnight" in low, gist
    assert "[de]" not in low and "[en]" not in low, f"raw tags leaked: {gist}"


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


def test_real_improve_tap_returns_the_main_models_translation(tmp_path):
    """The tap against real models, which is the only place its value lives.

    The offline comparison found the main model fixing errors a learner would
    care about on ~13% of utterances; under live pacing the automatic refine
    reaches a minority of cards because it is gated and timed out. This asks
    for one deliberately, with neither handicap, and checks a real answer
    comes back — not that it is better, which needs a human reading.
    """
    if not ollama_up():
        pytest.skip("Ollama not running")
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "config", "mode": "de-en",
                                     "model": server.DEFAULT_MODEL}))
            ws.send_text(json.dumps({
                "type": "improve", "id": 1, "source": "de", "target": "en",
                "text": "Das verdunstet aber eigentlich auch eine Menge."}))
            msgs = collect_until(ws, stop_types=("improved", "error"))
    reply = msgs[-1]
    assert reply["type"] == "improved", reply
    assert "error" not in reply, reply
    assert "evaporat" in reply["text"].lower(), reply["text"]


def test_real_voice_change_separates_two_speakers(tmp_path):
    """Two macOS voices, which is the only ground truth available without
    someone labelling the real recording by hand.

    Synthetic voices are more separable than two people sharing one
    microphone, so this is an upper bound — it can show the detector works at
    all, never that the shipped threshold is right for a room.
    """
    import voiceprint as vp

    a1 = np.frombuffer(tts_wav(tmp_path, GERMAN_TEXT, "Anna"), dtype=np.int16)
    a2 = np.frombuffer(tts_wav(tmp_path, SPANISH_TEXT, "Anna"), dtype=np.int16)
    other = np.frombuffer(
        tts_wav(tmp_path, GERMAN_TEXT, "Reed (German (Germany))"),
        dtype=np.int16)

    same = vp.voice_distance(vp.voice_signature(a1), vp.voice_signature(a2))
    diff = vp.voice_distance(vp.voice_signature(a1), vp.voice_signature(other))
    assert same < server.VOICE_CHANGE_DIST < diff, (
        f"same-speaker {same:.3f}, different {diff:.3f}, "
        f"threshold {server.VOICE_CHANGE_DIST}")
