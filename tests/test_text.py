"""Hallucination filtering, direction resolution, and prompt construction."""
import pytest

import server
from server import (HALLUCINATION_RE, clean_transcript, is_echo, lang_hint,
                    normalize_text, resolve_direction, translation_messages)


@pytest.mark.parametrize("junk", [
    "Thank you.",
    "THANK YOU",
    "Thanks for watching.",
    "you.",
    "Untertitelung des ZDF, 2020",
    "Untertitel der Amara.org-Gemeinschaft",
    "Vielen Dank.",
    "Amara.org community",
    "Subtítulos realizados por la comunidad de Amara.org",
    "¡Gracias por ver el vídeo!",
    "Gracias.",
    "Suscríbete al canal",
    "...",
])
def test_hallucinations_filtered(junk):
    assert HALLUCINATION_RE.match(junk)


@pytest.mark.parametrize("real", [
    "Guten Tag, wie geht's?",
    "Thank you for helping me move yesterday.",
    "Vielen Dank für Ihre Hilfe gestern.",
    "Can you help me?",
    "Ich habe die Untertitel nicht gesehen.",
    "You are welcome to join us.",
    "Gracias por tu ayuda ayer.",
    "¿Puedes ayudarme con esto?",
])
def test_real_sentences_kept(real):
    assert not HALLUCINATION_RE.match(real)


def test_forced_directions_ignore_detection():
    assert resolve_direction("de-en", "en") == ("de", "en")
    assert resolve_direction("en-de", "de") == ("en", "de")
    assert resolve_direction("es-en", "de") == ("es", "en")
    assert resolve_direction("en-es", "de") == ("en", "es")
    assert resolve_direction("es-de", "en") == ("es", "de")


def test_auto_pairs_follow_detected_language():
    assert resolve_direction("auto-de-en", "de") == ("de", "en")
    assert resolve_direction("auto-de-en", "en") == ("en", "de")
    assert resolve_direction("auto-es-en", "es") == ("es", "en")
    assert resolve_direction("auto-es-en", "en") == ("en", "es")
    assert resolve_direction("auto-es-de", "de") == ("de", "es")
    assert resolve_direction("auto-es-de", "es") == ("es", "de")


def test_auto_defaults_to_first_pair_language_for_other_detections():
    assert resolve_direction("auto-de-en", "ja") == ("de", "en")
    assert resolve_direction("auto-es-en", "ja") == ("es", "en")


def test_legacy_auto_alias_still_works():
    assert resolve_direction("auto", "de") == ("de", "en")
    assert resolve_direction("auto", "en") == ("en", "de")


def test_garbage_modes_fall_back_safely():
    assert resolve_direction("auto-xx-yy", "de") == ("de", "en")
    assert resolve_direction("klingon-en", "de") == ("de", "en")
    assert resolve_direction("de-de", "de") == ("de", "en")
    assert resolve_direction("", "de") == ("de", "en")


def test_lang_hint_pinned_only_for_forced_modes():
    assert lang_hint("de-en") == "de"
    assert lang_hint("es-en") == "es"
    assert lang_hint("en-es") == "en"
    assert lang_hint("auto-es-en") is None
    assert lang_hint("auto") is None
    assert lang_hint("garbage") is None


def test_translation_prompt_structure():
    msgs = translation_messages("Guten Morgen", "de", "en")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "German" in msgs[0]["content"] and "English" in msgs[0]["content"]
    assert "ONLY" in msgs[0]["content"]  # bare-translation instruction
    assert msgs[1]["content"] == "Guten Morgen"


def test_history_becomes_chat_turns_with_static_system():
    h = [{"source": "de", "target": "en", "text": "A", "translation": "B"}]
    m1 = translation_messages("X", "de", "en", h)
    m2 = translation_messages("Y", "de", "en", [])
    assert m1[0] == m2[0]  # static prefix → Ollama's prompt cache can reuse it
    assert [m["role"] for m in m1] == ["system", "user", "assistant", "user"]
    assert m1[1]["content"] == "A" and m1[2]["content"] == "B"


def test_reverse_direction_history_is_flipped():
    h = [{"source": "en", "target": "de", "text": "Hello", "translation": "Hallo"}]
    m = translation_messages("Wie bitte?", "de", "en", h)
    assert m[1] == {"role": "user", "content": "Hallo"}
    assert m[2] == {"role": "assistant", "content": "Hello"}


def test_other_pair_history_is_excluded():
    h = [{"source": "es", "target": "en", "text": "Hola", "translation": "Hi"}]
    m = translation_messages("Guten Tag", "de", "en", h)
    assert [x["role"] for x in m] == ["system", "user"]


# ------------------------------------------------------- confidence filtering


def test_clean_transcript_drops_low_confidence_segments():
    result = {"text": "ignored", "segments": [
        {"text": " Guten Tag.", "no_speech_prob": 0.1, "avg_logprob": -0.3},
        {"text": " la la la", "no_speech_prob": 0.9, "avg_logprob": -1.5},
    ]}
    assert clean_transcript(result) == "Guten Tag."


def test_clean_transcript_keeps_borderline_segments():
    # Only the combination of high no-speech AND low logprob drops a segment.
    result = {"text": "x", "segments": [
        {"text": " Musik läuft.", "no_speech_prob": 0.9, "avg_logprob": -0.4},
        {"text": " Leise gesprochen.", "no_speech_prob": 0.2, "avg_logprob": -1.4},
    ]}
    assert clean_transcript(result) == "Musik läuft. Leise gesprochen."


def test_clean_transcript_without_segments_falls_back_to_text():
    assert clean_transcript({"text": " Hallo "}) == "Hallo"
    assert clean_transcript({"segments": [
        {"text": " x", "no_speech_prob": 0.99, "avg_logprob": -2.0}]}) == ""


# ---------------------------------------------------------------- echo dedupe


def test_near_identical_transcripts_are_echoes():
    a = normalize_text("Ich komme aus Berlin und arbeite als Lehrer.")
    b = normalize_text("ich komme aus Berlin und arbeite als Lehrer")
    c = normalize_text("Ähm, ich komme aus Berlin und arbeite als Lehrer.")
    assert is_echo(a, b)      # punctuation/case differences
    assert is_echo(a, c)      # near-duplicate with a filler word


def test_different_sentences_are_not_echoes():
    a = normalize_text("Ich komme aus Berlin und arbeite als Lehrer.")
    b = normalize_text("Wie lange wohnst du schon in München?")
    assert not is_echo(a, b)
    assert not is_echo("", a)


# ------------------------------------------------------------------ glossary


def test_glossary_loads_skips_comments_and_reloads(tmp_path, monkeypatch):
    path = tmp_path / "glossary.txt"
    path.write_text("Data Society = Data Society\n# comment\n\nRevealKit\n")
    monkeypatch.setattr(server, "GLOSSARY_PATH", path)
    server._glossary_cache.update(mtime=None, lines=[])
    assert server.load_glossary() == ["Data Society = Data Society", "RevealKit"]
    assert server.glossary_whisper_terms() == "Data Society, RevealKit"
    msgs = translation_messages("Hallo", "de", "en")
    assert "Glossary" in msgs[0]["content"]
    assert "RevealKit" in msgs[0]["content"]


def test_missing_glossary_is_harmless(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "GLOSSARY_PATH", tmp_path / "absent.txt")
    server._glossary_cache.update(mtime=1.0, lines=["stale"])
    assert server.load_glossary() == []
    assert "Glossary" not in translation_messages("Hi", "en", "de")[0]["content"]
