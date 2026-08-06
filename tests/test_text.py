"""Hallucination filtering, direction resolution, and prompt construction."""
import pytest

import server
from server import (HALLUCINATION_RE, clean_transcript, collapse_repeats,
                    has_phrase_loop, is_degenerate, is_echo, lang_hint,
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


def test_grammar_notes_only_for_targets_that_have_them():
    into_de = translation_messages("The margarita is good.", "en", "de")
    assert "feminine" in into_de[0]["content"]      # German gender note present
    for source, target in [("de", "en"), ("en", "es")]:
        msgs = translation_messages("x", source, target)
        assert "feminine" not in msgs[0]["content"]  # no note defined yet


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


# Real repetition-loop artifacts captured from a live session (music in a
# YouTube video drove Whisper's decoder into degenerate loops).
NIN_LOOP = "Möhnnydin" + "nin" * 200
DASH_LOOP = "Niiiim" + "-n" * 110
L_LOOP_MIXED = ("M-A-Z-Z-Z So, hallo, tschüss sagen ist nicht so oft am "
                "Start, auch wenn man so ein " + "L-" * 100)
PHRASE_LOOP = "Ich bin ein Berliner. " * 30


def test_repetition_loops_are_dropped():
    for junk in (NIN_LOOP, DASH_LOOP, PHRASE_LOOP):
        assert clean_transcript({"text": junk}) == "", junk[:40]


def test_mixed_segment_keeps_speech_drops_loop_tail():
    text = clean_transcript({"text": L_LOOP_MIXED})
    assert "hallo, tschüss sagen" in text
    assert "L-L-L-L" not in text


def test_whisper_compression_ratio_signal_is_honored():
    result = {"segments": [
        {"text": " Guten Tag.", "no_speech_prob": 0.1, "avg_logprob": -0.3,
         "compression_ratio": 1.4},
        {"text": " irgendwas", "no_speech_prob": 0.1, "avg_logprob": -0.5,
         "compression_ratio": 5.2},
    ]}
    assert clean_transcript(result) == "Guten Tag."


def test_normal_speech_is_never_flagged_degenerate():
    for real in (
        "Das ist wahrscheinlich ein soziales Ding. Ja, na voll sozial. Die "
        "suchen Kumpels, beziehungsweise die Kumpels suchen Mädchen. Ja.",
        "Machen mich glücklich, wenn so Leute nett zueinander sind.",
        "Und die Kumpels tanzen wie die Blöden mit ihrem Arsch und drehen "
        "in drei Nase mal.",
    ):
        assert clean_transcript({"text": real}) == real


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


# ------------------------------------------------------ typed-text detection
# Accuracy over the full phrase corpus lives in test_detect.py; these are the
# direction-resolution cases that the rest of this module depends on.


def test_detect_language_german_english():
    from server import detect_language
    assert detect_language("Wie geht es dir und der Familie?") == "de"
    assert detect_language("Where is the train station, please?") == "en"
    assert detect_language("Das ist doch schön hier.", ("de", "en")) == "de"


def test_detect_language_spanish_pair():
    from server import detect_language
    assert detect_language("¿Dónde está la estación?", ("es", "en")) == "es"
    assert detect_language("I would like to buy a ticket.", ("es", "en")) == "en"


def test_detect_language_ignores_candidate_order():
    """A bare noun used to be decided purely by which language was listed
    first, because it scored zero in both. That tie-break is what made
    "Happy birthday!" German; the answer must not turn on argument order."""
    from server import detect_language
    assert (detect_language("Xylophon", ("de", "en"))
            == detect_language("Xylophon", ("en", "de")))

# ------------------------------------------------------ German output flavor


def test_flavor_note_added_for_german_target():
    for flavor, marker in (("berlin", "ick (ich)"), ("hessian", "isch (ich)"),
                           ("worms", "Grumbeere")):
        system = translation_messages("Hello", "en", "de",
                                      flavor=flavor)[0]["content"]
        assert marker in system
        assert "never change the meaning" in system


def test_flavor_off_or_wrong_target_stays_standard():
    assert "Berlinerisch" not in translation_messages(
        "Hello", "en", "de")[0]["content"]
    # Flavor styles German output only — translating TO English ignores it.
    assert "Berlinerisch" not in translation_messages(
        "Hallo", "de", "en", flavor="berlin")[0]["content"]


# ------------------------------------------------------------- address forms


def test_address_note_pins_the_you_form():
    for target, form, marker in (
            ("de", "informal", '"du"'), ("de", "formal", '"Sie"'),
            ("de", "plural", '"ihr"'), ("es", "informal", '"tú"'),
            ("es", "formal", '"usted"'), ("es", "plural", '"ustedes"')):
        system = translation_messages("Can you help me?", "en", target,
                                      address=form)[0]["content"]
        assert marker in system, (target, form)


def test_address_auto_or_english_target_adds_nothing():
    assert '"du"' not in translation_messages(
        "Can you help me?", "en", "de")[0]["content"]
    # Translating TO English never needs a you-form note.
    assert "Address the listener" not in translation_messages(
        "Kannst du mir helfen?", "de", "en", address="formal")[0]["content"]


def test_address_and_flavor_notes_compose():
    system = translation_messages("Can you help me?", "en", "de",
                                  flavor="berlin",
                                  address="informal")[0]["content"]
    assert "Berlinerisch" in system and '"du"' in system


def test_spanish_flavor_notes_and_barcelona_plural_override():
    mx = translation_messages("That's cool!", "en", "es",
                              flavor="mexico")[0]["content"]
    assert "Mexican" in mx and "chido" in mx and "vosotros" in mx
    bcn = translation_messages("That's cool!", "en", "es",
                               flavor="barcelona")[0]["content"]
    assert "Barcelona" in bcn and "vosotros" in bcn
    # Barcelona + plural address: vosotros beats the ustedes default.
    both = translation_messages("Can you help me?", "en", "es",
                                flavor="barcelona",
                                address="plural")[0]["content"]
    assert "podéis" in both and "¿Me podéis ayudar?" in both
    # Flavors never leak across targets.
    assert "chido" not in translation_messages(
        "Hi", "en", "de", flavor="mexico")[0]["content"]


# The exact output of forcing German onto English speech in the real
# recording — "we had some sort of problem where they came and fixed
# something" decoded as German. Kept verbatim because a synthesised loop
# would not have the same near-miss statistics: this one scores 1.33
# compression against is_degenerate's 4.0 and 1.57 against Whisper's 2.4.
REAL_PHRASE_LOOP = ("Und so haben wir so ein Problem, wo sie sich und die "
                    "Füße starete, die sich so starete, die Füße starete, "
                    "die Füße starete.")


def test_the_real_phrase_loop_is_dropped():
    assert has_phrase_loop(REAL_PHRASE_LOOP)
    assert clean_transcript({"text": REAL_PHRASE_LOOP}) == ""


def test_the_loop_slips_every_older_guard():
    """Documents why a new check was needed rather than a tuned threshold:
    all three existing guards score this text as ordinary speech."""
    assert not is_degenerate(REAL_PHRASE_LOOP)
    assert collapse_repeats(REAL_PHRASE_LOOP) == REAL_PHRASE_LOOP
    assert clean_transcript({"segments": [
        {"text": REAL_PHRASE_LOOP, "compression_ratio": 1.57,
         "no_speech_prob": 0.1, "avg_logprob": -0.3}]}) == ""


@pytest.mark.parametrize("real", [
    # Verbatim from the 51-utterance dump of the real recording.
    "Ja, ja, ja.",
    "Ja. Ja. Ja, genau.",
    "It was .46 yesterday. Yeah. Yeah.",
    "We had some sort of problem where they came and fixed something and "
    "then they also adjusted the refiller.",
    "Guten Tag, wie geht's? Wie geht es Ihnen denn heute so?",
])
def test_real_speech_is_not_mistaken_for_a_loop(real):
    """Emphasis repeats a word, never a three-word phrase. Zero of the 51
    real utterances flagged; these are the closest calls among them."""
    assert not has_phrase_loop(real)
    assert clean_transcript({"text": real}) == real.strip()
