"""The words on the card decide its direction when they contradict Whisper.

Whisper names the language from the sound and then writes the words, and the
two can disagree: German spoken with an English accent, or with one English
word in it, is called "en" and still transcribed as German. The card then
asks for German into German and shows the same sentence twice. Over a real
71-minute conversation (2026-10-04) 45 of the 95 cards labeled English were
German text, and 81 of 399 cards repeated the heard text as the translation.
The sentences below are invented in the shapes that conversation produced
(the recording itself is private and stays out of this public repo).
"""
import json

import pytest

import server as srv
from conftest import collect_until, speak, trace_records


def card(client, stub_transcribe, text, language, mode="auto"):
    stub_transcribe.result = {"text": text, "language": language}
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": mode}))
        speak(ws)
        msgs = collect_until(ws)
    return next(m for m in msgs if m["type"] == "final")


@pytest.mark.parametrize("text", [
    "Ja, ich wollte gerade sagen, das ist ziemlich teuer.",
    "Und das war wirklich schön.",
    "Nice weather. Ja, es ist heute sehr warm.",          # one English word in it
    "Alles, you know, alles war nass, auch mein Koffer.",
])
def test_german_text_heard_as_english_is_translated_as_german(
        client, stub_transcribe, trace_file, text):
    final = card(client, stub_transcribe, text, "en")
    assert (final["source"], final["target"]) == ("de", "en")
    assert trace_records(trace_file)[0]["relabel"] == "en>de"


def test_english_text_heard_as_german_is_translated_as_english(
        client, stub_transcribe):
    final = card(client, stub_transcribe,
                 "Oh, I think we already missed the bus? Oh, thanks.", "de")
    assert (final["source"], final["target"]) == ("en", "de")


def test_agreement_changes_nothing_and_leaves_no_mark(client, stub_transcribe,
                                                      trace_file):
    final = card(client, stub_transcribe, "Und das war wirklich schön.", "de")
    assert final["source"] == "de"
    assert "relabel" not in trace_records(trace_file)[0]


@pytest.mark.parametrize("text", ["So.", "Ja.", "Ugh!"])
def test_a_one_word_card_keeps_whispers_call(client, stub_transcribe, text):
    """Too little text to overrule the audio: the detector scores these below
    DIRECTION_TEXT_CONF, and the chip's flip button stays the way out."""
    assert srv.detect_language_scored(text, ("de", "en"))[1] \
        < srv.DIRECTION_TEXT_CONF
    assert card(client, stub_transcribe, text, "en")["source"] == "en"


def test_a_forced_direction_is_never_relabeled(client, stub_transcribe,
                                                trace_file):
    """The user pinned the direction; the text does not get a vote."""
    final = card(client, stub_transcribe, "Und das war wirklich schön.", "en",
                 mode="en-de")
    assert (final["source"], final["target"]) == ("en", "de")
    assert "relabel" not in trace_records(trace_file)[0]


def test_the_spanish_pair_uses_its_own_languages(client, stub_transcribe):
    final = card(client, stub_transcribe,
                 "¿Dónde está la estación de tren, por favor?", "en",
                 mode="auto-es-en")
    assert (final["source"], final["target"]) == ("es", "en")
