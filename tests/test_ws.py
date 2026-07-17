"""End-to-end WebSocket flow with Whisper and Ollama doubled out."""
import json

from conftest import SPEECH_CHUNK, collect_until, speak


def test_full_flow_auto_german(client, stub_transcribe):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto",
                                 "model": "gemma3:12b"}))
        speak(ws)
        msgs = collect_until(ws)

    types = [m["type"] for m in msgs]
    assert "segment_start" in types
    final = next(m for m in msgs if m["type"] == "final")
    assert final["text"] == "Wie geht es dir?"
    assert (final["source"], final["target"]) == ("de", "en")
    assert final["targets"] == ["en"]
    assert final["speaker"] == "you"
    translated = "".join(m["text"] for m in msgs if m["type"] == "translation_delta")
    assert translated == "How are you?"
    done = msgs[-1]
    assert done["type"] == "translation_done"
    assert done["transcribe_ms"] >= 0 and done["translate_ms"] >= 0


def test_forced_direction_overrides_detection(client, stub_transcribe):
    stub_transcribe.result = {"text": "Das ist gut", "language": "de"}
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "en-de",
                                 "model": "gemma3:12b"}))
        speak(ws)
        msgs = collect_until(ws)

    final = next(m for m in msgs if m["type"] == "final")
    assert (final["source"], final["target"]) == ("en", "de")
    # Forcing the direction also pins Whisper's language hint.
    assert stub_transcribe.calls[-1]["language"] == "en"


def test_auto_spanish_pair(client, stub_transcribe):
    stub_transcribe.result = {"text": "¿Dónde está la estación?", "language": "es"}
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-es-en",
                                 "model": "gemma3:12b"}))
        speak(ws)
        msgs = collect_until(ws)

    final = next(m for m in msgs if m["type"] == "final")
    assert (final["source"], final["target"]) == ("es", "en")
    # Auto mode must leave Whisper's language detection free.
    assert stub_transcribe.calls[-1]["language"] is None


def test_hallucination_is_discarded(client, stub_transcribe):
    stub_transcribe.result = {"text": "Thank you.", "language": "en"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)

    assert msgs[-1]["type"] == "discard"
    assert not any(m["type"] == "final" for m in msgs)


def test_empty_transcript_is_discarded(client, stub_transcribe):
    stub_transcribe.result = {"text": "   ", "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    assert msgs[-1]["type"] == "discard"


def test_repetition_loop_transcript_is_discarded(client, stub_transcribe):
    stub_transcribe.result = {"text": "Möhnnydin" + "nin" * 200,
                              "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    assert msgs[-1]["type"] == "discard"
    assert not any(m["type"] == "final" for m in msgs)


def test_malformed_control_message_survives(client, stub_transcribe):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("{{{ not json")          # must not kill the connection
        ws.send_text(json.dumps({"type": "config", "mode": "de-en"}))
        speak(ws)
        msgs = collect_until(ws)
    assert msgs[-1]["type"] == "translation_done"


def test_odd_length_audio_survives(client, stub_transcribe):
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(SPEECH_CHUNK[:-1])      # stray byte, not valid int16
        speak(ws)
        msgs = collect_until(ws)
    assert msgs[-1]["type"] == "translation_done"


def test_two_utterances_get_distinct_ids(client, stub_transcribe):
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs1 = collect_until(ws)
        speak(ws)
        msgs2 = collect_until(ws)
    id1 = next(m for m in msgs1 if m["type"] == "segment_start")["id"]
    id2 = next(m for m in msgs2 if m["type"] == "segment_start")["id"]
    assert id1 != id2


def test_partials_emitted_during_long_speech(client, stub_transcribe):
    with client.websocket_connect("/ws") as ws:
        # Long utterance: >1 s of speech so a partial is allowed.
        speak(ws, speech_chunks=20)
        msgs = collect_until(ws)
    assert any(m["type"] == "partial" for m in msgs)


def test_tagged_frames_are_labeled_them(client, stub_transcribe):
    with client.websocket_connect("/ws") as ws:
        for _ in range(3):
            ws.send_bytes(b"\x01" + bytes(4096))
        for _ in range(10):
            ws.send_bytes(b"\x01" + SPEECH_CHUNK)
        for _ in range(8):
            ws.send_bytes(b"\x01" + bytes(4096))
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    assert final["speaker"] == "them"


def test_multi_target_translates_to_both(client, stub_transcribe):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "de-en+es"}))
        speak(ws)
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    assert final["targets"] == ["en", "es"]
    by_target = {}
    for m in msgs:
        if m["type"] == "translation_delta":
            by_target.setdefault(m["target"], []).append(m["text"])
    assert set(by_target) == {"en", "es"}
    assert "".join(by_target["en"]) == "How are you?"
    assert "".join(by_target["es"]) == "How are you?"  # fake ollama is monolingual
    assert msgs[-1]["type"] == "translation_done"


def test_cut_fragment_is_merged_into_next_utterance(client, stub_transcribe):
    stub_transcribe.result = {"text": "Ich glaube, dass wir das Projekt",
                              "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs1 = collect_until(ws)
        stub_transcribe.result = {"text": "nächste Woche abschließen werden.",
                                  "language": "de"}
        speak(ws)
        msgs2 = collect_until(ws)

    final1 = next(m for m in msgs1 if m["type"] == "final")
    final2 = next(m for m in msgs2 if m["type"] == "final")
    assert final2["replaces"] == final1["id"]
    assert final2["text"] == ("Ich glaube, dass wir das Projekt "
                              "nächste Woche abschließen werden.")


def test_complete_sentence_is_not_merged(client, stub_transcribe):
    with client.websocket_connect("/ws") as ws:
        speak(ws)                      # "Wie geht es dir?" ends a sentence
        collect_until(ws)
        speak(ws)
        msgs2 = collect_until(ws)
    final2 = next(m for m in msgs2 if m["type"] == "final")
    assert "replaces" not in final2
    assert final2["text"] == "Wie geht es dir?"


def test_merge_never_grows_past_cap(client, stub_transcribe):
    long_unfinished = ("und dann haben wir über das neue Projekt gesprochen "
                       "und die Pläne für nächstes Jahr besprochen ") * 4
    stub_transcribe.result = {"text": long_unfinished.strip(), "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        collect_until(ws)
        stub_transcribe.result = {"text": "genau darum geht es.", "language": "de"}
        speak(ws)
        msgs2 = collect_until(ws)
    final2 = next(m for m in msgs2 if m["type"] == "final")
    assert "replaces" not in final2  # >300-char fragments are left as-is


def test_pause_setting_shortens_finalization(client, stub_transcribe):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "pause_ms": 320}))
        # Only ~380 ms of trailing silence: finalizes at 320 ms, but would
        # still be buffering with the 700 ms default.
        speak(ws, silence_chunks=3)
        msgs = collect_until(ws)
    assert any(m["type"] == "final" for m in msgs)


def test_history_context_flows_into_later_requests(client, stub_transcribe,
                                                   fake_ollama):
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        collect_until(ws)
        stub_transcribe.result = {"text": "Und morgen?", "language": "de"}
        speak(ws)
        collect_until(ws)
    messages = fake_ollama["chat"]["messages"]
    users = [m["content"] for m in messages if m["role"] == "user"]
    assistants = [m["content"] for m in messages if m["role"] == "assistant"]
    assert "Wie geht es dir?" in users       # first utterance, as a chat turn
    assert "How are you?" in assistants      # ... with its translation
    assert messages[-1] == {"role": "user", "content": "Und morgen?"}


def test_speculation_avoids_double_transcription(client, stub_transcribe,
                                                 monkeypatch):
    import server as srv
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)  # no partial noise
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    assert any(m["type"] == "final" for m in msgs)
    # One speculative transcription during the pause, reused at finalization.
    assert len(stub_transcribe.calls) == 1


def test_stale_speculation_not_used_when_speech_resumes(client, stub_transcribe,
                                                        monkeypatch):
    import server as srv
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    with client.websocket_connect("/ws") as ws:
        for _ in range(3):
            ws.send_bytes(bytes(4096))
        for _ in range(10):
            ws.send_bytes(SPEECH_CHUNK)
        for _ in range(4):          # ~510 ms: past EARLY, before finalize
            ws.send_bytes(bytes(4096))
        for _ in range(10):         # speaker resumes
            ws.send_bytes(SPEECH_CHUNK)
        for _ in range(8):
            ws.send_bytes(bytes(4096))
        msgs = collect_until(ws)
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(finals) == 1
    # Two speculations ran (one went stale), and the final transcription
    # covered the WHOLE resumed utterance, not just the first fragment.
    assert len(stub_transcribe.calls) == 2
    assert max(c["seconds"] for c in stub_transcribe.calls) > 2.5


def test_out_of_pair_detection_is_retried_pinned(client, stub_transcribe,
                                                 monkeypatch):
    import server as srv
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    stub_transcribe.queue = [
        {"text": "Goedemorgen allemaal", "language": "nl"},   # misdetection
        {"text": "Guten Morgen allerseits", "language": "de"},
    ]
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    assert final["source"] == "de"
    assert final["text"] == "Guten Morgen allerseits"
    assert stub_transcribe.calls[-1]["language"] == "de"  # retry was pinned


def test_glossary_terms_reach_whisper(client, stub_transcribe, monkeypatch,
                                      tmp_path):
    import server as srv
    path = tmp_path / "glossary.txt"
    path.write_text("RevealKit\nData Society\n")
    monkeypatch.setattr(srv, "GLOSSARY_PATH", path)
    srv._glossary_cache.update(mtime=None, lines=[])
    try:
        with client.websocket_connect("/ws") as ws:
            speak(ws)
            collect_until(ws)
    finally:
        srv._glossary_cache.update(mtime=None, lines=[])
    assert any(c["prompt"] and "RevealKit" in c["prompt"]
               for c in stub_transcribe.calls)


def test_forced_mode_reuses_whisper_prompt(client, stub_transcribe):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "de-en"}))
        speak(ws)
        collect_until(ws)
        speak(ws)
        collect_until(ws)
    last = stub_transcribe.calls[-1]
    assert last["language"] == "de"
    assert last["prompt"] == "Wie geht es dir?"


def speak_tagged(ws, tag):
    for _ in range(3):
        ws.send_bytes(bytes([tag]) + bytes(4096))
    for _ in range(10):
        ws.send_bytes(bytes([tag]) + SPEECH_CHUNK)
    for _ in range(8):
        ws.send_bytes(bytes([tag]) + bytes(4096))


def test_cross_channel_echo_is_discarded(client, stub_transcribe):
    stub_transcribe.result = {"text": "Ich komme aus Berlin und arbeite "
                                      "als Lehrer.", "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)                    # channel 0 ("you") hears it first
        msgs1 = collect_until(ws)
        speak_tagged(ws, 1)          # mic bleed: same speech on channel 1
        msgs2 = collect_until(ws)
    assert any(m["type"] == "final" for m in msgs1)
    assert msgs2[-1]["type"] == "discard"
    assert not any(m["type"] == "final" for m in msgs2)


def test_short_repeats_are_not_treated_as_echo(client, stub_transcribe):
    stub_transcribe.result = {"text": "Genau!", "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs1 = collect_until(ws)
        speak_tagged(ws, 1)          # the other person really says it too
        msgs2 = collect_until(ws)
    assert any(m["type"] == "final" for m in msgs1)
    assert any(m["type"] == "final" for m in msgs2)


def test_pure_silence_produces_nothing(client, stub_transcribe):
    import server as srv
    with client.websocket_connect("/ws") as ws:
        for _ in range(60):
            ws.send_bytes(bytes(4096))
        ws.send_text(json.dumps({"type": "config", "mode": "auto"}))
    srv.whisper_executor.submit(lambda: None).result()
    assert stub_transcribe.calls == []


def test_correction_message_patches_history(client, stub_transcribe,
                                            fake_ollama):
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
        uid = next(m["id"] for m in msgs if m["type"] == "final")
        ws.send_text(json.dumps({"type": "correction", "id": uid,
                                 "target": "en", "corrected": "How's it going?"}))
        stub_transcribe.result = {"text": "Und morgen?", "language": "de"}
        speak(ws)
        collect_until(ws)
    assistants = [m["content"] for m in fake_ollama["chat"]["messages"]
                  if m["role"] == "assistant"]
    # The edited translation, not the model's original, is the context now.
    assert "How's it going?" in assistants
    assert "How are you?" not in assistants


def test_malformed_correction_message_survives(client, stub_transcribe):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "correction", "id": 99,
                                 "target": "en", "corrected": None}))
        ws.send_text(json.dumps({"type": "correction"}))
        speak(ws)
        msgs = collect_until(ws)
    assert any(m["type"] == "final" for m in msgs)  # still fully functional
