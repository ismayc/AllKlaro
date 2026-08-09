"""End-to-end WebSocket flow with Whisper and Ollama doubled out."""
import json

import server
from conftest import (SILENCE_CHUNK, SPEECH_CHUNK, collect_until, speak,
                      speak_flowing, trace_records)


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


def test_trailing_off_fragment_is_merged(client, stub_transcribe):
    """Whisper ends a cut-off utterance with an ellipsis, and that used to
    read as terminal punctuation — so the two halves of this real sentence
    from 25:00 became separate cards, each translated blind to the other
    ("They were called Gottbergs, where one was as…" / "as a kitchen boy.").
    23% of real utterances end this way, 86% of them cut mid-speech."""
    stub_transcribe.result = {"text": "Von Gottbergs hießen die, wo man da als...",
                              "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs1 = collect_until(ws)
        stub_transcribe.result = {"text": "als Küchenbammser gearbeitet hat.",
                                  "language": "de"}
        speak(ws)
        msgs2 = collect_until(ws)

    final1 = next(m for m in msgs1 if m["type"] == "final")
    final2 = next(m for m in msgs2 if m["type"] == "final")
    assert final2["replaces"] == final1["id"]
    assert final2["text"] == ("Von Gottbergs hießen die, wo man da als... "
                              "als Küchenbammser gearbeitet hat.")


def test_the_single_character_ellipsis_counts_too(client, stub_transcribe):
    """Whisper emits both "..." and "…"; only spelling one of them would fix
    half the cases and look like it worked."""
    stub_transcribe.result = {"text": "Sie hatte da überhaupt gar nicht die "
                                      "Energie oder den auch so…",
                              "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs1 = collect_until(ws)
        stub_transcribe.result = {
            "text": "sie konnte sich damit gar nicht auseinandersetzen.",
            "language": "de"}
        speak(ws)
        msgs2 = collect_until(ws)
    final1 = next(m for m in msgs1 if m["type"] == "final")
    final2 = next(m for m in msgs2 if m["type"] == "final")
    assert final2["replaces"] == final1["id"]


def test_a_real_full_stop_still_ends_a_sentence():
    """The point of the change is narrow. If it also swallowed ordinary full
    stops, every consecutive sentence would be glued into one card."""
    assert server.looks_finished("Das war klar.")
    assert server.looks_finished("Wie geht es dir?")
    assert server.looks_finished('Er sagte "ja."')
    assert not server.looks_finished("Von Gottbergs hießen die, wo man da als...")
    assert not server.looks_finished("oder den auch so…")
    assert not server.looks_finished("und dann haben wir")


def test_a_full_stop_does_not_end_a_sentence_if_the_next_chunk_continues_it(
        client, stub_transcribe):
    """The case punctuation-only merging structurally cannot see.

    Whisper ends the fragment with a full stop — nothing about it says
    unfinished — and puts the verb in the next chunk. This exact pair is from
    the real recording, and separately the cards read "they PUT gravel in that
    area of the garden" (a verb the model invented, because `ausgestreut` had
    not been said) and "spread out so that the water drains better" (a
    participle with no subject). German capitalises every noun and every
    sentence start, so the lowercase opening is the tell.
    """
    stub_transcribe.result = {
        "text": "Und darunter haben sie Kies in dem Bereich des Gartens.",
        "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs1 = collect_until(ws)
        stub_transcribe.result = {
            "text": "ausgestreut, damit das Wasser besser abläuft.",
            "language": "de"}
        speak(ws)
        msgs2 = collect_until(ws)

    final1 = next(m for m in msgs1 if m["type"] == "final")
    final2 = next(m for m in msgs2 if m["type"] == "final")
    assert final2["replaces"] == final1["id"]
    assert final2["text"] == ("Und darunter haben sie Kies in dem Bereich des "
                              "Gartens. ausgestreut, damit das Wasser besser "
                              "abläuft.")


def test_two_finished_sentences_are_still_two_cards(client, stub_transcribe):
    """The guard that keeps the casing rule narrow. If a capitalised opening
    also merged, every consecutive sentence would glue into one running card
    and the feed would stop being readable."""
    stub_transcribe.result = {"text": "Das war klar.", "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        collect_until(ws)
        stub_transcribe.result = {"text": "Wir sehen uns morgen.",
                                  "language": "de"}
        speak(ws)
        msgs2 = collect_until(ws)
    final2 = next(m for m in msgs2 if m["type"] == "final")
    assert "replaces" not in final2


def test_continues_previous_reads_the_casing():
    assert server.continues_previous("ausgestreut, damit das Wasser abläuft.")
    assert server.continues_previous("gibt es da auch keine Abdeckung für.")
    assert server.continues_previous("...so eine L-Form")   # punctuation first
    # A German noun is capitalised mid-sentence, so an opening capital is not
    # proof of a sentence start — but it is all the evidence there is, and
    # treating it as "not a continuation" is the conservative direction.
    assert not server.continues_previous("Wasservolumen sofort aufwärmt.")
    assert not server.continues_previous("Der Pool ist ja so ein L.")
    assert not server.continues_previous("")
    assert not server.continues_previous("2 Grad ab, würde ich sagen")


def test_flowed_on_reads_the_split_reason():
    """`soft_max` is the cap firing on speech that never stopped, so the next
    chunk is the same person carrying on. `pause` and `hard_max` are not: a
    pause is a real gap anyone could speak into, and a hard cut means the
    speaker ran past MAX_UTTERANCE_SEC without any micro-pause at all."""
    assert server.flowed_on("soft_max")
    assert not server.flowed_on("pause")
    assert not server.flowed_on("hard_max")
    assert not server.flowed_on(None)       # first utterance of a session


def test_speech_cut_by_the_cap_merges_with_what_follows(
        client, stub_transcribe):
    """The case both text rules structurally miss.

    Whisper ends the cap-cut fragment with a full stop and starts the next one
    with a capital, so `looks_finished` says finished and `continues_previous`
    says new sentence. Both are reading the text, and the text cannot know the
    speaker never paused. The split reason knows.

    This is the passage from the real recording that the hour-long comparison
    flagged: the rain-gutter description arrives as two cards, and the second
    one ("Und darunter...") has no idea what "darunter" is under.
    """
    stub_transcribe.result = {
        "text": "Regenrinnen gibt es hier nicht, dann tropft das da runter.",
        "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak_flowing(ws)                  # cut by the cap, still talking
        msgs1 = collect_until(ws)
        stub_transcribe.result = {
            "text": "Und darunter haben sie Kies ausgestreut.", "language": "de"}
        speak(ws)
        msgs2 = collect_until(ws)

    final1 = next(m for m in msgs1 if m["type"] == "final")
    final2 = next(m for m in msgs2 if m["type"] == "final")
    assert final2["replaces"] == final1["id"]
    assert final2["text"].endswith("Und darunter haben sie Kies ausgestreut.")
    assert final2["text"].startswith("Regenrinnen gibt es hier nicht")


def test_a_real_pause_between_two_finished_sentences_still_splits(
        client, stub_transcribe):
    """The guard that keeps the new rule from becoming a clock.

    Merging on elapsed time alone reaches the batch service's segment count but
    splices a question onto its answer, because it cannot tell a turn change
    from a breath. Requiring `soft_max` is what keeps that from happening: two
    finished sentences separated by a real pause stay two cards.
    """
    stub_transcribe.result = {"text": "Das war klar.", "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)                          # ends on silence -> `pause` split
        collect_until(ws)
        stub_transcribe.result = {"text": "Wir sehen uns morgen.",
                                  "language": "de"}
        speak(ws)
        msgs2 = collect_until(ws)
    final2 = next(m for m in msgs2 if m["type"] == "final")
    assert "replaces" not in final2


def test_the_uncased_transcript_is_a_known_cost_not_a_surprise():
    """Whisper occasionally emits a chunk with no casing at all, and then a
    real sentence start looks like a continuation and merges one card too
    many. Measured at 2.3% of German cards over the real hour, and pinned here
    so it is a known price rather than a mystery card."""
    assert server.continues_previous("morgen für knapp zwei wochen nach hawaii")


def test_a_lowercase_opening_after_a_long_gap_is_still_a_new_card(
        client, stub_transcribe, monkeypatch):
    """Casing says "this continues something"; the gap says whether the thing
    it continues was a moment ago. Without the gap check a lowercase chunk
    would merge into whatever happened to precede it, minutes earlier.

    The gap is wall clock (`loop.time()`) minus the utterance's own duration,
    so sending more silence does not widen it, and under a test harness that
    feeds faster than realtime it is *negative*. Both facts are worth knowing
    before writing any other timing test against this path — a threshold of
    0.0 here would not close the window at all.
    """
    monkeypatch.setattr(server, "MERGE_GAP_SEC", -1e9)
    stub_transcribe.result = {"text": "Das war klar.", "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        collect_until(ws)
        stub_transcribe.result = {"text": "ausgestreut, damit es abläuft.",
                                  "language": "de"}
        speak(ws)
        msgs2 = collect_until(ws)
    final2 = next(m for m in msgs2 if m["type"] == "final")
    assert "replaces" not in final2


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


# The measured failure: forcing "German → English" pins Whisper to German, so
# an English utterance in the same conversation decodes as a German repetition
# loop. has_phrase_loop now rejects that text, which keeps it off the screen but
# turns the utterance into a silent discard — the speech is lost either way.
LOOPED = ("Und so haben wir so ein Problem, wo sie sich und die Füße starete, "
          "die sich so starete, die Füße starete, die Füße starete.")


def test_forced_direction_falls_back_to_auto_when_the_decode_loops(
        client, stub_transcribe, monkeypatch):
    import server as srv
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    stub_transcribe.queue = [
        {"text": LOOPED, "language": "de"},                 # pinned, degenerate
        {"text": "We had some sort of problem.", "language": "en"},
    ]
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "de-en"}))
        speak(ws)
        msgs = collect_until(ws)

    final = next(m for m in msgs if m["type"] == "final")
    assert final["text"] == "We had some sort of problem."
    # Rescued *and* pointed the right way: without this the English would be
    # handed to the translator labelled German, the same error one stage later.
    assert final["source"] == "en"
    assert final["target"] == "de"
    # The chip has to appear, because the direction came from the audio rather
    # than from the mode the user picked.
    assert final["auto"] is True
    assert stub_transcribe.calls[0]["language"] == "de"    # first pinned
    assert stub_transcribe.calls[1]["language"] is None    # retry set it free


def test_forced_direction_does_not_redo_a_short_rejected_decode(
        client, stub_transcribe, monkeypatch):
    """The redo costs a full decode on the one Whisper thread, so it must fire
    only when a *substantial* decode was thrown away. This text is rejected by
    the loop filter exactly like the long one, but at 28 characters it is the
    noise-shaped garbage a second decode would not improve — so the length
    threshold, not the rejection, is what has to decide."""
    import server as srv
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    short_loop = "ab cd ef, ab cd ef, ab cd ef"
    assert len(short_loop) < srv.FORCED_REDO_MIN_CHARS
    assert srv.clean_transcript({"text": short_loop}) == ""   # rejected
    stub_transcribe.queue = [{"text": short_loop, "language": "de"}]
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "de-en"}))
        speak(ws)
        msgs = collect_until(ws)
    assert any(m["type"] == "discard" for m in msgs)
    assert len(stub_transcribe.calls) == 1        # no second decode


def test_auto_mode_never_uses_the_forced_fallback(client, stub_transcribe,
                                                  monkeypatch):
    """An auto mode never pinned anything, so a loop there is a real loop —
    redoing it would just spend a second decode to get the same answer."""
    import server as srv
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    stub_transcribe.queue = [{"text": LOOPED, "language": "de"}]
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en"}))
        speak(ws)
        msgs = collect_until(ws)
    assert any(m["type"] == "discard" for m in msgs)
    assert len(stub_transcribe.calls) == 1


def test_multi_target_forced_mode_keeps_its_pinned_decode(client,
                                                          stub_transcribe,
                                                          monkeypatch):
    """"de-en+es" has no single language to fall back to — resolve_targets
    would have to pick one of the targets as the new source. Left pinned."""
    import server as srv
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    stub_transcribe.queue = [{"text": LOOPED, "language": "de"}]
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "de-en+es"}))
        speak(ws)
        collect_until(ws)
    assert len(stub_transcribe.calls) == 1


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


# ------------------------------------------------------------ draft + refine


def draft_config(ws, draft="qwen2.5:7b-instruct", model="gemma3:12b"):
    ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en",
                             "model": model, "draft_model": draft}))


def test_draft_streams_then_main_model_refines(client, stub_transcribe,
                                               fake_ollama):
    with client.websocket_connect("/ws") as ws:
        draft_config(ws)
        speak(ws)
        msgs = collect_until(ws, stop_types=("translation_revised", "error"))
    done = next(m for m in msgs if m["type"] == "translation_done")
    assert done["refining"] is True
    revised = next(m for m in msgs if m["type"] == "translation_revised")
    assert revised["texts"] == {"en": "Refined translation."}
    streamed = [b for b in fake_ollama["all"] if b.get("stream", True)]
    quiet = [b for b in fake_ollama["all"] if b.get("stream", True) is False]
    assert streamed[-1]["model"] == "qwen2.5:7b-instruct"  # draft, visible
    assert quiet[-1]["model"] == "gemma3:12b"              # refinement, quiet


def test_refined_text_becomes_context_for_next_utterance(client,
                                                         stub_transcribe,
                                                         fake_ollama):
    with client.websocket_connect("/ws") as ws:
        draft_config(ws)
        speak(ws)
        collect_until(ws, stop_types=("translation_revised", "error"))
        stub_transcribe.result = {"text": "Und morgen?", "language": "de"}
        speak(ws)
        collect_until(ws)
    assistants = [m["content"] for m in fake_ollama["chat"]["messages"]
                  if m["role"] == "assistant"]
    assert "Refined translation." in assistants
    assert "How are you?" not in assistants  # draft was superseded


def test_no_draft_model_means_single_pass(client, stub_transcribe,
                                          fake_ollama):
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws)
    done = next(m for m in msgs if m["type"] == "translation_done")
    assert done["refining"] is False
    assert all(b.get("stream", True) for b in fake_ollama["all"])


def test_draft_same_as_main_model_means_single_pass(client, stub_transcribe,
                                                    fake_ollama):
    with client.websocket_connect("/ws") as ws:
        draft_config(ws, draft="gemma3:12b", model="gemma3:12b")
        speak(ws)
        msgs = collect_until(ws)
    assert next(m for m in msgs
                if m["type"] == "translation_done")["refining"] is False


def test_failed_refinement_keeps_draft_quietly(client, stub_transcribe,
                                               fake_ollama):
    with client.websocket_connect("/ws") as ws:
        draft_config(ws, draft="gemma3:12b", model="missing:1b")
        speak(ws)
        msgs = collect_until(ws, stop_types=("translation_revised", "error"))
    revised = next(m for m in msgs if m["type"] == "translation_revised")
    assert revised["texts"] == {}            # UI just clears the spinner
    assert not any(m["type"] == "error" for m in msgs)


# --------------------------------------------------------------- typed input


def test_typed_text_translates_without_audio(client, stub_transcribe,
                                             fake_ollama):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "text",
                                 "text": "Wie geht es dir und der Familie?"}))
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    assert final["source"] == "de" and final["target"] == "en"
    assert final["speaker"] == "you"
    assert any(m["type"] == "translation_done" for m in msgs)
    assert stub_transcribe.calls == []       # no Whisper involved


def test_typed_english_reverses_direction(client, stub_transcribe,
                                          fake_ollama):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "text",
                                 "text": "Where is the train station, please?"}))
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    assert final["source"] == "en" and final["target"] == "de"


def test_typed_text_shares_history_with_speech(client, stub_transcribe,
                                               fake_ollama):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "text", "text": "Wie geht es dir?"}))
        collect_until(ws)
        speak(ws)                            # spoken follow-up sees typed turn
        collect_until(ws)
    users = [m["content"] for m in fake_ollama["chat"]["messages"]
             if m["role"] == "user"]
    assert users.count("Wie geht es dir?") == 2  # history turn + new sentence


# ------------------------------------------- overriding the detected language


def test_typed_source_pin_skips_detection(client, stub_transcribe,
                                          fake_ollama):
    """The type bar's language pin. "Happy birthday!" is English, but pinning
    German must be obeyed without argument — the point of the pin is that the
    user knows something the detector doesn't."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "text", "text": "Happy birthday!",
                                 "source": "de"}))
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    assert final["source"] == "de" and final["target"] == "en"
    assert "conf" not in final           # nothing was detected
    # ...so the chip must not claim it detected anything either.
    assert final["chosen"] is True


def test_pin_outside_the_active_pair_is_ignored(client, stub_transcribe,
                                                fake_ollama):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en"}))
        ws.send_text(json.dumps({"type": "text", "text": "Happy birthday!",
                                 "source": "es"}))
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    assert final["source"] == "en"       # fell back to detection
    assert final["conf"] > 0


def test_detected_typed_card_is_marked_flippable(client, stub_transcribe,
                                                 fake_ollama):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "text", "text": "Happy birthday!"}))
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    assert final["source"] == "en"       # the bug this all started from
    assert final["auto"] is True         # UI shows the ⇄ chip
    assert final["chosen"] is False      # ...labelled "Detected", not "Set to"
    assert 0 < final["conf"] <= 1


def test_forced_direction_card_is_not_flippable(client, stub_transcribe,
                                                fake_ollama):
    """Nothing to flip when the user already pinned the whole direction."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "de-en"}))
        ws.send_text(json.dumps({"type": "text", "text": "Guten Morgen"}))
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    assert final["auto"] is False and "conf" not in final


def test_spoken_card_is_flippable_too(client, stub_transcribe, fake_ollama):
    """Whisper can pick the wrong language from the audio as easily as the
    text detector can from the text."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en"}))
        speak(ws)
        msgs = collect_until(ws)
    final = next(m for m in msgs if m["type"] == "final")
    assert final["auto"] is True


def test_retranslate_replaces_the_card_with_the_other_direction(
        client, stub_transcribe, fake_ollama):
    """Tapping a card's chip: same text, opposite direction, and the old card
    goes away instead of leaving both readings on screen."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "text", "text": "Happy birthday!"}))
        first = next(m for m in collect_until(ws) if m["type"] == "final")
        ws.send_text(json.dumps({"type": "retranslate", "id": first["id"],
                                 "text": "Happy birthday!", "source": "de"}))
        msgs = collect_until(ws)
    redone = next(m for m in msgs if m["type"] == "final")
    assert redone["source"] == "de" and redone["target"] == "en"
    assert redone["replaces"] == first["id"]
    assert redone["id"] != first["id"]
    assert any(m["type"] == "translation_done" for m in msgs)


def test_retranslate_drops_the_wrong_reading_from_context(
        client, stub_transcribe, fake_ollama):
    """The mistranslation must not keep steering later turns from history."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "text",
                                 "text": "Wie geht es dir und der Familie?"}))
        first = next(m for m in collect_until(ws) if m["type"] == "final")
        ws.send_text(json.dumps({"type": "retranslate", "id": first["id"],
                                 "text": "Wie geht es dir und der Familie?",
                                 "source": "en"}))
        collect_until(ws)
        ws.send_text(json.dumps({"type": "text", "text": "Und sonst?"}))
        collect_until(ws)
    # A context turn renders as user=source/assistant=translation, and the
    # roles swap when the next turn runs the other way — so count the text
    # itself, not the role it landed in.
    context = [m["content"] for m in fake_ollama["chat"]["messages"]
               if m["role"] != "system"]
    # Once, as the redone turn. Twice would mean the replaced card's reading
    # is still in the context, competing with the corrected one.
    assert context.count("Wie geht es dir und der Familie?") == 1


def test_retranslate_garbage_is_ignored(client, stub_transcribe, fake_ollama):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "retranslate", "id": 1}))
        ws.send_text(json.dumps({"type": "retranslate", "id": 1, "text": "  "}))
        ws.send_text(json.dumps({"type": "text", "text": "Guten Morgen"}))
        msgs = collect_until(ws)
    assert any(m["type"] == "final" for m in msgs)   # still fully functional


def test_typed_garbage_is_ignored(client, stub_transcribe):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "text", "text": "   "}))
        ws.send_text(json.dumps({"type": "text", "text": 42}))
        ws.send_text(json.dumps({"type": "text"}))
        speak(ws)
        msgs = collect_until(ws)
    assert any(m["type"] == "final" for m in msgs)  # still fully functional


def test_single_pass_agreement_fix_sends_revision(client, stub_transcribe,
                                                  fake_ollama, monkeypatch):
    import server as srv
    # Deterministic guard: flag the streamed text, accept the retry.
    monkeypatch.setattr(srv, "agreement_issues",
                        lambda text, target: (['"How" is wrong.']
                                              if "How" in text else []))
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until(ws, stop_types=("translation_revised", "error"))
    revised = next(m for m in msgs if m["type"] == "translation_revised")
    assert revised["texts"] == {"en": "Refined translation."}


def test_german_flavor_reaches_the_translation_prompt(client, stub_transcribe,
                                                      fake_ollama):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "en-de",
                                 "model": "gemma3:12b",
                                 "de_flavor": "berlin"}))
        ws.send_text(json.dumps({"type": "text", "text": "See you tomorrow!"}))
        collect_until(ws)
    system = fake_ollama["chat"]["messages"][0]["content"]
    assert "Berlinerisch" in system and "ick (ich)" in system


def test_unknown_flavor_falls_back_to_standard_german(client, stub_transcribe,
                                                      fake_ollama):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "en-de",
                                 "model": "gemma3:12b",
                                 "de_flavor": "bavarian"}))
        ws.send_text(json.dumps({"type": "text", "text": "See you tomorrow!"}))
        collect_until(ws)
    system = fake_ollama["chat"]["messages"][0]["content"]
    assert "dialect" not in system.lower()


def test_address_config_reaches_the_translation_prompt(client, stub_transcribe,
                                                       fake_ollama):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "en-de",
                                 "model": "gemma3:12b",
                                 "address": "formal"}))
        ws.send_text(json.dumps({"type": "text", "text": "See you tomorrow!"}))
        collect_until(ws)
    system = fake_ollama["chat"]["messages"][0]["content"]
    assert '"Sie"' in system
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "en-de",
                                 "model": "gemma3:12b",
                                 "address": "royal-we"}))
        ws.send_text(json.dumps({"type": "text", "text": "See you tomorrow!"}))
        collect_until(ws)
    assert "Address the listener" not in fake_ollama["chat"]["messages"][0]["content"]


def test_spanish_flavor_config_reaches_the_prompt(client, stub_transcribe,
                                                  fake_ollama):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "en-es",
                                 "model": "gemma3:12b",
                                 "es_flavor": "mexico"}))
        ws.send_text(json.dumps({"type": "text", "text": "That's cool!"}))
        collect_until(ws)
    assert "Mexican" in fake_ollama["chat"]["messages"][0]["content"]


# ------------------------------------------------------- pipeline instrumentation


def test_trace_records_where_an_utterance_spent_its_time(client, stub_transcribe,
                                                         fake_ollama, trace_file):
    """One JSON line per utterance, carrying what the on-card latency omits."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en",
                                 "model": "gemma3:12b"}))
        speak(ws)
        collect_until(ws)

    records = trace_records(trace_file)
    assert len(records) == 1
    rec = records[0]
    assert rec["outcome"] == "final"
    assert rec["split"] == "pause"          # a normal, well-paced utterance
    assert rec["speaker"] == "you"
    assert rec["chunk_sec"] > 0
    # The felt delay includes the chunk's own duration; the card's number
    # does not. first_word_lag must therefore always exceed lag.
    assert rec["first_word_lag_ms"] >= rec["lag_ms"]
    assert rec["first_word_lag_ms"] >= rec["chunk_sec"] * 1000
    for field in ("wait_ms", "transcribe_ms", "translate_ms", "whisper_queue",
                  "in_flight", "partials_skipped", "spec", "uid", "t"):
        assert field in rec, f"trace lost {field}"


def test_trace_records_discards_too(client, stub_transcribe, fake_ollama,
                                    trace_file):
    """A dropped utterance still consumed the Whisper thread — it must show up,
    or the trace understates the load during fast speech."""
    stub_transcribe.result = {"text": "Untertitel der Amara.org-Community",
                              "language": "de"}
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en"}))
        speak(ws)
        collect_until(ws)

    records = trace_records(trace_file)
    assert [r["outcome"] for r in records] == ["discard_empty"]
    assert "transcribe_ms" in records[0]


def test_tracing_can_be_switched_off(client, stub_transcribe, fake_ollama,
                                     trace_file, monkeypatch):
    monkeypatch.setattr(server, "TRACE_PATH", "off")
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en"}))
        speak(ws)
        collect_until(ws)
    assert not trace_file.exists()


def test_stats_are_only_pushed_when_the_overlay_asks(client, stub_transcribe,
                                                     fake_ollama, monkeypatch):
    """Off by default: an idle phone on cellular should not carry telemetry."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en"}))
        speak(ws)
        msgs = collect_until(ws)
    assert not [m for m in msgs if m["type"] == "stats"]

    # A whole test utterance takes milliseconds, so the real half-second
    # cadence would only ever emit the opening (idle) sample.
    monkeypatch.setattr(server, "STATS_INTERVAL_SEC", 0)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en",
                                 "stats": True}))
        speak(ws, speech_chunks=40, silence_chunks=10)
        msgs = collect_until(ws)
    stats = [m for m in msgs if m["type"] == "stats"]
    assert stats, "overlay asked for stats and got none"
    for key in ("in_flight", "whisper_queue", "speech_sec", "partials_skipped"):
        assert key in stats[0]
    # While speech is still accumulating, the overlay must show it — that is
    # the number explaining an empty screen mid-sentence.
    assert max(s["speech_sec"] for s in stats) > 0


def test_fast_talker_is_traced_as_soft_max(client, stub_transcribe, fake_ollama,
                                           trace_file):
    """The case that started this: continuous speech with only a micro-pause.

    The chunk gets cut by the soft-max rule rather than by the speaker, and
    its first word is older on screen than the card's own latency suggests.
    """
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en",
                                 "model": "gemma3:12b"}))
        for _ in range(3):
            ws.send_bytes(SILENCE_CHUNK)
        for _ in range(40):            # ~5.1 s of unbroken speech
            ws.send_bytes(SPEECH_CHUNK)
        for _ in range(2):             # ~256 ms dip: a micro-pause, not a pause
            ws.send_bytes(SILENCE_CHUNK)
        for _ in range(30):            # past SOFT_MAX_SEC -> forced split
            ws.send_bytes(SPEECH_CHUNK)
        for _ in range(30):
            ws.send_bytes(SILENCE_CHUNK)
        collect_until(ws, limit=400)

    records = trace_records(trace_file)
    assert records, "a forced split must still be traced"
    forced = records[0]
    assert forced["split"] == "soft_max"
    assert forced["chunk_sec"] > 4      # a long chunk nobody asked for
    # The whole point of the metric: the card's latency omits the seconds the
    # audio spent accumulating, so the felt delay is strictly larger.
    assert (forced["first_word_lag_ms"]
            > forced["transcribe_ms"] + forced["translate_ms"])
