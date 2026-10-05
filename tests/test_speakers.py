"""Paragraphs that follow the voice ("Speakers", off by default).

The real model is never loaded here: `voices` stands in for it, handing each
long chunk the next vector from a list, so a test states who is speaking and
checks where the cards break. Sentences are invented.
"""
import json
from contextlib import contextmanager

import numpy as np
import pytest

import server as srv
import speakers
from conftest import collect_until, speak, trace_records

ANNA = np.array([1.0, 0.0])
BERT = np.array([0.0, 1.0])
LONG = 30      # speech chunks of 128 ms: 3.8 s, long enough to judge
SHORT = 6      # 0.8 s: a short reply


class Voices:
    def __init__(self):
        self.queue = []
        self.calls = 0

    def __call__(self, audio):
        self.calls += 1
        vec = self.queue.pop(0)
        if isinstance(vec, Exception):
            raise vec
        return vec


@pytest.fixture
def voices(monkeypatch):
    stub = Voices()
    monkeypatch.setattr(speakers, "_model", object())   # "loaded"
    monkeypatch.setattr(speakers, "embed", stub)
    return stub


def say(ws, stub_transcribe, text, chunks=LONG, language="de"):
    stub_transcribe.result = {"text": text, "language": language}
    speak(ws, speech_chunks=chunks)
    msgs = collect_until(ws)
    return next(m for m in msgs if m["type"] == "final")


@contextmanager
def session(client, on=True):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en",
                                 "speakers": on}))
        yield ws


def test_the_same_voice_joins_a_finished_sentence(client, stub_transcribe,
                                                  voices, trace_file):
    """Two complete sentences by one person are one paragraph, which the
    text rules alone would have split."""
    voices.queue = [ANNA, ANNA]
    with session(client) as ws:
        first = say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        second = say(ws, stub_transcribe, "Danach haben wir Kaffee getrunken.")
    assert second["replaces"] == first["id"]
    assert second["text"] == ("Wir waren gestern im Garten. "
                              "Danach haben wir Kaffee getrunken.")
    rec = trace_records(trace_file)[-1]
    assert rec["by_voice"] is True and rec["voice_cos"] == 1.0


def test_a_new_voice_starts_a_new_card_mid_sentence(client, stub_transcribe,
                                                    voices, trace_file):
    """An unfinished sentence would merge under the text rules; a different
    voice ends the paragraph anyway."""
    voices.queue = [ANNA, BERT]
    with session(client) as ws:
        say(ws, stub_transcribe, "Wir waren gestern im Garten und")
        second = say(ws, stub_transcribe, "haben dort lange gesessen.")
    assert "replaces" not in second
    assert trace_records(trace_file)[-1]["voice_cos"] == 0.0


def test_a_short_reply_stays_in_the_paragraph(client, stub_transcribe, voices):
    """Too short to judge, so it is not asked about and does not break the
    card or become the voice the next long chunk is compared with."""
    voices.queue = [ANNA, ANNA]
    with session(client) as ws:
        first = say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        reply = say(ws, stub_transcribe, "Das klingt aber wirklich schön.",
                    chunks=SHORT)
        third = say(ws, stub_transcribe, "Danach haben wir Kaffee getrunken.")
    assert reply["replaces"] == first["id"]
    assert third["replaces"] == reply["id"]
    assert voices.calls == 2


def test_a_new_voice_is_not_absorbed_as_an_interjection(client,
                                                        stub_transcribe,
                                                        voices):
    voices.queue = [ANNA, BERT]
    with session(client) as ws:
        say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        second = say(ws, stub_transcribe, "Ja, genau.")
    assert "replaces" not in second


def test_another_language_still_gets_its_own_card(client, stub_transcribe,
                                                  voices):
    """A card has one direction, whoever is speaking."""
    voices.queue = [ANNA, ANNA]
    with session(client) as ws:
        say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        second = say(ws, stub_transcribe,
                     "I think we already missed the bus this morning.",
                     language="en")
    assert "replaces" not in second and second["source"] == "en"


def test_a_long_silence_ends_the_paragraph(client, stub_transcribe, voices,
                                           monkeypatch):
    monkeypatch.setattr(srv, "SPEAKER_GAP_SEC", -1e9)
    voices.queue = [ANNA, ANNA]
    with session(client) as ws:
        say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        second = say(ws, stub_transcribe, "Danach haben wir Kaffee getrunken.")
    assert "replaces" not in second


def test_a_model_failure_falls_back_to_joining(client, stub_transcribe,
                                               voices, trace_file):
    """One bad chunk must not take the session down: with no verdict the
    chunk stays in the paragraph, and the next comparison has no reference."""
    voices.queue = [ANNA, RuntimeError("boom"), BERT]
    with session(client) as ws:
        first = say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        second = say(ws, stub_transcribe, "Danach haben wir Kaffee getrunken.")
        third = say(ws, stub_transcribe, "Und dann sind wir heimgefahren.")
    assert second["replaces"] == first["id"]
    assert third["replaces"] == second["id"]
    assert "voice_cos" not in trace_records(trace_file)[-1]


def test_off_by_default_the_model_is_never_asked(client, stub_transcribe,
                                                 voices, trace_file):
    with session(client, on=False) as ws:
        say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        second = say(ws, stub_transcribe, "Danach haben wir Kaffee getrunken.")
    assert "replaces" not in second          # the text rules: both finished
    assert voices.calls == 0
    assert "by_voice" not in trace_records(trace_file)[-1]


def test_turning_it_on_without_the_model_says_so(client, monkeypatch):
    monkeypatch.setattr(speakers, "_model", None)
    monkeypatch.setattr(speakers, "load", lambda: None)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "speakers": True}))
        msg = collect_until(ws, stop_types=("error",))[-1]
    assert "uv sync --extra speakers" in msg["message"]


def test_turning_it_on_loads_the_model_quietly_when_present(
        client, stub_transcribe, monkeypatch):
    """Until the load finishes the session keeps the text rules."""
    loaded = []
    monkeypatch.setattr(speakers, "_model", None)
    monkeypatch.setattr(speakers, "load", lambda: loaded.append(1) or object())
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en",
                                 "speakers": True}))
        final = say(ws, stub_transcribe, "Wir waren gestern im Garten.")
    assert loaded == [1] and final["type"] == "final"


# ------------------------------------------------------------- the wrapper

class FakeModel:
    def encode_batch(self, wav):
        import torch
        assert wav.shape[0] == 1 and wav.dtype == torch.float32
        assert float(wav.abs().max()) <= 1.0
        return torch.tensor([[[3.0, 4.0]]])


def test_embed_returns_a_unit_vector_for_a_long_chunk(monkeypatch):
    monkeypatch.setattr(speakers, "_model", FakeModel())
    audio = (np.ones(int(speakers.MIN_SEC * 16000)) * 1000).astype(np.int16)
    vec = speakers.embed(audio)
    assert np.allclose(vec, [0.6, 0.8])


def test_embed_declines_a_short_chunk_without_loading(monkeypatch):
    monkeypatch.setattr(speakers, "load",
                        lambda: pytest.fail("loaded for a short chunk"))
    assert speakers.embed(np.zeros(16000, dtype=np.int16)) is None


def test_embed_is_none_without_a_model(monkeypatch):
    monkeypatch.setattr(speakers, "load", lambda: None)
    assert speakers.embed(np.zeros(4 * 16000, dtype=np.int16)) is None


def test_similarity_needs_both_vectors():
    assert speakers.similarity(ANNA, BERT) == 0.0
    assert speakers.similarity(ANNA, ANNA) == 1.0
    assert speakers.similarity(None, ANNA) is None
    assert speakers.similarity(ANNA, None) is None


def test_load_reports_a_missing_install_once(monkeypatch, caplog):
    import builtins
    real_import = builtins.__import__

    def no_speechbrain(name, *a, **k):
        if name.startswith("speechbrain"):
            raise ImportError("No module named 'speechbrain'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(speakers, "_model", None)
    monkeypatch.setattr(speakers, "_unavailable", False)
    monkeypatch.setattr(builtins, "__import__", no_speechbrain)
    assert speakers.load() is None
    assert speakers.load() is None and not speakers.ready()
    assert caplog.text.count("uv sync --extra speakers") == 1


def test_the_page_offers_the_setting_and_sends_it():
    from pathlib import Path
    root = Path(srv.__file__).parent / "static"
    assert 'id="speakersChk"' in (root / "index.html").read_text()
    assert "speakers: speakersChk.checked" in (root / "app.js").read_text()


@pytest.mark.integration
@pytest.mark.skipif(not __import__("os").environ.get("RUN_INTEGRATION"),
                    reason="set RUN_INTEGRATION=1 to run integration tests")
def test_the_real_model_tells_two_synthetic_voices_apart(tmp_path):
    """Loads the real model (needs `uv sync --extra speakers`). Synthetic
    voices are far easier than real ones, so this only proves the wiring:
    audio format, vector shape, and that similarity runs the right way."""
    import subprocess
    import wave

    def voice(name, text):
        out = tmp_path / f"{name}.wav"
        subprocess.run(["say", "-v", name, "--data-format=LEI16@16000",
                        "--file-format=WAVE", "-o", str(out), text],
                       check=True)
        with wave.open(str(out)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    a1 = speakers.embed(voice("Anna", "Wir waren gestern lange im Garten "
                                      "und haben Kaffee getrunken."))
    a2 = speakers.embed(voice("Anna", "Danach sind wir mit dem Fahrrad in "
                                      "die Stadt gefahren."))
    b = speakers.embed(voice("Daniel", "Yesterday we stayed in the garden "
                                       "for a long time and had coffee."))
    assert a1.shape == (192,) and abs(float(a1 @ a1) - 1.0) < 1e-5
    assert speakers.similarity(a1, a2) > speakers.SAME_COS
    assert speakers.similarity(a1, b) < speakers.SAME_COS
