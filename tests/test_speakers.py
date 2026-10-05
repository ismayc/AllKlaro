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

    def __call__(self, audio, min_sec=None):
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


# ------------------------------------------------------------------- names

CLARA = np.array([0.0, 0.0, 1.0])


def save_voices(**voices):
    speakers.VOICES_PATH.write_text(json.dumps(
        {name: list(map(float, vec)) for name, vec in voices.items()}))


@contextmanager
def named_session(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "mode": "auto-de-en",
                                 "speakers": True, "names": True}))
        yield ws


def test_a_card_carries_the_name_of_its_saved_voice(client, stub_transcribe,
                                                    voices, trace_file):
    save_voices(Anna=[1, 0], Bert=[0, 1])
    voices.queue = [ANNA, BERT]
    with named_session(client) as ws:
        first = say(ws, stub_transcribe, "Wir waren gestern im Garten und")
        second = say(ws, stub_transcribe, "haben dort lange gesessen.")
    assert first["voice"] == "Anna" and second["voice"] == "Bert"
    assert "replaces" not in second
    assert trace_records(trace_file)[-1]["voice"] == "Bert"


def test_two_named_chunks_follow_the_names_not_the_similarity(
        client, stub_transcribe, voices):
    """Both chunks are Anna by name even though they are less alike than
    SAME_COS: the card must not break under one name."""
    save_voices(Anna=[1, 0, 0], Bert=[0, 1, 0])
    a1 = np.array([0.8, 0.0, 0.6])
    a2 = np.array([0.8, 0.0, -0.6])
    assert float(a1 @ a2) < speakers.SAME_COS
    voices.queue = [a1, a2]
    with named_session(client) as ws:
        first = say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        second = say(ws, stub_transcribe, "Danach haben wir Kaffee getrunken.")
    assert second["replaces"] == first["id"] and second["voice"] == "Anna"


def test_a_short_reply_keeps_the_cards_name(client, stub_transcribe, voices):
    save_voices(Anna=[1, 0], Bert=[0, 1])
    voices.queue = [ANNA]
    with named_session(client) as ws:
        say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        reply = say(ws, stub_transcribe, "Das klingt aber wirklich schön.",
                    chunks=SHORT)
    assert reply["voice"] == "Anna"


def test_an_unknown_voice_gets_no_name_and_falls_back_to_similarity(
        client, stub_transcribe, voices):
    save_voices(Anna=[1, 0, 0], Bert=[0, 1, 0])
    voices.queue = [np.array([1.0, 0.0, 0.0]), CLARA]
    with named_session(client) as ws:
        say(ws, stub_transcribe, "Wir waren gestern im Garten und")
        second = say(ws, stub_transcribe, "haben dort lange gesessen.")
    assert "voice" not in second and "replaces" not in second


def test_names_stay_off_the_cards_until_asked_for(client, stub_transcribe,
                                                  voices):
    save_voices(Anna=[1, 0], Bert=[0, 1])
    voices.queue = [ANNA]
    with session(client) as ws:
        first = say(ws, stub_transcribe, "Wir waren gestern im Garten.")
    assert "voice" not in first


def test_identify_wants_a_clear_winner():
    save_voices(Anna=[1, 0, 0], Bert=[0, 1, 0])
    assert speakers.identify(np.array([1.0, 0.0, 0.0])) == "Anna"
    assert speakers.identify(np.array([0.0, 0.0, 1.0])) is None    # far
    tie = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    assert speakers.identify(tie) is None                          # too close
    assert speakers.identify(None) is None
    assert speakers.identify(np.array([1.0, 0.0])) is None         # wrong size


def test_a_single_saved_voice_can_still_be_named():
    save_voices(Anna=[1, 0])
    assert speakers.identify(ANNA) == "Anna"
    assert speakers.identify(BERT) is None


def test_no_profiles_means_no_names():
    assert speakers.load_profiles() == ([], None)
    assert speakers.identify(ANNA) is None


def test_profiles_are_normalized_and_reread_when_the_file_changes():
    import os
    save_voices(Anna=[3, 4])
    names, matrix = speakers.load_profiles()
    assert names == ["Anna"] and np.allclose(matrix, [[0.6, 0.8]])
    save_voices(Bert=[0, 2])
    os.utime(speakers.VOICES_PATH, (1, 1))       # a different mtime for sure
    assert speakers.load_profiles()[0] == ["Bert"]


@pytest.mark.parametrize("content", [
    "not json", "[1, 2]", '{"Anna": "x"}', '{"Anna": [1, 0], "Bert": [1]}'])
def test_an_unreadable_profile_file_is_ignored(content, caplog):
    speakers.VOICES_PATH.write_text(content)
    assert speakers.load_profiles() == ([], None)
    assert "unreadable voice profiles" in caplog.text


def test_a_zero_or_nested_vector_is_skipped():
    speakers.VOICES_PATH.write_text('{"Zero": [0, 0], "Nest": [[1, 0]], '
                                    '"Anna": [1, 0]}')
    assert speakers.load_profiles()[0] == ["Anna"]


# The cut chunk is the speech plus about 0.9 s of lead-in and closing silence.
MIDDLE = 12    # about 2.4 s cut: too short to compare, enough to name


def test_a_shorter_chunk_is_named_against_saved_voices_and_breaks(
        client, stub_transcribe, voices):
    """A 2 to 3 s turn by someone else used to stay in the other person's
    paragraph. With saved voices it is placed and gets its own card, and the
    first speaker coming back starts a new card again."""
    save_voices(Anna=[1, 0], Bert=[0, 1])
    voices.queue = [ANNA, BERT, ANNA]
    with named_session(client) as ws:
        first = say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        reply = say(ws, stub_transcribe, "Das klingt aber wirklich schön.",
                    chunks=MIDDLE)
        third = say(ws, stub_transcribe, "Danach haben wir Kaffee getrunken.")
    assert reply["voice"] == "Bert" and "replaces" not in reply
    assert third["voice"] == "Anna" and "replaces" not in third
    assert first["voice"] == "Anna"


def test_a_shorter_chunk_needs_a_wider_margin_to_be_named(
        client, stub_transcribe, voices):
    save_voices(Anna=[1, 0], Bert=[0, 1])
    lean = np.array([0.74, 0.6726])           # nearer Anna by about 0.07
    assert speakers.identify(lean) == "Anna"
    assert speakers.identify(lean, speakers.SHORT_NAME_MARGIN) is None
    voices.queue = [BERT, lean]
    with named_session(client) as ws:
        first = say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        reply = say(ws, stub_transcribe, "Das klingt aber wirklich schön.",
                    chunks=MIDDLE)
    assert reply["replaces"] == first["id"] and reply["voice"] == "Bert"


def test_without_names_a_shorter_chunk_is_not_asked_about(
        client, stub_transcribe, voices):
    voices.queue = [ANNA]
    with session(client) as ws:
        say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        say(ws, stub_transcribe, "Das klingt aber wirklich schön.",
            chunks=MIDDLE)
    assert voices.calls == 1


def test_embed_takes_a_lower_floor_when_asked(monkeypatch):
    monkeypatch.setattr(speakers, "_model", FakeModel())
    audio = (np.ones(int(2.2 * 16000)) * 1000).astype(np.int16)
    assert speakers.embed(audio) is None
    assert speakers.embed(audio, speakers.NAME_MIN_SEC) is not None


# ------------------------------------------------- the user says who it is

def label(ws, card_id, name):
    ws.send_text(json.dumps({"type": "voice_label", "id": card_id,
                             "name": name}))
    return collect_until(ws, stop_types=("voice", "error"))[-1]


def test_turning_names_on_lists_the_saved_voices(client):
    save_voices(Anna=[1, 0], Bert=[0, 1])
    with named_session(client) as ws:
        msg = collect_until(ws, stop_types=("voices",))[-1]
    assert msg["names"] == ["Anna", "Bert"]


def test_naming_a_card_teaches_a_new_voice(client, stub_transcribe, voices):
    """Nobody is saved yet. The user names the first card; the next long
    chunk in that voice is recognized without being told."""
    voices.queue = [CLARA, np.array([0.0, 1.0, 0.0]), CLARA]
    with named_session(client) as ws:
        first = say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        assert "voice" not in first
        reply = label(ws, first["id"], "  Clara ")
        assert reply == {"type": "voice", "id": first["id"], "name": "Clara",
                         "names": ["Clara"]}
        say(ws, stub_transcribe, "Danach haben wir Kaffee getrunken.")
        third = say(ws, stub_transcribe, "Und dann sind wir heimgefahren.")
    assert third["voice"] == "Clara"
    assert speakers.identify(CLARA) == "Clara"


def test_naming_the_live_card_names_what_merges_into_it(client,
                                                        stub_transcribe,
                                                        voices):
    voices.queue = [CLARA]
    with named_session(client) as ws:
        first = say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        label(ws, first["id"], "Clara")
        reply = say(ws, stub_transcribe, "Das klingt aber wirklich schön.",
                    chunks=SHORT)
    assert reply["replaces"] == first["id"] and reply["voice"] == "Clara"


def test_a_merged_card_teaches_all_its_long_chunks(client, stub_transcribe,
                                                   voices, monkeypatch):
    taught = []
    monkeypatch.setattr(speakers, "teach",
                        lambda name, vecs: taught.append((name, len(vecs)))
                        or [name])
    voices.queue = [ANNA, ANNA]
    with named_session(client) as ws:
        say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        second = say(ws, stub_transcribe, "Danach haben wir Kaffee getrunken.")
        label(ws, second["id"], "Anna")
    assert taught == [("Anna", 2)]


@pytest.mark.parametrize("name", ["", "   ", None, 7])
def test_a_label_without_a_name_is_refused(client, stub_transcribe, voices,
                                           name):
    voices.queue = [ANNA]
    with named_session(client) as ws:
        first = say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        assert label(ws, first["id"], name)["type"] == "error"
    assert speakers.load_profiles() == ([], None)


def test_a_card_with_only_short_speech_cannot_teach(client, stub_transcribe,
                                                    voices):
    with named_session(client) as ws:
        first = say(ws, stub_transcribe, "Ja, genau so.", chunks=SHORT)
        reply = label(ws, first["id"], "Anna")
    assert reply["type"] == "error" and "long enough" in reply["message"]


def test_only_recent_cards_keep_their_audio_description(
        client, stub_transcribe, voices, monkeypatch):
    monkeypatch.setattr(srv, "CARD_VECS_KEPT", 1)
    voices.queue = [ANNA, BERT]
    with named_session(client) as ws:
        first = say(ws, stub_transcribe, "Wir waren gestern im Garten.")
        second = say(ws, stub_transcribe, "Danach haben wir Kaffee getrunken.")
        assert label(ws, first["id"], "Anna")["type"] == "error"
        assert label(ws, second["id"], "Bert")["type"] == "voice"


def test_teach_starts_a_voice_and_then_moves_it():
    assert speakers.teach("Anna", [ANNA]) == ["Anna"]
    assert speakers.identify(ANNA) == "Anna"
    speakers.teach("Anna", [BERT])
    names, matrix = speakers.load_profiles()
    assert names == ["Anna"] and np.allclose(matrix, [[np.sqrt(.5)] * 2])
    stored = json.loads(speakers.VOICES_PATH.read_text())["Anna"]
    assert stored["n"] == 2 and len(stored["vec"]) == 2


def test_an_established_voice_still_moves_when_taught():
    """A profile built from hundreds of chunks counts as TEACH_CAP of them,
    so a correction is never rounding error."""
    speakers.VOICES_PATH.write_text(json.dumps(
        {"Anna": {"vec": [1.0, 0.0], "n": 500}}))
    speakers.teach("Anna", [BERT])
    vec = speakers.load_profiles()[1][0]
    assert vec[1] == pytest.approx(
        1 / np.hypot(speakers.TEACH_CAP, 1), rel=1e-4)
    assert json.loads(speakers.VOICES_PATH.read_text())["Anna"]["n"] == 501


def test_teach_keeps_other_voices_and_reads_the_old_format():
    save_voices(Bert=[0, 1])                       # a bare list of floats
    assert speakers.teach("Anna", [ANNA]) == ["Bert", "Anna"]
    assert speakers.identify(BERT) == "Bert"


def test_teach_replaces_a_profile_of_another_size_and_a_broken_file():
    save_voices(Anna=[1, 0, 0])
    speakers.teach("Anna", [BERT])
    assert np.allclose(speakers.load_profiles()[1], [[0, 1]])
    speakers.VOICES_PATH.write_text("not json")
    assert speakers.teach("Bert", [BERT]) == ["Bert"]


def test_each_voice_gets_its_own_stable_color():
    """Run the page's own color function: saved voices take hues in list
    order, away from the language colors, and an unsaved name still gets one."""
    import shutil
    import subprocess
    from pathlib import Path
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    js = (Path(srv.__file__).parent / "static" / "app.js").read_text()
    start = js.index("const VOICE_HUES")
    snippet = js[start:js.index("function setVoiceChip")]
    script = ('let voiceNames = ["Anna", "Bert", "Clara"];\n' + snippet +
              'console.log(JSON.stringify([voiceHue("Anna"), voiceHue("Bert"),'
              ' voiceHue("Clara"), voiceHue("Anna"), voiceHue("Zed")]));')
    out = subprocess.run(["node", "-e", script], capture_output=True,
                         text=True, check=True).stdout
    anna, bert, clara, anna_again, zed = json.loads(out)
    assert len({anna, bert, clara}) == 3 and anna == anna_again
    assert isinstance(zed, int)
    for hue in (anna, bert, clara):       # not the blue, green, or orange
        assert all(abs(hue - lang) > 15 for lang in (212, 153, 37))


def test_the_name_on_a_card_is_a_control():
    from pathlib import Path
    js = (Path(srv.__file__).parent / "static" / "app.js").read_text()
    assert 'type: "voice_label"' in js
    assert "chip.textContent = name" in js           # text, never innerHTML
    assert "b.textContent = name" in js


# ------------------------------------------------------- tools/enroll_voices

def enroll():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "enroll_voices",
        Path(srv.__file__).parent / "tools" / "enroll_voices.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cloud(center, n, seed):
    rng = np.random.default_rng(seed)
    pts = np.asarray(center, dtype=float) + rng.normal(0, 0.05, (n, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def test_grouping_finds_the_voices_biggest_first_and_drops_strays():
    pytest.importorskip("scipy")
    tool = enroll()
    pts = np.vstack([cloud([1, 0, 0], 10, 1), cloud([0, 1, 0], 20, 2),
                     cloud([0, 0, 1], 3, 3)])
    groups = tool.group_voices(pts)
    assert [len(g) for g in groups] == [20, 10]
    assert set(groups[0]) == set(range(10, 30))
    assert tool.group_voices(pts[:1]) == []


def test_a_profile_is_the_unit_mean():
    tool = enroll()
    vec = tool.profile(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(vec, [np.sqrt(0.5), np.sqrt(0.5)])


def test_rename_keeps_order_and_refuses_unknown_names(tmp_path):
    tool = enroll()
    voices = {"Voice 1": [1, 0], "Voice 2": [0, 1]}
    assert list(tool.rename(voices, ["Voice 2=Bert"])) == ["Voice 1", "Bert"]
    for bad in ("Voice 9=X", "Voice 1", "Voice 1= "):
        with pytest.raises(SystemExit):
            tool.rename(voices, [bad])


def test_rename_and_list_from_the_command_line(tmp_path, capsys):
    tool = enroll()
    out = tmp_path / "v" / "voices.json"
    tool.save(out, {"Voice 1": [1, 0]})
    assert tool.main(["--rename", "Voice 1=Anna", "--out", str(out)]) == 0
    assert tool.main(["--list", "--out", str(out)]) == 0
    assert capsys.readouterr().out.strip().endswith("Anna")
    assert tool.load_saved(tmp_path / "missing.json") == {}
    assert tool.stamp(125.4) == "2:05"


def test_the_page_offers_names_and_shows_them():
    from pathlib import Path
    root = Path(srv.__file__).parent / "static"
    js = (root / "app.js").read_text()
    assert 'id="namesChk"' in (root / "index.html").read_text()
    assert "names: namesChk.checked" in js
    assert "voiceChip(msg.id, msg.voice)" in js


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
