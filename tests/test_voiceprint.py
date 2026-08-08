"""Voice-change marks: the claim is "the voice changed", never "this is Anna".

The threshold that ships is not the highest-scoring one. Swept over synthetic
voices (`tools/voice_eval.py`) F1 peaks at 0.125; 0.35 ships because the two
errors are not symmetric — a false mark asserts a speaker change that did not
happen, while a missed one only omits a divider — and because synthetic voices
are an upper bound on separability.
"""
import json

import numpy as np
import pytest

import server as srv
import voiceprint as vp
from conftest import (SILENCE_CHUNK, collect_until_bounded, speak,
                      trace_records)

RATE = vp.SAMPLE_RATE


def synth_voice(f0, formants, seconds=2.0, amplitude=6000, seed=0):
    """A vowel: harmonics of `f0`, their amplitudes shaped by resonances.

    Not speech, but it varies the two things the detector actually reads —
    fundamental and spectral shape — which is what lets these tests pin
    behaviour without shipping audio of anyone's voice in a public repo.

    Built as a harmonic stack rather than a buzz multiplied by carriers: the
    multiplied version produced sidebands that were not multiples of f0, so
    it was only weakly periodic and the voicing test rejected it — a stand-in
    for voiced speech that is not itself voiced proves nothing.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(RATE * seconds)) / RATE
    out = np.zeros_like(t)
    for k in range(1, int((RATE / 2) / f0)):
        hz = f0 * k
        # Each formant is a resonance the harmonic sits nearer to or further
        # from; 120 Hz half-width is roughly a real one.
        gain = sum(g / (1 + ((hz - centre) / 120.0) ** 2)
                   for centre, g in formants)
        out += (gain + 0.04) * np.sin(2 * np.pi * hz * t) / k
    out += rng.normal(0, 0.005, len(out))      # a little breath
    out /= np.abs(out).max()
    return (out * amplitude).astype(np.int16)


LOW = dict(f0=110, formants=[(500, 1.0), (1100, 0.7), (2400, 0.3)])
HIGH = dict(f0=220, formants=[(700, 1.0), (1900, 0.8), (2900, 0.4)])


def test_a_signature_describes_the_voice():
    sig = vp.voice_signature(synth_voice(**LOW))
    assert sig is not None
    assert sig["ltas"].shape == (vp.BANDS,)
    assert np.isclose(np.linalg.norm(sig["ltas"]), 1.0)
    assert 90 < sig["f0"] < 135, f"f0 {sig['f0']} is not near the 110 Hz source"


def test_the_same_voice_twice_is_not_a_change():
    a = vp.voice_signature(synth_voice(seed=1, **LOW))
    b = vp.voice_signature(synth_voice(seed=2, **LOW))
    assert vp.voice_distance(a, b) < srv.VOICE_CHANGE_DIST


def test_two_different_voices_are_a_change():
    a = vp.voice_signature(synth_voice(**LOW))
    b = vp.voice_signature(synth_voice(**HIGH))
    assert vp.voice_distance(a, b) > srv.VOICE_CHANGE_DIST


def test_loudness_alone_is_never_a_change():
    """Someone leaning toward the microphone is the commonest way to fake a
    speaker change, and the mean-removal in the signature exists for it."""
    quiet = vp.voice_signature(synth_voice(amplitude=1200, **LOW))
    loud = vp.voice_signature(synth_voice(amplitude=20000, **LOW))
    assert vp.voice_distance(quiet, loud) < srv.VOICE_CHANGE_DIST


def test_pitch_separates_voices_that_timbre_cannot():
    """Two people with near-identical timbre are exactly where spectrum alone
    fails, and are why F0 is in the distance at all.

    Built as signatures rather than as audio: a synthesized voice cannot hold
    its spectrum fixed while its fundamental moves — shifting f0 moves every
    harmonic and so moves the bands too — so audio cannot isolate this.
    """
    shape = vp.voice_signature(synth_voice(**LOW))["ltas"]
    a = {"ltas": shape, "f0": 105.0, "frames": 60}
    b = {"ltas": shape, "f0": 210.0, "frames": 60}
    assert float(1.0 - np.dot(a["ltas"], b["ltas"])) == pytest.approx(0, abs=1e-6)
    assert vp.voice_distance(a, b) > srv.VOICE_CHANGE_DIST
    # ...and an octave is the unit, so the same ratio counts the same high or
    # low in the range.
    c = {"ltas": shape, "f0": 150.0, "frames": 60}
    d = {"ltas": shape, "f0": 300.0, "frames": 60}
    assert vp.voice_distance(c, d) == pytest.approx(vp.voice_distance(a, b))


def test_a_missing_fundamental_falls_back_to_timbre_alone():
    """A whispered or creaky utterance has no detectable f0. The comparison
    must degrade to spectrum rather than refuse or invent a pitch."""
    shape = vp.voice_signature(synth_voice(**LOW))["ltas"]
    a = {"ltas": shape, "f0": None, "frames": 60}
    b = {"ltas": shape, "f0": 220.0, "frames": 60}
    assert vp.voice_distance(a, b) == pytest.approx(0, abs=1e-6)


def test_breath_and_fricatives_do_not_move_the_pitch():
    """Real speech is only part periodic — "s", "f", "ch" and breath have no
    fundamental at all. Estimating F0 across those frames as if they did
    yields whatever lag the noise happened to peak at, and the median walks
    away from the speaker's actual pitch.

    The synthetic vowels elsewhere in this file are periodic end to end, so
    they cannot show this: it needs audio with genuinely unvoiced stretches.
    """
    rng = np.random.default_rng(7)
    voiced = synth_voice(seconds=1.0, **LOW).astype(np.float64)
    # Unvoiced on purpose the MAJORITY of the utterance — a whispered aside,
    # or a short vowel between long fricatives. An even split proves nothing:
    # the median still lands on the voiced half whether or not the noise
    # frames were excluded.
    noise = rng.normal(0, np.abs(voiced).mean(), int(RATE * 2.5))
    mixed = np.concatenate([voiced, noise]).astype(np.int16)

    clean = vp.voice_signature(synth_voice(seconds=1.0, **LOW))
    with_noise = vp.voice_signature(mixed)
    assert with_noise is not None and with_noise["f0"] is not None
    assert abs(with_noise["f0"] - clean["f0"]) < 5, (
        f"unvoiced frames dragged F0 from {clean['f0']:.0f} Hz to "
        f"{with_noise['f0']:.0f} Hz")


@pytest.mark.parametrize("audio", [
    np.zeros(RATE, dtype=np.int16),                    # pure silence
    np.zeros(100, dtype=np.int16),                     # shorter than a frame
    np.array([], dtype=np.int16),
])
def test_unusable_audio_yields_no_signature(audio):
    """None must not be a vector. A confident-looking signature built from
    silence would make the next real utterance read as a new voice."""
    assert vp.voice_signature(audio) is None


def test_too_short_to_judge_yields_no_signature():
    """"Mhm" and "ja" are most of a conversation's utterances and carry no
    usable voice."""
    assert vp.voice_signature(synth_voice(seconds=0.15, **LOW)) is None


def test_an_unknown_distance_is_not_zero():
    """None and 0.0 mean opposite things — "we cannot tell" versus "identical"
    — and the caller draws no mark on None."""
    sig = vp.voice_signature(synth_voice(**LOW))
    assert vp.voice_distance(sig, None) is None
    assert vp.voice_distance(None, sig) is None
    assert vp.voice_distance(None, None) is None


def test_float_and_int16_audio_agree():
    """The pipeline hands over int16; the tools sometimes hold float."""
    ints = synth_voice(**LOW)
    a = vp.voice_signature(ints)
    b = vp.voice_signature(ints.astype(np.float32) / 32768.0)
    assert vp.voice_distance(a, b) < 0.01


# ------------------------------------------------------ through the pipeline


def speak_with(ws, audio, silence_chunks=8):
    """Feed real samples through the websocket the way the worklet would."""
    for _ in range(3):
        ws.send_bytes(SILENCE_CHUNK)
    step = 2048
    for i in range(0, len(audio) - step, step):
        ws.send_bytes(audio[i:i + step].tobytes())
    for _ in range(silence_chunks):
        ws.send_bytes(SILENCE_CHUNK)


def test_the_marks_are_off_by_default():
    """Measured on the real 54-minute recording, this does not work.

    Using the pipeline's own split reasons as labels — a `soft_max` cut means
    the 5 s cap fired mid-sentence so the next utterance is the SAME speaker,
    a `pause` cut is where a turn actually changes — the detector marks 33.7%
    of continuations and 35.1% of pauses. The same number: one mark in three
    would land in the middle of somebody's sentence.

    Synthetic voices said the opposite (same-speaker p90 0.161 against
    different-speaker p10 0.295) because TTS is far more self-consistent, and
    far more mutually distinct, than three people round one microphone.

    So the default is off, and this test is what stops it drifting back on
    without new evidence.
    """
    assert srv.VOICE_MARKS_ON is False


def test_the_distance_is_still_measured_while_the_marks_are_off(
        client, stub_transcribe, trace_file, monkeypatch):
    """Off means "do not claim it on screen", not "stop looking".

    The distance is what a future attempt — a real speaker-embedding model,
    say — would be re-tuned against, and it costs about a millisecond.
    """
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    assert srv.VOICE_MARKS_ON is False
    with client.websocket_connect("/ws") as ws:
        speak_with(ws, synth_voice(seed=1, **LOW))
        collect_until_bounded(ws, ("translation_done", "discard", "error"))
        speak_with(ws, synth_voice(**HIGH))
        msgs = collect_until_bounded(ws, ("translation_done", "discard", "error"))
    assert next(m for m in msgs if m["type"] == "final")["voice_change"] is False
    recs = [r for r in trace_records(trace_file) if r.get("uid")]
    assert recs[-1]["voice_dist"] is not None, "stopped measuring as well"


def test_the_first_utterance_is_never_a_change(client, stub_transcribe):
    """There is nothing to have changed from, and a break above the very first
    card would be pure noise."""
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until_bounded(ws, ("translation_done", "discard", "error"))
    final = next(m for m in msgs if m["type"] == "final")
    assert final["voice_change"] is False


def test_a_spoken_final_always_carries_the_field(client, stub_transcribe):
    """Absent and False must not be the same thing to the client: a missing
    field means a server too old to send it."""
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs = collect_until_bounded(ws, ("translation_done", "discard", "error"))
    assert "voice_change" in next(m for m in msgs if m["type"] == "final")


def test_a_new_voice_marks_and_the_same_voice_does_not(client,
                                                       stub_transcribe,
                                                       monkeypatch):
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    monkeypatch.setattr(srv, "VOICE_MARKS_ON", True)
    with client.websocket_connect("/ws") as ws:
        speak_with(ws, synth_voice(seed=1, **LOW))
        collect_until_bounded(ws, ("translation_done", "discard", "error"))
        speak_with(ws, synth_voice(seed=2, **LOW))
        same = collect_until_bounded(ws, ("translation_done", "discard", "error"))
        speak_with(ws, synth_voice(**HIGH))
        other = collect_until_bounded(ws, ("translation_done", "discard", "error"))
    assert next(m for m in same if m["type"] == "final")["voice_change"] is False
    assert next(m for m in other if m["type"] == "final")["voice_change"] is True


def test_a_merged_fragment_never_shows_a_break(client, stub_transcribe,
                                               monkeypatch):
    """A merge means one person carried on through a micro-pause — that is why
    it merged. A "new voice" break above the joined card would contradict the
    merge that just happened.

    The detector is forced to "changed" rather than fed two voices: what is
    under test is the suppression, and making the audio both merge *and* sound
    like two people would be testing the VAD's split behaviour by accident.
    """
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    monkeypatch.setattr(vp, "voice_distance", lambda a, b: 99.0)
    stub_transcribe.result = {"text": "Ich glaube, dass wir das Projekt",
                              "language": "de"}
    with client.websocket_connect("/ws") as ws:
        speak(ws)
        msgs1 = collect_until_bounded(ws, ("translation_done", "discard",
                                           "error"))
        stub_transcribe.result = {"text": "nächste Woche abschließen werden.",
                                  "language": "de"}
        speak(ws)
        msgs2 = collect_until_bounded(ws, ("translation_done", "discard",
                                           "error"))
    final1 = next(m for m in msgs1 if m["type"] == "final")
    final2 = next(m for m in msgs2 if m["type"] == "final")
    assert final2["replaces"] == final1["id"], "these fragments did not merge"
    assert final2["voice_change"] is False, \
        "a merged continuation was marked as a new voice"


def test_an_undescribable_utterance_does_not_blind_the_next_comparison(
        client, stub_transcribe, monkeypatch):
    """"Mhm" between two people must not cost the mark on the real change.

    If a chunk with no usable voice became the new reference, the comparison
    after it would be against nothing — which reads as "cannot tell" and draws
    no mark. The break would then be silently lost at exactly the moment the
    speaker changed, and backchannels are most of a conversation's utterances.
    """
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    monkeypatch.setattr(srv, "VOICE_MARKS_ON", True)
    real = vp.voice_signature
    calls = {"n": 0}

    def sometimes_unusable(audio):
        calls["n"] += 1
        return None if calls["n"] == 2 else real(audio)

    monkeypatch.setattr(vp, "voice_signature", sometimes_unusable)
    stop = ("translation_done", "discard", "error")
    with client.websocket_connect("/ws") as ws:
        speak_with(ws, synth_voice(seed=1, **LOW))
        collect_until_bounded(ws, stop)
        speak_with(ws, synth_voice(seed=2, **LOW))     # the unusable one
        collect_until_bounded(ws, stop)
        speak_with(ws, synth_voice(**HIGH))
        msgs = collect_until_bounded(ws, stop)
    assert calls["n"] >= 3, "the middle utterance never reached the detector"
    assert next(m for m in msgs if m["type"] == "final")["voice_change"] is True


def test_the_trace_records_the_distance(client, stub_transcribe, trace_file,
                                        monkeypatch):
    """Recorded so a threshold can be re-chosen from a real call rather than
    from synthetic voices — the one thing tools/voice_eval.py cannot supply."""
    monkeypatch.setattr(srv, "PARTIAL_INTERVAL_SEC", 1e9)
    with client.websocket_connect("/ws") as ws:
        speak_with(ws, synth_voice(seed=1, **LOW))
        collect_until_bounded(ws, ("translation_done", "discard", "error"))
        speak_with(ws, synth_voice(**HIGH))
        collect_until_bounded(ws, ("translation_done", "discard", "error"))
    recs = [r for r in trace_records(trace_file) if r.get("uid")]
    assert recs[-1]["voice_dist"] is not None
    assert recs[0]["voice_dist"] is None, "nothing preceded the first utterance"
