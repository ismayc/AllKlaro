"""Unit tests for the VAD segmenter and its voicing scorers."""
from types import SimpleNamespace

import numpy as np

from server import (EARLY_SILENCE_FRAMES, END_SILENCE_FRAMES, FRAME_SAMPLES,
                    EnergyScorer, SileroScorer, VadSession)


def silence():
    return np.zeros(FRAME_SAMPLES, dtype=np.int16)


def speech(amp=8000):
    t = np.arange(FRAME_SAMPLES)
    return (amp * np.sin(2 * np.pi * 440 * t / 16000)).astype(np.int16)


def feed_all(vad, frames):
    return [vad.feed(f) for f in frames]


class ScriptScorer:
    """Voicing decisions scripted per frame, for exact state-machine tests."""

    def __init__(self, seq):
        self.seq = list(seq)

    def __call__(self, frame):
        return self.seq.pop(0) if self.seq else False


# ----------------------------------------------------------- state machine


def test_silence_never_triggers():
    vad = VadSession()
    results = feed_all(vad, [silence()] * 100)
    assert all(r is None for r in results)
    assert not vad.in_speech


def test_utterance_detected_and_finalized():
    vad = VadSession()
    frames = [silence()] * 10 + [speech()] * 40 + [silence()] * 30
    utterances = [u for u in feed_all(vad, frames) if u is not None]
    assert len(utterances) == 1
    utt = utterances[0]
    assert utt.dtype == np.float32
    assert np.abs(utt).max() <= 1.0
    assert 40 * FRAME_SAMPLES <= len(utt) <= 80 * FRAME_SAMPLES
    assert not vad.in_speech


def test_short_blip_discarded():
    vad = VadSession()
    frames = [speech()] * 6 + [silence()] * 30  # ~190 ms voiced < MIN_UTTERANCE_SEC
    assert all(u is None for u in feed_all(vad, frames))


def test_two_utterances_are_separated():
    vad = VadSession()
    frames = ([speech()] * 30 + [silence()] * 30) * 2
    utterances = [u for u in feed_all(vad, frames) if u is not None]
    assert len(utterances) == 2


def test_long_speech_force_flushed():
    vad = VadSession()
    utterances = [u for u in feed_all(vad, [speech()] * 1000) if u is not None]
    assert len(utterances) >= 1  # MAX_UTTERANCE_SEC flush, no silence needed


def test_start_needs_consecutive_voiced_frames():
    # Voiced pairs separated by gaps never reach the 3-in-a-row trigger.
    vad = VadSession(ScriptScorer([True, True, False] * 30))
    feed_all(vad, [speech()] * 90)
    assert not vad.in_speech


def test_configurable_end_silence():
    # 40 voiced frames, then silence; flush must happen after exactly 8
    # unvoiced frames when end_silence_frames=8.
    vad = VadSession(ScriptScorer([True] * 40), end_silence_frames=8)
    results = feed_all(vad, [speech()] * 40 + [silence()] * 20)
    flush_positions = [i for i, r in enumerate(results) if r is not None]
    assert flush_positions == [47]


def test_current_audio_needs_a_second_of_speech():
    vad = VadSession()
    assert vad.current_audio() is None  # idle
    feed_all(vad, [speech()] * 10)
    assert vad.current_audio() is None  # too short for a stable partial
    feed_all(vad, [speech()] * 30)
    audio = vad.current_audio()
    assert audio is not None
    assert audio.dtype == np.float32
    assert len(audio) >= 32 * FRAME_SAMPLES


def test_continuous_speech_soft_splits_at_micro_pause():
    # 150 voiced frames, a ~190 ms dip, then 200 more voiced frames — no real
    # pause ever happens (like video audio). Once past SOFT_MAX_SEC the VAD
    # must emit a chunk ending at the micro-pause and keep listening.
    vad = VadSession(ScriptScorer([True] * 150 + [False] * 6 + [True] * 200))
    frames = [speech()] * 150 + [silence()] * 6 + [speech()] * 200
    emitted = []
    for i, f in enumerate(frames):
        u = vad.feed(f)
        if u is not None:
            emitted.append((i, u))
    assert len(emitted) == 1
    assert vad.in_speech                      # still capturing the rest
    _, chunk = emitted[0]
    assert len(chunk) <= 160 * FRAME_SAMPLES  # ends at the micro-pause

    # The remainder finalizes normally on real silence.
    tail = [vad.feed(silence()) for _ in range(30)]
    assert any(u is not None for u in tail)


def test_partial_window_is_bounded():
    vad = VadSession(ScriptScorer([True] * 500))
    for _ in range(240):  # keep under SOFT_MAX so no split interferes
        vad.feed(speech())
    assert vad.in_speech
    audio = vad.current_audio()
    assert len(audio) <= 375 * FRAME_SAMPLES


def test_early_event_fires_during_pause_and_matches_final():
    vad = VadSession(ScriptScorer([True] * 40))
    early = None
    final = None
    for f in [speech()] * 40 + [silence()] * 30:
        u = vad.feed(f)
        if vad.early_event is not None and early is None:
            early = vad.early_event
            vad.early_event = None
            assert vad.speculating
        if u is not None:
            final = u
    assert early is not None and final is not None
    # The final utterance is the speculative audio plus the rest of the pause,
    # which is how the server validates that a speculation is reusable.
    assert len(final) == len(early) + (
        END_SILENCE_FRAMES - EARLY_SILENCE_FRAMES) * FRAME_SAMPLES


def test_speculation_marked_stale_when_speech_resumes():
    vad = VadSession(ScriptScorer(
        [True] * 40 + [False] * 12 + [True] * 20))
    saw_early = False
    for f in [speech()] * 40 + [silence()] * 12 + [speech()] * 20:
        vad.feed(f)
        if vad.early_event is not None:
            saw_early = True
            vad.early_event = None
    assert saw_early
    assert not vad.speculating  # resumed speech invalidated it


def test_no_early_event_when_pause_setting_is_very_short():
    # end_silence <= EARLY means the flush would win the race; skip speculation.
    vad = VadSession(ScriptScorer([True] * 40), end_silence_frames=10)
    for f in [speech()] * 40 + [silence()] * 12:
        vad.feed(f)
        assert vad.early_event is None


# ---------------------------------------------------------------- scorers


class FakeSileroSession:
    """Silero v5-shaped session with scripted speech probabilities."""

    def __init__(self, probs):
        self.probs = list(probs)

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in ("input", "state", "sr")]

    def run(self, _outputs, feeds):
        return np.array([[self.probs.pop(0)]]), feeds["state"]


def test_silero_hysteresis_start_hard_continue_easy():
    scorer = SileroScorer(FakeSileroSession([0.6, 0.4, 0.3, 0.4]))
    frame = speech()
    assert scorer(frame) is True    # 0.6 > 0.5: speech starts
    assert scorer(frame) is True    # 0.4 > 0.35: quiet word-ending continues
    assert scorer(frame) is False   # 0.3 < 0.35: speech ends
    assert scorer(frame) is False   # 0.4 < 0.5: doesn't restart on noise


def test_energy_noise_floor_adapts_to_steady_background():
    noisy = np.full(FRAME_SAMPLES, 250, dtype=np.int16)   # below initial threshold
    louder = np.full(FRAME_SAMPLES, 600, dtype=np.int16)

    assert EnergyScorer()(louder)  # fresh scorer: 600 rms reads as speech

    adapted = EnergyScorer()
    for _ in range(200):
        adapted(noisy)             # let the noise floor rise to ~250
    assert not adapted(louder)     # same level no longer reads as speech


def test_energy_scorer_detects_tone_over_silence():
    scorer = EnergyScorer()
    assert not scorer(silence())
    assert scorer(speech())
