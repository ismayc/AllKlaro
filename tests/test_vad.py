"""Unit tests for the VAD segmenter and its voicing scorers."""
from types import SimpleNamespace

import numpy as np

from server import (EARLY_SILENCE_FRAMES, END_SILENCE_FRAMES, FRAME_MS,
                    FRAME_SAMPLES, MICRO_PAUSE_FRAMES, SOFT_MAX_SEC,
                    EnergyScorer, SileroScorer, VadSession)


def frames_for(seconds):
    return int(seconds * 1000 / FRAME_MS)


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


def test_split_reason_distinguishes_pause_from_being_outpaced():
    """The trace leans on split_reason to tell "someone finished a sentence"
    apart from "nobody ever paused and we cut them off"."""
    vad = VadSession()
    feed_all(vad, [silence()] * 10 + [speech()] * 40 + [silence()] * 30)
    assert vad.split_reason == "pause"

    vad = VadSession(ScriptScorer([True] * 150 + [False] * 6 + [True] * 200))
    frames = [speech()] * 150 + [silence()] * 6 + [speech()] * 200
    emitted = [u for u in feed_all(vad, frames) if u is not None]
    assert len(emitted) == 1
    assert vad.split_reason == "soft_max"


def test_split_reason_flags_the_mid_word_hard_cut():
    # Voiced without any micro-pause: no soft split is possible, so the chunk
    # survives to MAX_UTTERANCE_SEC and gets cut mid-word.
    vad = VadSession(ScriptScorer([True] * 2000))
    frames = [speech()] * 1000       # 1000 * 32 ms = 32 s > MAX_UTTERANCE_SEC
    emitted = [u for u in feed_all(vad, frames) if u is not None]
    assert len(emitted) == 1
    assert vad.split_reason == "hard_max"


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


# --------------------------------------------- why soft_max splits go unspec'd
#
# Every `spec:none` in the live traces is a soft_max split, and there are
# exactly two ways to get one. Both come from the same root: the split point
# is chosen at MICRO_PAUSE_FRAMES but a speculation only launches at
# EARLY_SILENCE_FRAMES, so a split can land on a dip that never became one.
# These two tests pin the geometry that makes the difference *free* in one
# case and merely *unrecoverable* in the other — see PROGRESS.md item 1.


def test_over_cap_split_fires_before_a_speculation_can_launch():
    """Speech runs past the cap with no dip at all. The first micro-pause both
    sets `split_at` and triggers the cut on the SAME frame, at
    MICRO_PAUSE_FRAMES — four frames before a speculation would launch.

    This is the majority `spec:none` case, and it costs nothing: the cut is
    simultaneous with the split decision, so there is no dead time for a
    speculation to have exploited. Submitting the audio speculatively would
    submit it at the very same instant.
    """
    voiced = frames_for(SOFT_MAX_SEC) + 40      # well past the cap, no dip
    vad = VadSession(ScriptScorer([True] * voiced + [False] * 30))
    saw_early = False
    emitted = []
    for f in [speech()] * voiced + [silence()] * 30:
        u = vad.feed(f)
        if vad.early_event is not None:
            saw_early = True
            vad.early_event = None
        if u is not None:
            emitted.append((u, vad.split_reason, vad.silence_run))

    chunk, reason, silence_run = emitted[0]
    assert reason == "soft_max"
    # The cut happens *at* the micro-pause, not at the speculation trigger.
    assert silence_run == MICRO_PAUSE_FRAMES
    assert MICRO_PAUSE_FRAMES < EARLY_SILENCE_FRAMES  # the whole reason
    assert not saw_early    # nothing was launched before the chunk was emitted
    # The chunk is longer than the cap, which is the trace signature of it.
    assert len(chunk) / FRAME_SAMPLES > frames_for(SOFT_MAX_SEC)


def test_stale_split_point_emits_old_audio_with_no_speculation():
    """A dip long enough to set `split_at` but too short to launch a
    speculation (MICRO <= run < EARLY), then speech resumes and runs past the
    cap. The cut fires mid-word when `seconds` crosses the cap, using that
    stale split point.

    Unlike the case above this one *does* leave dead time — the gap between
    the dip and the cap crossing — but nothing is in flight to fill it.
    """
    dip = EARLY_SILENCE_FRAMES - 1              # 9: sets split_at, no spec
    assert MICRO_PAUSE_FRAMES <= dip < EARLY_SILENCE_FRAMES
    lead = frames_for(2.0)
    rest = frames_for(SOFT_MAX_SEC)             # carries well past the cap
    pattern = [True] * lead + [False] * dip + [True] * rest
    vad = VadSession(ScriptScorer(pattern))
    audio = [speech()] * lead + [silence()] * dip + [speech()] * rest

    saw_early = False
    emitted = []
    for f in audio:
        u = vad.feed(f)
        if vad.early_event is not None:
            saw_early = True
            vad.early_event = None
        if u is not None:
            emitted.append((u, vad.split_reason, vad.silence_run))

    chunk, reason, silence_run = emitted[0]
    assert reason == "soft_max"
    assert not saw_early          # the 9-frame dip never reached EARLY
    # The cut lands during voiced speech, not in a pause at all.
    assert silence_run == 0
    # It emits audio that ended back at the dip, well under the cap, while
    # more than a cap's worth had accumulated — that is the "stale" part.
    assert len(chunk) / FRAME_SAMPLES < frames_for(SOFT_MAX_SEC)
    assert vad.in_speech          # the rest is still being captured


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


def test_soft_max_picks_the_last_micro_pause_before_the_cap(monkeypatch):
    """Lowering SOFT_MAX_SEC has to cut at an *earlier* micro-pause, not just
    trip sooner at the same one — that is the whole lever for accumulation
    lag, which is 65% of first-word lag at p50 on the real recording.

    Two dips: one 3.2 s in, one 7.2 s in. A cap above both must take the
    later one; a cap between them must take the earlier one.
    """
    import server as srv

    def run(cap):
        monkeypatch.setattr(srv, "SOFT_MAX_SEC", cap)
        script = [True] * 94 + [False] * 6 + [True] * 119 + [False] * 6 \
            + [True] * 100
        frames = [speech()] * 94 + [silence()] * 6 + [speech()] * 119 \
            + [silence()] * 6 + [speech()] * 100
        vad = srv.VadSession(ScriptScorer(script))
        out = [u for u in feed_all(vad, frames) if u is not None]
        assert out, f"cap {cap}: no soft_max split happened"
        assert vad.split_reason == "soft_max"
        return len(out[0]) / FRAME_SAMPLES        # chunk length in frames

    late = run(8.0)
    early = run(4.0)
    assert late == 225, late      # the second dip, at 7.2 s
    assert early == 100, early    # the first dip, at 3.2 s
    assert early < late           # the lever actually moves the cut point


def test_soft_max_sec_is_env_overridable():
    """The A/B sweeps this from the environment so both arms run identical
    source; if the override silently did nothing, every arm would measure the
    same 8.0 s and the comparison would look like pure noise."""
    import importlib
    import os

    import server as srv

    assert srv.SOFT_MAX_SEC == 5.0            # measured default when unset
    prev = os.environ.get("ALLKLARO_SOFT_MAX_SEC")
    os.environ["ALLKLARO_SOFT_MAX_SEC"] = "3.5"
    try:
        reloaded = importlib.reload(srv)
        assert reloaded.SOFT_MAX_SEC == 3.5
    finally:
        if prev is None:
            del os.environ["ALLKLARO_SOFT_MAX_SEC"]
        else:
            os.environ["ALLKLARO_SOFT_MAX_SEC"] = prev
        importlib.reload(srv)                 # leave the module as we found it
