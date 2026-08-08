"""Detect that the voice changed between two utterances. Not who is speaking.

The distinction is the whole design. `SPEAKERS = {0: "you", 1: "them"}` is a
channel tag: it records which input stream the audio arrived on, and with one
microphone carrying a whole room every card says "you". Real diarization would
answer "who", and costs a third model on a GPU that Whisper, Parakeet and
Ollama already share. This answers the much smaller question — *did the voice
change* — from the audio already in hand, in numpy, in about a millisecond,
and says nothing further.

What it compares:

- **Long-term average spectrum**, 24 log-spaced bands from 100 Hz to 4 kHz,
  in log power, mean-removed so overall loudness cancels out. This is vocal
  timbre: roughly, the shape of someone's vocal tract.
- **Median F0**, by autocorrelation over voiced frames. Pitch is the single
  most separating cheap feature between adult speakers, and it is what carries
  the detector when two people have similar timbre.

Both are needed. Timbre alone confuses people with similar builds; pitch alone
confuses two speakers in the same range, and vanishes entirely on a whispered
or creaky utterance where no periodicity is detectable.

Known limits, stated because they bound what the mark can claim:
  - One utterance with two speakers in it reads as one voice. The VAD splits
    on pauses, not on speakers, so a fast interruption lands in one chunk.
  - The same speaker's own signature drifts with loudness, distance from the
    mic, and laughter. Distance from the mic is the big one in a room.
  - It cannot tell "Anna again" from "a third person who sounds like Anna",
    because it never builds identities — only adjacent comparisons.
"""
import numpy as np

SAMPLE_RATE = 16000
FRAME = 512                    # 32 ms
HOP = 256                      # 16 ms
BANDS = 24
BAND_LO, BAND_HI = 100.0, 4000.0
# Below this there is not enough voiced audio to describe a voice; short
# backchannels ("mhm", "ja") are the common case and must return nothing
# rather than a signature built from noise.
MIN_VOICED_FRAMES = 20         # ~0.3 s of speech
F0_LO, F0_HI = 70.0, 350.0     # adult speech, generously bracketed
F0_MIN_FRAMES = 8              # voiced frames needed before F0 is trusted
# How periodic a frame must be to count as voiced, as a fraction of its own
# energy. Swept over the synthetic corpus, and it is a genuine trade rather
# than a free parameter: at 0.3 weak transitional frames get in and the SAME
# speaker's p90 distance is 0.642 — larger than the gap it has to fit inside.
# At 0.6 precision reaches 1.000 but recall falls to 0.69, because so many
# utterances lose their fundamental entirely and fall back to timbre alone.
# 0.45 is where the distributions stop overlapping at all: same-speaker p90
# 0.161 against different-speaker p10 0.295.
VOICED_STRENGTH = 0.45
# Weight on pitch relative to timbre in the combined distance. Chosen so that
# an octave of pitch difference counts about as much as a completely
# uncorrelated spectrum; see tools/voice_eval.py for the sweep.
PITCH_WEIGHT = 0.6


def _frames(audio: np.ndarray) -> np.ndarray:
    """Overlapping analysis frames, Hann-windowed."""
    n = 1 + (len(audio) - FRAME) // HOP
    if n < 1:
        return np.empty((0, FRAME), dtype=np.float32)
    strided = np.lib.stride_tricks.as_strided(
        audio, shape=(n, FRAME),
        strides=(audio.strides[0] * HOP, audio.strides[0]))
    return strided * np.hanning(FRAME).astype(np.float32)


def _band_edges() -> np.ndarray:
    """Log-spaced FFT bin boundaries — the ear's spacing, roughly."""
    hz = np.geomspace(BAND_LO, BAND_HI, BANDS + 1)
    return np.round(hz * FRAME / SAMPLE_RATE).astype(int)


_EDGES = _band_edges()


def _median_f0(frames: np.ndarray) -> float | None:
    """Median fundamental over frames that are actually periodic.

    Autocorrelation rather than a fancier estimator: this only has to be
    stable enough to separate two people, not accurate enough to transcribe a
    melody, and it costs one FFT per frame on audio already in memory.
    """
    if len(frames) == 0:
        return None
    lo, hi = int(SAMPLE_RATE / F0_HI), int(SAMPLE_RATE / F0_LO)
    size = 1 << int(np.ceil(np.log2(2 * FRAME)))
    spec = np.fft.rfft(frames, size, axis=1)
    corr = np.fft.irfft(spec * np.conj(spec), size, axis=1)[:, :hi + 1]
    energy = corr[:, :1]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(energy > 0, corr / energy, 0.0)
    window = norm[:, lo:hi + 1]
    if window.shape[1] == 0:
        return None
    peak = window.argmax(axis=1)
    strength = window[np.arange(len(window)), peak]
    # A clear periodic peak means the frame is voiced. Unvoiced frames
    # (fricatives, breath) have no fundamental and would otherwise contribute
    # an arbitrary lag.
    voiced = strength > VOICED_STRENGTH
    if voiced.sum() < F0_MIN_FRAMES:
        return None
    return float(np.median(SAMPLE_RATE / (peak[voiced] + lo)))


def voice_signature(audio: np.ndarray) -> dict | None:
    """A compact description of the voice in `audio`, or None if too little.

    Returning None matters as much as returning a signature: a chunk with no
    usable speech must not produce a confident-looking vector that the next
    comparison then treats as a different person.
    """
    if audio is None or len(audio) < FRAME * 2:
        return None
    samples = np.asarray(audio)
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
        if np.issubdtype(np.asarray(audio).dtype, np.integer):
            samples /= 32768.0
    samples = np.ascontiguousarray(samples - samples.mean())

    frames = _frames(samples)
    if len(frames) == 0:
        return None
    energy = (frames ** 2).sum(axis=1)
    if not np.any(energy > 0):
        return None
    # Keep the louder half. A VAD chunk still carries leading and trailing
    # near-silence, and averaging that in flattens the spectrum toward the
    # room rather than the person.
    loud = frames[energy >= np.percentile(energy, 40)]
    if len(loud) < MIN_VOICED_FRAMES:
        return None

    # F0 over *all* frames, not the loud ones. The voicing test below already
    # selects periodic frames, so filtering by energy first is not just
    # redundant — it couples the two. Measured: appending 1.5 s of unvoiced
    # noise to a vowel shifted the estimate 19 Hz, not because the noise was
    # counted (no noise frame passes the voicing test) but because it raised
    # the energy percentile and so changed *which voiced frames* survived it.
    # Decoupled, the same audio with and without the noise gives the identical
    # answer.
    f0 = _median_f0(frames)

    power = np.abs(np.fft.rfft(loud, axis=1)) ** 2
    bands = np.stack([power[:, a:b].sum(axis=1) if b > a else power[:, a:a + 1].sum(axis=1)
                      for a, b in zip(_EDGES[:-1], _EDGES[1:])], axis=1)
    ltas = np.log10(bands + 1e-12).mean(axis=0)
    ltas -= ltas.mean()                    # loudness cancels; shape remains
    norm = np.linalg.norm(ltas)
    if norm == 0:
        return None
    return {"ltas": ltas / norm, "f0": f0, "frames": int(len(loud))}


def voice_distance(a: dict | None, b: dict | None) -> float | None:
    """How different two voices sound, or None if either is unusable.

    None is not zero. "We could not tell" and "the same person" have to stay
    distinguishable, because the caller's default on None is to draw no mark.
    """
    if not a or not b:
        return None
    timbre = float(1.0 - np.dot(a["ltas"], b["ltas"]))
    if a["f0"] and b["f0"]:
        # In octaves, so a difference means the same thing high or low in the
        # range — 110 Hz vs 220 Hz is one octave, and so is 150 vs 300.
        pitch = abs(np.log2(a["f0"] / b["f0"]))
        return timbre + PITCH_WEIGHT * float(pitch)
    return timbre
