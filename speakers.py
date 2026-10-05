"""Who is speaking, well enough to decide where a paragraph ends.

An optional pretrained speaker model (ECAPA-TDNN, trained on VoxCeleb, loaded
through speechbrain) turns a stretch of speech into a vector; two stretches by
the same voice give vectors that point the same way. The app uses it for one
thing: when a new chunk arrives, does it sound like the person who spoke the
last one? It names someone only when the user has saved voice profiles
(`voices.json`, built from a recording by `tools/enroll_voices.py`) and
turned "Names" on.

It replaces nothing by default. `voiceprint.py` tried the same job with numpy
alone and could not separate a speaker change from the same speaker carrying
on (33.7% against 35.1% on a real hour). This model can, given enough audio:
scored against a transcription service's speaker labels on a 67-minute family
call (2026-10-05), adjacent chunks of 4 s or more separated at AUC 0.97, of
3 to 4 s at 0.95, and under 2 s at 0.74. That is why the caller only asks
about chunks of `MIN_SEC` or more.

Install with `uv sync --extra speakers`. Without it `load()` returns None and
the app keeps its text-and-pause paragraphs.
"""
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("translator")

SAMPLE_RATE = 16000
MODEL_REPO = os.environ.get("ALLKLARO_SPEAKER_MODEL",
                            "speechbrain/spkrec-ecapa-voxceleb")
MODEL_DIR = Path.home() / ".cache" / "allklaro" / "speaker-model"
# Below this much audio the model is close to guessing (AUC 0.74 under 2 s),
# so a shorter chunk neither starts a paragraph nor becomes the reference.
MIN_SEC = 3.0
# Cosine similarity to the last long chunk; under it, a different voice. At
# 0.5 the model caught 77% of the speaker changes that arrived on a long chunk
# and broke 14.5% of the places where the same speaker carried on.
SAME_COS = 0.5
# Saved voices: {"name": {"vec": [192 floats], "n": chunks behind it}}, one
# unit vector per person (a bare list of floats is read too). Private by
# nature (it describes real people's voices), so it lives beside the other
# per-user data and never in the repo.
VOICES_PATH = Path(os.environ.get(
    "ALLKLARO_VOICES", Path.home() / ".cache" / "allklaro" / "voices.json"))
# A chunk gets a name when it is this close to a saved voice and this much
# closer to it than to the next one. With profiles built from the first half
# of a 67-minute call, 98% of the second half's long chunks were named at
# these settings and 98.8% of the names agreed with the reference labels.
NAME_COS = 0.4
NAME_MARGIN = 0.05
# Against saved voices a shorter chunk can be placed too, which a comparison
# with one other chunk cannot do. Held out on the same call: chunks of 2.5 to
# 3 s were named 92% of the time and every name agreed with the reference; of
# 2 to 2.5 s, 79% and 87%; of 1.5 to 2 s, 82% and 83%. So with names on,
# chunks down to NAME_MIN_SEC are named, on a stricter margin, and anything
# shorter still stays in the paragraph it lands in.
NAME_MIN_SEC = 2.0
SHORT_NAME_MARGIN = 0.1
# When the user says whose voice a card is, its chunks are averaged into that
# person's profile. A profile built from hundreds of chunks would barely move,
# so its weight is capped: each taught chunk shifts the profile by at least
# one part in TEACH_CAP + 1.
TEACH_CAP = 40
_voices_lock = threading.Lock()

_model = None
_unavailable = False
_lock = threading.Lock()


def load():
    """Load the model once. None when the optional dependency is missing or
    the model cannot be fetched, which leaves paragraphs on the old rules."""
    global _model, _unavailable
    if _model is not None or _unavailable:
        return _model
    with _lock:
        if _model is not None or _unavailable:
            return _model
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            _model = EncoderClassifier.from_hparams(
                source=MODEL_REPO, savedir=str(MODEL_DIR),
                run_opts={"device": "cpu"})
            log.info("Speaker model ready (%s).", MODEL_REPO)
        except Exception as exc:
            _unavailable = True
            log.warning("Speaker model unavailable (%s); paragraphs keep the "
                        "text-and-pause rules. Install it with "
                        "`uv sync --extra speakers`.", exc)
    return _model


def ready() -> bool:
    return _model is not None


def embed(audio: np.ndarray, min_sec: float = MIN_SEC) -> np.ndarray | None:
    """Unit-length voice vector for one chunk of 16 kHz int16 audio, or None
    when the chunk is too short to describe a voice or no model is loaded."""
    if len(audio) < min_sec * SAMPLE_RATE:
        return None
    model = load()
    if model is None:
        return None
    import torch

    wav = torch.from_numpy(audio.astype(np.float32) / 32768.0)[None]
    with torch.no_grad():
        vec = model.encode_batch(wav)[0, 0].numpy()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else None


_profiles = {"mtime": None, "names": [], "matrix": None}


def load_profiles() -> tuple[list[str], np.ndarray | None]:
    """The saved voices as (names, one unit vector per row). Re-read when the
    file changes, so renaming a voice needs no restart."""
    try:
        mtime = VOICES_PATH.stat().st_mtime
    except OSError:
        _profiles.update(mtime=None, names=[], matrix=None)
        return [], None
    if mtime != _profiles["mtime"]:
        names, rows = [], []
        try:
            for name, (vec, _n) in _read_voices().items():
                names.append(name)
                rows.append(vec)
            matrix = np.stack(rows) if rows else None
        except (ValueError, AttributeError, TypeError):
            log.warning("ignoring unreadable voice profiles in %s", VOICES_PATH)
            names, matrix = [], None
        _profiles.update(mtime=mtime, names=names, matrix=matrix)
    return _profiles["names"], _profiles["matrix"]


def _read_voices() -> dict[str, tuple[np.ndarray, int]]:
    """The file as {name: (unit vector, chunk count)}; unusable entries are
    skipped. Raises ValueError/AttributeError/TypeError on a malformed file."""
    out = {}
    for name, entry in json.loads(VOICES_PATH.read_text()).items():
        count = TEACH_CAP
        if isinstance(entry, dict):
            count = int(entry.get("n", TEACH_CAP))
            entry = entry.get("vec")
        vec = np.asarray(entry, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if vec.ndim == 1 and norm:
            out[str(name)] = (vec / norm, count)
    return out


def teach(name: str, vectors: list[np.ndarray]) -> list[str]:
    """The user said these chunks are `name`: average them into that saved
    voice, or start it. Returns the saved names afterwards."""
    with _voices_lock:
        try:
            voices = _read_voices()
        except (OSError, ValueError, AttributeError, TypeError):
            voices = {}
        total = np.sum(vectors, axis=0)
        count = len(vectors)
        if name in voices and len(voices[name][0]) == len(total):
            vec, had = voices[name]
            total = total + vec * min(had, TEACH_CAP)
            count += had
        voices[name] = (total / np.linalg.norm(total), count)
        VOICES_PATH.parent.mkdir(parents=True, exist_ok=True)
        VOICES_PATH.write_text(json.dumps(
            {n: {"vec": [float(x) for x in v], "n": c}
             for n, (v, c) in voices.items()}))
        _profiles["mtime"] = None            # re-read on the next lookup
        return list(voices)


def identify(vec: np.ndarray | None,
             margin: float = NAME_MARGIN) -> str | None:
    """The saved voice this chunk belongs to, or None when there are no
    profiles, it is not close enough to any, or two are too close to call."""
    names, matrix = load_profiles()
    if vec is None or matrix is None or matrix.shape[1] != len(vec):
        return None
    sims = matrix @ vec
    order = np.argsort(sims)
    best = float(sims[order[-1]])
    runner_up = float(sims[order[-2]]) if len(order) > 1 else -1.0
    if best < NAME_COS or best - runner_up < margin:
        return None
    return names[int(order[-1])]


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Cosine similarity of two voice vectors, None unless both exist."""
    if a is None or b is None:
        return None
    return float(a @ b)
