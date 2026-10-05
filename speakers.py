"""Who is speaking, well enough to decide where a paragraph ends.

An optional pretrained speaker model (ECAPA-TDNN, trained on VoxCeleb, loaded
through speechbrain) turns a stretch of speech into a vector; two stretches by
the same voice give vectors that point the same way. The app uses it for one
thing: when a new chunk arrives, does it sound like the person who spoke the
last one? It never names anyone.

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


def embed(audio: np.ndarray) -> np.ndarray | None:
    """Unit-length voice vector for one chunk of 16 kHz int16 audio, or None
    when the chunk is too short to describe a voice or no model is loaded."""
    if len(audio) < MIN_SEC * SAMPLE_RATE:
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


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Cosine similarity of two voice vectors, None unless both exist."""
    if a is None or b is None:
        return None
    return float(a @ b)
