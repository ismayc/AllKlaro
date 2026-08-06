"""How the segmenter behaves on a real conversation, not synthesized speech.

`tools/replay.py` speaks with macOS `say`, which never pauses; that turned out
to be an *easier* load than three friends talking over each other, and it led
the earlier investigation to the wrong root cause (see
docs/findings/real-conversation-pace.md).

The fixture here is the per-frame Silero voiced/unvoiced envelope of a real
54-minute German conversation — one bit per 32 ms frame. It carries the
rhythm the VAD reacts to and nothing else: no audio, no words, nothing
recoverable. That keeps a private recording out of the repo while still
letting the chunker be tested against how people actually talk.
"""
import base64
import json
import statistics
import zlib
from pathlib import Path

import numpy as np
import pytest

from server import (FRAME_MS, FRAME_SAMPLES, MAX_UTTERANCE_SEC, VadSession)

FIXTURE = Path(__file__).parent / "fixtures" / "real_conversation_voicing.json"


@pytest.fixture(scope="module")
def voicing():
    data = json.loads(FIXTURE.read_text())
    raw = zlib.decompress(base64.b64decode(data["bits_b64_zlib"]))
    bits = [(raw[i // 8] >> (7 - i % 8)) & 1 for i in range(data["frames"])]
    return data, bits


class ReplayScorer:
    """Voicing replayed from the fixture, one decision per frame."""

    def __init__(self, bits):
        self.bits = bits
        self.i = 0

    def __call__(self, frame):
        v = self.bits[self.i] if self.i < len(self.bits) else 0
        self.i += 1
        return bool(v)


def segment(bits, **kwargs):
    """Run the real segmenter over the fixture; return (duration, reason) pairs."""
    vad = VadSession(ReplayScorer(bits), **kwargs)
    quiet = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    out = []
    for _ in range(len(bits)):
        utt = vad.feed(quiet)
        if utt is not None:
            out.append((len(utt) / 16000, vad.split_reason))
    return out


def test_fixture_describes_a_continuously_voiced_hour(voicing):
    data, bits = voicing
    assert data["frame_ms"] == FRAME_MS, "fixture was built for a different frame size"
    assert len(bits) == data["frames"]
    assert data["duration_sec"] > 3000, "expected ~54 minutes"
    # Three people on a hot afternoon leave the pipeline almost no idle air.
    assert data["voiced_fraction"] > 0.75


def test_real_conversation_never_reaches_the_30s_backstop(voicing):
    """The synthetic `say` load hit hard_max constantly; real speech never does.

    Real speakers leave a ~190 ms micro-pause often enough that `split_at` is
    nearly always available, so the mid-word backstop is not what a listener is
    waiting on. If this ever fails, chunk accumulation has become a real cause
    again and the findings doc needs revisiting.
    """
    _, bits = voicing
    reasons = [r for _, r in segment(bits)]
    assert reasons, "fixture produced no utterances at all"
    assert "hard_max" not in reasons


def test_chunks_stay_conversational_in_length(voicing):
    _, bits = voicing
    durs = sorted(d for d, _ in segment(bits))
    median = statistics.median(durs)
    # Measured 2026-07-27: p50 5.8s, p90 8.5s, max 18.8s.
    assert 4.0 < median < 8.0, f"median chunk {median:.1f}s drifted from ~5.8s"
    assert max(durs) < MAX_UTTERANCE_SEC
    assert durs[int(0.9 * (len(durs) - 1))] < 12.0


def test_chunk_rate_sets_the_pipeline_budget(voicing):
    """~10 chunks a minute is the arrival rate the rest of the pipeline must meet.

    Each one costs a Whisper decode and an Ollama translation, and the
    conversation never goes quiet enough to catch up in. Measured: 553 chunks
    over 54.3 minutes.
    """
    data, bits = voicing
    chunks = segment(bits)
    per_min = len(chunks) / (data["duration_sec"] / 60)
    assert 8.0 < per_min < 13.0, f"{per_min:.1f} chunks/min drifted from ~10.2"


def test_shorter_end_silence_does_not_rescue_the_rate(voicing):
    """Tightening the pause threshold cuts more often, not less.

    Worth pinning: the instinct on seeing lag is to make the VAD cut sooner,
    but that raises the arrival rate into an already over-committed pipeline.
    """
    _, bits = voicing
    base = len(segment(bits))
    tighter = len(segment(bits, end_silence_frames=10))
    assert tighter >= base
