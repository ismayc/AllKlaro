#!/usr/bin/env python3
"""Measure the voice-change detector against known speaker boundaries.

    uv run python tools/voice_eval.py                 # sweep and report
    uv run python tools/voice_eval.py --threshold .35 # score one setting

**Read the caveat before quoting any number from this.** The ground truth is
synthetic: several macOS `say` voices, each speaking several sentences, so
every boundary is known exactly. That is the only way to get labels without
someone sitting down with the real recording and a stopwatch — but synthetic
voices are *more separable* than real ones sharing a room. They have no
overlap, no varying distance from the microphone, no laughter, and no
identical twins. **Every accuracy figure here is an upper bound**, and the
same mistake as trusting `say` for pipeline load (see PROGRESS.md) is
available here in a different costume.

What the real recording can tell us — and this script also reports — is the
*rate* at which marks would appear. It has no labels, so a rate is descriptive
only: 3+ friends really do change speaker constantly, and a high rate there is
not evidence of false positives.
"""
import argparse
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import voiceprint  # noqa: E402

# Deliberately mixed in pitch and build, and all German so the phonetics are
# comparable — a detector that only separates a bass from a soprano would
# score well on an easier set and fail on this conversation.
VOICES = ["Anna", "Reed (German (Germany))", "Grandpa (German (Germany))",
          "Shelley (German (Germany))", "Rocko (German (Germany))",
          "Sandy (German (Germany))"]

LINES = [
    "Wir haben gestern noch über den Garten gesprochen.",
    "Das Wasser verdunstet eigentlich auch eine ganze Menge.",
    "Ich wusste gar nicht, dass das so komfortabel ist.",
    "Meistens guckt der Arvid nach, wenn er hier ist.",
    "Dann war es windig und danach hat es geregnet.",
    "Also das automatische Teil ist auch nur ein kleiner Ballon.",
]


def synth(voice: str, text: str, out: Path) -> Path:
    subprocess.run(["say", "-v", voice, "--data-format=LEI16@16000",
                    "--file-format=WAVE", "-o", str(out), text], check=True)
    return out


def read(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def build_corpus(tmp: Path) -> list[tuple[str, np.ndarray]]:
    """(speaker, audio) pairs — every voice says every line."""
    corpus = []
    for vi, voice in enumerate(VOICES):
        for li, line in enumerate(LINES):
            path = tmp / f"v{vi}_l{li}.wav"
            corpus.append((voice, read(synth(voice, line, path))))
    return corpus


def pairs(corpus):
    """Every ordered pair, with whether it is a real speaker change.

    All pairs rather than one conversation ordering: a single interleaving
    samples a handful of the transitions and its score moves with the shuffle.
    """
    sigs = [(spk, voiceprint.voice_signature(audio)) for spk, audio in corpus]
    out = []
    for i, (spk_a, a) in enumerate(sigs):
        for spk_b, b in sigs[i + 1:]:
            d = voiceprint.voice_distance(a, b)
            if d is not None:
                out.append((spk_a != spk_b, d))
    unusable = sum(1 for _, s in sigs if s is None)
    return out, unusable


def score(data, threshold):
    tp = sum(1 for changed, d in data if changed and d > threshold)
    fp = sum(1 for changed, d in data if not changed and d > threshold)
    fn = sum(1 for changed, d in data if changed and d <= threshold)
    tn = sum(1 for changed, d in data if not changed and d <= threshold)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def real_utterances(path: Path):
    """Chunk a real recording with the app's own VAD, so the utterances are
    exactly the ones the feature would see."""
    import server as srv

    srv.load_silero()
    vad = srv.VadSession(srv.make_scorer())
    audio = read(path)
    out = []
    for i in range(0, len(audio) - srv.FRAME_SAMPLES, srv.FRAME_SAMPLES):
        chunk = vad.feed(audio[i:i + srv.FRAME_SAMPLES])
        if chunk is not None:
            out.append(chunk)
    return out


def report_real(path: Path, thresholds) -> None:
    """No labels exist here, so this reports a RATE and nothing more.

    Three-plus friends trading turns really do change speaker constantly, so a
    high rate is not evidence of false positives — and a low one is not
    evidence of correctness. It is here to catch the two useless extremes:
    marking nothing, and marking everything.
    """
    chunks = real_utterances(path)
    sigs = [voiceprint.voice_signature(c) for c in chunks]
    usable = sum(s is not None for s in sigs)
    dists = [d for a, b in zip(sigs, sigs[1:])
             if (d := voiceprint.voice_distance(a, b)) is not None]
    print(f"\n{path.name}: {len(chunks)} utterances, {usable} describable, "
          f"{len(dists)} adjacent pairs comparable")
    if dists:
        print(f"  adjacent distance p10 {np.percentile(dists, 10):.3f}  "
              f"p50 {np.median(dists):.3f}  p90 {np.percentile(dists, 90):.3f}")
    for t in thresholds:
        marked = sum(1 for d in dists if d > t)
        pct = marked / len(dists) * 100 if dists else 0
        print(f"  threshold {t:.3f}: {marked}/{len(dists)} pairs marked "
              f"({pct:.0f}%)")
    print("  Descriptive only — this recording has no speaker labels.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float)
    p.add_argument("--real", help="a real recording: report mark RATE only")
    args = p.parse_args()

    if args.real:
        report_real(Path(args.real), [0.175, 0.25, 0.30, 0.35, 0.45])
        return 0

    with tempfile.TemporaryDirectory(prefix="allklaro-voice-") as td:
        data, unusable = pairs(build_corpus(Path(td)))

    same = [d for changed, d in data if not changed]
    diff = [d for changed, d in data if changed]
    print(f"{len(VOICES)} voices x {len(LINES)} lines "
          f"({unusable} utterances too short to describe)")
    print(f"  same speaker : n={len(same):4d}  p50 {np.median(same):.3f}  "
          f"p90 {np.percentile(same, 90):.3f}")
    print(f"  different    : n={len(diff):4d}  p50 {np.median(diff):.3f}  "
          f"p10 {np.percentile(diff, 10):.3f}")

    if args.threshold is not None:
        print("\n", score(data, args.threshold))
        return 0

    print("\n threshold  precision  recall     f1     (marks drawn)")
    best = None
    for t in np.arange(0.10, 0.80, 0.025):
        s = score(data, round(float(t), 3))
        drawn = s["tp"] + s["fp"]
        print(f"   {s['threshold']:.3f}     {s['precision']:.3f}     "
              f"{s['recall']:.3f}   {s['f1']:.3f}    {drawn}")
        if best is None or s["f1"] > best["f1"]:
            best = s
    print(f"\nbest f1 at threshold {best['threshold']:.3f}: "
          f"precision {best['precision']:.3f}, recall {best['recall']:.3f}")
    print("\nThese are synthetic voices and therefore an UPPER BOUND — real "
          "speakers\nin one room, at varying distance from one microphone, "
          "are harder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
