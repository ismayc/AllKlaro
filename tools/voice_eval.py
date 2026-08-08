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

**The real recording CAN be labelled**, which this script also does and which
settles the question — see `boundary_labels`. The pipeline's own split reasons
say where a speaker cannot have changed (a `soft_max` cut interrupts someone
mid-sentence) and where one may have (a 700 ms pause). Scored that way, the
shipped signature marks 33.7% of same-speaker continuations against 35.1% of
real turn boundaries: no separation, which is why the marks are off by
default. Use `--boundaries` to re-score any replacement the same way.
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

SAMPLE_RATE = voiceprint.SAMPLE_RATE

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


def real_utterances(path: Path, with_reason=False):
    """Chunk a real recording with the app's own VAD, so the utterances are
    exactly the ones the feature would see.

    With `with_reason`, each chunk carries the split that ended it — which is
    what turns an unlabelled recording into a labelled one, see
    `boundary_labels`.
    """
    import server as srv

    srv.load_silero()
    vad = srv.VadSession(srv.make_scorer())
    audio = read(path)
    out = []
    for i in range(0, len(audio) - srv.FRAME_SAMPLES, srv.FRAME_SAMPLES):
        chunk = vad.feed(audio[i:i + srv.FRAME_SAMPLES])
        if chunk is not None:
            out.append((chunk, vad.split_reason) if with_reason else chunk)
    return out


def boundary_labels(path: Path):
    """Real speaker labels from an unlabelled recording, via the split reason.

    This is the measurement the section spent a long time believing was
    impossible without paid diarization or someone listening with a
    stopwatch. It is neither.

    A `soft_max` cut means the 5 s cap fired while somebody was still
    talking, so the NEXT utterance continues the **same speaker**. A `pause`
    cut is 700 ms of silence, which is exactly where a turn changes. So
    continuations are a same-speaker set and pauses are a may-have-changed
    set, both made of whole utterances, both adjacent in time, differing only
    in whether a speaker change was possible. A detector that works separates
    them; one that does not, does not.

    Returns (continuation distances, pause distances).
    """
    chunks = real_utterances(path, with_reason=True)
    sigs = [(voiceprint.voice_signature(c), r) for c, r in chunks]
    cont, turn = [], []
    for (a, reason), (b, _) in zip(sigs, sigs[1:]):
        d = voiceprint.voice_distance(a, b)
        if d is not None:
            (cont if reason == "soft_max" else turn).append(d)
    return cont, turn


def report_boundaries(path: Path, thresholds) -> None:
    cont, turn = boundary_labels(path)
    print(f"\n{path.name}: {len(cont)} continuations (same speaker), "
          f"{len(turn)} pause boundaries (turn may change)")
    print(f"  continuation p50 {np.median(cont):.3f}  p90 {np.percentile(cont, 90):.3f}")
    print(f"  pause        p50 {np.median(turn):.3f}  p90 {np.percentile(turn, 90):.3f}")
    print("\n  threshold   marked across a continuation   across a pause   gap")
    for t in thresholds:
        fp = sum(d > t for d in cont) / len(cont)
        hit = sum(d > t for d in turn) / len(turn)
        print(f"    {t:.2f}          {fp:6.1%} (want LOW)            "
              f"{hit:6.1%}      {hit - fp:+.1%}")
    print("\n  A working detector shows a large positive gap. The shipped")
    print("  signature shows ~0 at every threshold, which is why the marks")
    print("  are off by default.")


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


def split_half_distances(chunks, min_sec=2.0):
    """Same-speaker distances measured on REAL audio, without labels.

    The two halves of one VAD utterance are the same person by construction —
    the VAD splits on pauses, and a chunk containing a complete speaker change
    is rare. So this yields a genuine same-speaker distribution recorded
    through one microphone in a real room, which is the distribution that
    governs false marks and the one synthetic voices cannot supply.

    It is optimistic and the amount matters: two halves of one sentence are
    seconds apart, at the same distance from the mic, at the same loudness.
    The same person two minutes later is further away than this says. Read it
    as a *floor* on same-speaker distance, not an estimate of it.
    """
    out = []
    for audio in chunks:
        if len(audio) < min_sec * SAMPLE_RATE:
            continue
        mid = len(audio) // 2
        d = voiceprint.voice_distance(voiceprint.voice_signature(audio[:mid]),
                                      voiceprint.voice_signature(audio[mid:]))
        if d is not None:
            out.append(d)
    return out


def split_ends_distances(chunks, min_sec=3.0):
    """The same idea with the middle thrown away — first third against last.

    Widens the gap between the two samples, which is the main thing making
    `split_half_distances` optimistic.
    """
    out = []
    for audio in chunks:
        if len(audio) < min_sec * SAMPLE_RATE:
            continue
        third = len(audio) // 3
        d = voiceprint.voice_distance(
            voiceprint.voice_signature(audio[:third]),
            voiceprint.voice_signature(audio[-third:]))
        if d is not None:
            out.append(d)
    return out


def validate_real(path: Path, threshold: float) -> None:
    print(f"chunking {path.name} with the app's own VAD…")
    chunks = real_utterances(path)
    secs = sum(len(c) for c in chunks) / SAMPLE_RATE
    print(f"{len(chunks)} utterances, {secs / 60:.1f} min of speech\n")

    half = split_half_distances(chunks)
    ends = split_ends_distances(chunks)
    sigs = [voiceprint.voice_signature(c) for c in chunks]
    adjacent = [d for a, b in zip(sigs, sigs[1:])
                if (d := voiceprint.voice_distance(a, b)) is not None]

    def line(name, xs):
        if not xs:
            print(f"  {name:34} (none)")
            return
        print(f"  {name:34} n={len(xs):4d}  p50 {np.median(xs):.3f}  "
              f"p90 {np.percentile(xs, 90):.3f}  p99 {np.percentile(xs, 99):.3f}"
              f"  >thr {sum(x > threshold for x in xs) / len(xs):5.1%}")

    print(f"REAL same-speaker (no labels needed), threshold {threshold}:")
    line("same utterance, half vs half", half)
    line("same utterance, first vs last third", ends)
    print("\nMixed same/different, for contrast:")
    line("adjacent utterances", adjacent)
    print("\n  '>thr' on the same-speaker rows is the FALSE-MARK rate: those")
    print("  pairs are one person, so anything above the threshold would have")
    print("  drawn a 'new voice' line that is simply wrong.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float)
    p.add_argument("--real", help="a real recording: report mark RATE only")
    p.add_argument("--validate-real",
                   help="a real recording: label-free same-speaker distances")
    p.add_argument("--boundaries",
                   help="a real recording: score against split-reason labels")
    args = p.parse_args()

    if args.boundaries:
        report_boundaries(Path(args.boundaries),
                          [0.25, 0.30, 0.35, 0.45, 0.60])
        return 0
    if args.validate_real:
        validate_real(Path(args.validate_real), args.threshold or 0.35)
        return 0
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
