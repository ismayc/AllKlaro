#!/usr/bin/env python3
"""How much lag could be saved by translating from a partial? Size it first.

    uv run python tools/partial_headroom.py --audio ~/.cache/allklaro/demo4.wav

Translating from live partials instead of waiting for the VAD to close the
chunk is the last untried lever on accumulation, which is ~65% of first-word
lag at p50. Before building it, the same question that killed the speculation
idea in item 1: **how much time is actually there?**

The headroom is the gap between the moment a partial is good enough to
translate and the moment the final chunk is emitted. Everything before that
moment is not available at any price — the words had not been said yet.

This measures it offline against the real recording, using the app's own VAD
and its own partial-pass ASR, with no Ollama involved at all. Deterministic
given the same audio, so unlike a live A/B it does not need bracketing.

"Good enough to translate" is the judgement call, so it is a knob and the
report sweeps it: a partial qualifies once it has at least `--min-chars`
characters and its text has stopped changing except by extension for
`--stable` consecutive partials. Both matter — length because a three-word
prefix translates to nothing useful, stability because Parakeet rewrites its
own output as more audio arrives.
"""
import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def read(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def collect(path: Path, interval: float):
    """Replay the audio through the VAD, recording partials and finals.

    Timeline is audio position, not wall clock: this is a property of the
    conversation, not of how fast this Mac happens to be today.
    """
    import server as srv

    srv.load_silero()
    vad = srv.VadSession(srv.make_scorer())
    audio = read(path)
    frame = srv.FRAME_SAMPLES
    utterances, partials, last_partial = [], [], -1e9
    start_t = 0.0

    for i in range(0, len(audio) - frame, frame):
        t = i / srv.SAMPLE_RATE
        chunk = vad.feed(audio[i:i + frame])
        if vad.in_speech and t - last_partial >= interval:
            window = vad.current_audio()
            if window is not None:
                text = srv.transcribe_partial(window)
                if text:
                    partials.append((t, srv.clean_partial(text) or ""))
                last_partial = t
        if chunk is not None:
            utterances.append({"t_emit": t, "sec": len(chunk) / srv.SAMPLE_RATE,
                               "split": vad.split_reason, "audio": chunk,
                               "partials": [p for p in partials
                                            if p[0] > start_t]})
            start_t = t
            partials = []
            last_partial = -1e9
    return utterances


def first_usable(parts, min_chars: int, stable: int):
    """When the partial first qualified, or None.

    "Stable" means the previous text is a prefix of the current one: Parakeet
    extending its guess is fine, Parakeet *revising* it means the words are
    still moving and a translation would be of something the speaker did not
    say.
    """
    run = 0
    for i, (t, text) in enumerate(parts):
        if len(text) < min_chars:
            run = 0
            continue
        prev = parts[i - 1][1] if i else ""
        run = run + 1 if (i and text.startswith(prev)) else 1
        if run >= stable:
            return t, text
    return None


def report(utterances, min_chars, stable):
    saved, missed, covered = [], 0, []
    for u in utterances:
        hit = first_usable(u["partials"], min_chars, stable)
        if hit is None:
            missed += 1
            continue
        t, text = hit
        saved.append(max(0.0, u["t_emit"] - t))
        covered.append(len(text))
    n = len(utterances)
    if not saved:
        print(f"  min_chars={min_chars:3d} stable={stable}:  no utterance ever "
              f"qualified ({n} utterances)")
        return
    arr = np.array(saved)
    print(f"  min_chars={min_chars:3d} stable={stable}:  "
          f"fires on {len(saved):3d}/{n} ({len(saved)/n:4.0%})   "
          f"saving p50 {np.median(arr):5.2f}s  p90 {np.percentile(arr, 90):5.2f}s  "
          f"max {arr.max():5.2f}s")


def fidelity(utterances, min_chars, stable):
    """Does the partial we would have translated survive into the final?

    Saving time is worthless if the text is wrong: a translation of words the
    speaker did not say has to be replaced, and the user read it in the
    meantime. This transcribes each final chunk with the real Whisper and
    asks how much of the partial's wording actually holds up.
    """
    import server as srv
    from difflib import SequenceMatcher

    rows = []
    for u in utterances:
        hit = first_usable(u["partials"], min_chars, stable)
        if hit is None or u.get("audio") is None:
            continue
        _t, ptext = hit
        final = srv.clean_transcript(
            srv.transcribe(u["audio"], None, None)) or ""
        pw, fw = ptext.lower().split(), final.lower().split()
        # How far the partial agrees with the final from the start — a
        # translation only survives if the *beginning* holds, since that is
        # what was already on screen.
        agree = 0
        for a, b in zip(pw, fw):
            if a.strip(".,!?") != b.strip(".,!?"):
                break
            agree += 1
        rows.append({"ratio": SequenceMatcher(None, ptext.lower(),
                                              final.lower()).ratio(),
                     "prefix": agree / max(1, len(pw)),
                     "partial": ptext, "final": final})
    return rows


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", required=True)
    p.add_argument("--interval", type=float, default=None,
                   help="partial cadence (default: the server's own)")
    p.add_argument("--min-chars", type=int, default=None)
    p.add_argument("--stable", type=int, default=None)
    p.add_argument("--fidelity", action="store_true",
                   help="also ask whether the partial's words survive")
    args = p.parse_args()

    import server as srv
    interval = args.interval or srv.PARTIAL_INTERVAL_SEC
    utterances = collect(Path(args.audio), interval)
    withp = sum(1 for u in utterances if u["partials"])
    print(f"\n{len(utterances)} utterances, {withp} had any partial at all "
          f"(cadence {interval}s)\n")

    if args.min_chars is not None and args.stable is not None:
        report(utterances, args.min_chars, args.stable)
        if args.fidelity:
            rows = fidelity(utterances, args.min_chars, args.stable)
            r = np.array([x["ratio"] for x in rows])
            pre = np.array([x["prefix"] for x in rows])
            print(f"\n  fidelity on {len(rows)} firing utterances:")
            print(f"    similarity to final  p50 {np.median(r):.2f}  "
                  f"p10 {np.percentile(r, 10):.2f}")
            print(f"    leading words that survive verbatim  p50 {np.median(pre):.0%}"
                  f"  share with none at all {np.mean(pre == 0):.0%}")
            print("\n  worst five, partial -> final:")
            for x in sorted(rows, key=lambda x: x["ratio"])[:5]:
                print(f"    [{x['ratio']:.2f}] {x['partial'][:64]}")
                print(f"           -> {x['final'][:64]}")
        return 0
    print("How much of the wait is reachable, by how strict 'usable' is:")
    for stable in (1, 2, 3):
        for min_chars in (20, 40, 60):
            report(utterances, min_chars, stable)
    print("\n  'saving' is the gap between a usable partial and the emitted")
    print("  chunk. It is an UPPER BOUND on what translating from partials")
    print("  could win: the translation itself still has to run, and a")
    print("  German prefix can translate very wrongly — the verb is at the")
    print("  end, so 'Ich habe das nicht…' and 'Ich habe das nicht gemacht'")
    print("  say opposite things in English.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
