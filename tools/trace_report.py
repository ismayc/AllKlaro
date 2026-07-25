#!/usr/bin/env python3
"""Summarize the pipeline trace the server writes, one line per utterance.

The card in the UI shows only a chunk's own transcribe+translate time, which
stays small even while the app visibly falls behind. This adds the parts that
number leaves out — how long the audio sat accumulating before it was cut,
how much was already queued on the single Whisper thread, and why each chunk
was cut at all — and attributes the delay a listener actually feels.

    uv run python tools/trace_report.py                # whole file
    uv run python tools/trace_report.py --last 20      # most recent 20
    uv run python tools/trace_report.py --since 10m    # last 10 minutes
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

DEFAULT_PATH = "/tmp/allklaro-trace.jsonl"


def load(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"no trace at {path} — run a session first, or pass --path "
                 "(ALLKLARO_TRACE=off disables tracing)")
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue      # a torn last line from a session still in progress
    return out


def parse_since(spec: str) -> float:
    unit = spec[-1]
    mult = {"s": 1, "m": 60, "h": 3600}.get(unit)
    return float(spec[:-1]) * mult if mult else float(spec)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def row(label: str, values: list[float], unit: str = "ms") -> str:
    if not values:
        return f"  {label:<18} —"
    scale = 1000 if unit == "ms" else 1
    fmt = (lambda v: f"{v / scale:.1f}s") if unit == "ms" else (lambda v: f"{v:.1f}s")
    return (f"  {label:<18} p50 {fmt(pct(values, .5)):>7}   "
            f"p90 {fmt(pct(values, .9)):>7}   max {fmt(max(values)):>7}")


def report(records: list[dict]) -> None:
    finals = [r for r in records if r.get("outcome") == "final"]
    print(f"{len(records)} utterances  ({len(finals)} translated, "
          f"{len(records) - len(finals)} discarded or failed)")
    if not finals:
        return

    splits = Counter(r.get("split") for r in records)
    total = sum(splits.values())
    print("\nwhy each chunk was cut")
    for reason, n in splits.most_common():
        note = {"pause": "the speaker stopped — the good case",
                "soft_max": "continuous speech, cut at a micro-pause",
                "hard_max": "30 s hard limit, cut mid-word"}.get(reason, "")
        print(f"  {str(reason):<10} {n:>4}  {n / total:>5.0%}  {note}")

    specs = Counter(r.get("spec") for r in records)
    if specs["hit"] or specs["miss"]:
        wasted = specs["miss"]
        print(f"\nspeculative decodes: {specs['hit']} used, {wasted} wasted"
              + (f"  ({wasted} full transcriptions that occupied the one "
                 "Whisper thread for nothing)" if wasted else ""))

    print("\nwhere the seconds go")
    print(row("chunk length", [r.get("chunk_sec", 0) for r in finals], "s"))
    print(row("queued (wait)", [r.get("wait_ms", 0) for r in finals]))
    print(row("transcribe", [r.get("transcribe_ms", 0) for r in finals]))
    print(row("translate", [r.get("translate_ms", 0) for r in finals]))
    refine = [r["refine_ms"] for r in finals if r.get("refine_ms")]
    if refine:
        print(row("refine (after)", refine))
    print(row("lag, last word", [r.get("lag_ms", 0) for r in finals]))
    print(row("lag, FIRST word", [r.get("first_word_lag_ms", 0) for r in finals]))

    # Attribute the felt delay. Everything here is measured, not modelled:
    # the chunk had to finish accumulating, then wait, then be transcribed,
    # then translated, and only then did the card appear.
    parts = {
        "audio accumulating": sum(r.get("chunk_sec", 0) * 1000 for r in finals),
        "queued behind other work": sum(r.get("wait_ms", 0) for r in finals),
        "transcribing": sum(r.get("transcribe_ms", 0) for r in finals),
        "translating": sum(r.get("translate_ms", 0) for r in finals),
    }
    grand = sum(parts.values()) or 1
    print("\nthe delay you feel, attributed")
    for name, ms in sorted(parts.items(), key=lambda kv: -kv[1]):
        bar = "█" * round(28 * ms / grand)
        print(f"  {name:<26} {ms / grand:>4.0%} {bar}")

    queues = [r.get("whisper_queue", 0) for r in records]
    backed_up = sum(1 for q in queues if q > 0)
    print(f"\nWhisper queue when a chunk arrived: max {max(queues)}, "
          f"non-empty for {backed_up}/{len(queues)} chunks")
    lost = sum(r.get("partials_skipped", 0) for r in records)
    if lost:
        print(f"live partials skipped because the pipe was busy: {lost} "
              "(that is blank screen while someone is still talking)")

    redos = sum(1 for r in records if r.get("redo"))
    if redos:
        print(f"language re-decodes (a second full pass): {redos}")

    hot = max(finals, key=lambda r: r.get("first_word_lag_ms", 0))
    print(f"\nworst case: #{hot.get('uid')} — {hot.get('chunk_sec')}s of audio "
          f"cut by {hot.get('split')}, first word was "
          f"{hot.get('first_word_lag_ms', 0) / 1000:.1f}s old on screen")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", default=DEFAULT_PATH)
    p.add_argument("--last", type=int, help="only the N most recent utterances")
    p.add_argument("--since", help="only the last 30s / 10m / 2h")
    p.add_argument("--json", action="store_true", help="dump the records instead")
    args = p.parse_args()

    records = load(Path(args.path))
    if args.since:
        cutoff = time.time() - parse_since(args.since)
        records = [r for r in records if r.get("t", 0) >= cutoff]
    if args.last:
        records = records[-args.last:]
    if not records:
        sys.exit("no records in that window")
    if args.json:
        for r in records:
            print(json.dumps(r))
        return 0
    report(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
