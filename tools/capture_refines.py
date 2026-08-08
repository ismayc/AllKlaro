#!/usr/bin/env python3
"""Capture what the refine pass actually replaced, under live pacing.

`wording_ab.py` answered the offline question — give both models every
utterance and diff them. This answers the live one, which is different and
harder: under real pacing most refines never run at all, and the ones that do
are competing with the next card for the same Ollama. What lands is a biased
sample of what was tried, so it has to be read as it happened rather than
reconstructed.

    # server on 8710, real models, then:
    uv run python tools/capture_refines.py --audio ~/.cache/allklaro/demo4.wav
    uv run python tools/capture_refines.py --replay-trace /tmp/allklaro-trace.jsonl

The join that matters. A card's text can change for two different reasons —
the refine pass, or the declension guard (`enforce_agreement`) — and they look
identical from outside: both arrive in one `translation_revised`. Reading an
agreement retry as a landed refine inflates exactly the number this tool
exists to measure. The server now names the outcome per utterance
(`refine`, `refine_changed`, `agreement_changed`), so this joins on that
rather than guessing from elapsed milliseconds.

The older heuristic — "a refine under the timeout landed, one at ~0 ms was
skipped" — is what produced the bogus 39% delivery figure, because a
gate-skipped refine also sits near 0 ms. Do not reintroduce it.
"""
import argparse
import asyncio
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import websockets  # noqa: E402

import replay  # noqa: E402

# Outcomes the server records per utterance, in the order worth reading them.
OUTCOMES = ("landed", "timeout", "gated", "error", "off")

_TYPOGRAPHY = {"’": "'", "‘": "'", "“": '"', "”": '"',
               "…": "...", "—": "-", "–": "-"}


def normalize(text: str) -> str:
    """Text with the two models' punctuation habits flattened.

    Measured on the real slice: gemma3:12b writes curly apostrophes and
    qwen2.5:7b writes straight ones, so one of eleven "the refine changed the
    text" rows was `it's` versus `it’s` and nothing else. Counting that as a
    changed translation overstates what the pass delivers, and the effect
    grows with how many contractions a passage happens to contain.
    """
    out = unicodedata.normalize("NFKC", text or "")
    for bad, good in _TYPOGRAPHY.items():
        out = out.replace(bad, good)
    return re.sub(r"\s+", " ", out).strip()


def collect_drafts(events: list[dict]) -> dict[tuple[int, str], str]:
    """The streamed draft per (uid, target), reassembled from the deltas.

    This is the text the user actually read first, which is the only fair
    thing to compare a refine against.
    """
    drafts: dict[tuple[int, str], list[str]] = {}
    for e in events:
        if e.get("type") == "translation_delta":
            key = (e["id"], e.get("target", ""))
            drafts.setdefault(key, []).append(e.get("text", ""))
    return {k: "".join(v) for k, v in drafts.items()}


def collect_revisions(events: list[dict]) -> dict[int, dict[str, str]]:
    """Post-refine text per uid, per target.

    An empty `texts` is meaningful, not missing: with a draft the message is
    always sent so the UI can clear its "refining…" hint, and empty means the
    final text is the draft unchanged.
    """
    out: dict[int, dict[str, str]] = {}
    for e in events:
        if e.get("type") == "translation_revised":
            out[e["id"]] = dict(e.get("texts") or {})
    return out


def collect_sources(events: list[dict]) -> dict[int, dict]:
    """The heard text per uid, following merges.

    A merged card replaces an earlier one, and the replaced uid never gets a
    translation of its own — so it must not be reported as a refine that
    vanished.
    """
    src, replaced = {}, set()
    for e in events:
        if e.get("type") != "final":
            continue
        src[e["id"]] = {"text": e.get("text", ""), "source": e.get("source", "")}
        if e.get("replaces") is not None:
            replaced.add(e["replaces"])
    for uid in replaced:
        src.pop(uid, None)
    return src


def join_refines(events: list[dict], records: list[dict]) -> list[dict]:
    """One row per utterance that reached translation, refine outcome named.

    `changed_by` is the whole point: "refine" rows are the ones worth reading
    by hand, "agreement" rows are the declension guard doing its job and must
    not be counted as refines.
    """
    drafts = collect_drafts(events)
    revisions = collect_revisions(events)
    sources = collect_sources(events)
    by_uid = {r["uid"]: r for r in records if "uid" in r}

    rows = []
    for uid, src in sorted(sources.items()):
        rec = by_uid.get(uid)
        if rec is None or rec.get("outcome") != "final":
            continue                      # discarded, or still in flight
        target = rec.get("target") or next(
            (t for (u, t) in drafts if u == uid), "")
        draft = drafts.get((uid, target), "")
        final = revisions.get(uid, {}).get(target, draft)
        refine_changed = bool(rec.get("refine_changed"))
        agreement_changed = bool(rec.get("agreement_changed"))
        if refine_changed and agreement_changed:
            changed_by = "both"
        elif refine_changed:
            changed_by = "refine"
        elif agreement_changed:
            changed_by = "agreement"
        else:
            changed_by = None
        rows.append({
            "uid": uid,
            "source": src["source"],
            "heard": src["text"],
            "target": target,
            "draft": draft,
            "final": final,
            # Whether anything a reader would notice changed, as opposed to
            # the two models' punctuation styles differing.
            "substantive": (changed_by in ("refine", "both")
                            and normalize(draft) != normalize(final)),
            # "+"-joined for a multi-target mode; single-target is the norm.
            "outcome": rec.get("refine", ""),
            "changed_by": changed_by,
            "wait_ms": rec.get("refine_wait_ms"),
            "refine_ms": rec.get("refine_ms"),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    counts = {o: 0 for o in OUTCOMES}
    for r in rows:
        for part in (r["outcome"] or "").split("+"):
            if part in counts:
                counts[part] += 1
    attempted = counts["landed"] + counts["timeout"] + counts["error"]
    changed = [r for r in rows if r["changed_by"] in ("refine", "both")]
    waits = sorted(r["wait_ms"] for r in rows
                   if r["wait_ms"] and "landed" in (r["outcome"] or ""))
    return {
        "utterances": len(rows),
        "counts": counts,
        "attempted": attempted,
        "kill_rate": counts["timeout"] / attempted if attempted else None,
        "changed_by_refine": len(changed),
        "substantive": sum(r["substantive"] for r in rows),
        "typographic_only": len(changed) - sum(r["substantive"] for r in rows),
        "changed_by_agreement": sum(r["changed_by"] == "agreement"
                                    for r in rows),
        "landed_wait_p50": waits[len(waits) // 2] if waits else None,
    }


def report(rows: list[dict], show_unchanged: bool = False) -> None:
    s = summarize(rows)
    c = s["counts"]
    print(f"\n{s['utterances']} utterances translated")
    print(f"  refine attempted : {s['attempted']}  "
          f"(landed {c['landed']}, timed out {c['timeout']}, "
          f"error {c['error']})")
    print(f"  never attempted  : gated {c['gated']}, no draft model {c['off']}")
    if s["kill_rate"] is not None:
        print(f"  kill rate        : {s['kill_rate']:.0%} of attempts")
    if s["landed_wait_p50"] is not None:
        print(f"  landed refine p50: {s['landed_wait_p50']} ms")
    print(f"\n  text changed by the refine     : {s['changed_by_refine']}"
          f"  ({s['substantive']} substantive, "
          f"{s['typographic_only']} punctuation only)")
    print(f"  text changed by the guard only : {s['changed_by_agreement']}"
          "   (not a refine — excluded below)")
    print("\n" + "=" * 72)
    print("Landed refines that changed the text — read these by hand.")
    print("An LLM judge was tried and failed here: scored twice with the")
    print("candidates swapped, 67% of its verdicts flipped with position.")
    print("=" * 72)
    shown = 0
    for r in rows:
        if not r["substantive"]:
            continue
        shown += 1
        flag = "  [guard also changed this]" if r["changed_by"] == "both" else ""
        print(f"\n#{r['uid']}  {r['source']}→{r['target']}  "
              f"{r['wait_ms']} ms{flag}")
        print(f"  heard   : {r['heard']}")
        print(f"  draft   : {r['draft']}")
        print(f"  refined : {r['final']}")
    if not shown:
        print("\n  (none — every landed refine agreed with the draft)")
    if show_unchanged:
        agreed = [r for r in rows
                  if "landed" in (r["outcome"] or "") and not r["changed_by"]]
        print(f"\n{len(agreed)} landed refines left the draft alone:")
        for r in agreed:
            print(f"  #{r['uid']}  {r['draft'][:70]}")


async def capture(args) -> int:
    src = Path(args.audio)
    # A .wav is already the server's native format only if it is 16 kHz mono
    # s16 — read_pcm says so loudly if not, so convert anything else first.
    if src.suffix.lower() != ".wav":
        src = replay.convert(src, Path(args.tmp) / "capture-in.wav")
    pcm = replay.read_pcm(src)
    secs = len(pcm) / 2 / replay.SAMPLE_RATE
    print(f"audio: {secs:.1f}s -> {args.url}  (pace {args.pace}x)")

    cfg = {"type": "config", "mode": args.mode, "stats": False,
           "draft_model": args.draft_model}
    if args.model:
        cfg["model"] = args.model

    events: list[dict] = []
    wall_start = time.time()
    t0 = time.monotonic()
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps(cfg))
        reader = asyncio.create_task(replay.collect(ws, events, t0, True))
        await replay.stream(ws, pcm, args.pace, t0)
        deadline = time.monotonic() + args.drain
        while time.monotonic() < deadline:
            started = sum(e["type"] == "segment_start" for e in events)
            ended = sum(e["type"] in ("translation_done", "discard", "error")
                        for e in events)
            if started and started == ended:
                break
            await asyncio.sleep(0.25)
        reader.cancel()
    # The server writes the trace record when the utterance completes, which
    # can trail the last websocket message. Give it a moment rather than
    # silently dropping the final rows.
    await asyncio.sleep(1.0)

    records = [r for r in read_trace(Path(args.trace))
               if r.get("t", 0) >= wall_start]
    rows = join_refines(events, records)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"wrote {len(rows)} rows to {args.json}")
    report(rows, args.show_unchanged)
    return 0


def read_trace(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", help="recording to replay through the live server")
    p.add_argument("--events", help="re-report a saved events json instead")
    p.add_argument("--trace", default="/tmp/allklaro-trace.jsonl")
    p.add_argument("--url", default="ws://127.0.0.1:8710/ws")
    p.add_argument("--mode", default="auto-de-en")
    p.add_argument("--model", default="")
    p.add_argument("--draft-model", default="qwen2.5:7b-instruct")
    p.add_argument("--pace", type=float, default=1.0)
    p.add_argument("--drain", type=float, default=120.0)
    p.add_argument("--json", help="write the joined rows here")
    p.add_argument("--tmp", default="/tmp")
    p.add_argument("--show-unchanged", action="store_true",
                   help="also list landed refines that agreed with the draft")
    args = p.parse_args()
    if args.events:
        events = json.loads(Path(args.events).read_text())
        report(join_refines(events, read_trace(Path(args.trace))),
               args.show_unchanged)
        return 0
    if not args.audio:
        p.error("need --audio (live capture) or --events (re-report)")
    return asyncio.run(capture(args))


if __name__ == "__main__":
    sys.exit(main())
