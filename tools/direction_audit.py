#!/usr/bin/env python3
"""Find the cards a replay translated in the wrong direction.

In an auto mode the direction of every card comes from the language Whisper
detects in the audio. When that call is wrong the card is labeled with one
language while its text is in the other, the translator is asked to turn (for
example) German into German, and the card shows the same sentence twice.

This reads a `replay.py --out` event stream, where the text is, and reports:

  * echo cards: the finished translation is the heard text again;
  * mislabeled cards: the heard text reads as the pair's other language
    (the app's own typed-text detector, run on what Whisper wrote).

    uv run python tools/direction_audit.py events.jsonl
    uv run python tools/direction_audit.py events.jsonl --show 40
    uv run python tools/direction_audit.py export.md      # a live session
"""
import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import detect_language_scored, normalize_text  # noqa: E402

ECHO_RATIO = 0.8          # translation this close to the source is an echo
SURE = 0.9                # text-detector confidence that counts as a verdict


def load_cards(path: Path) -> list[dict]:
    """One record per card that was still on screen at the end of the run."""
    cards: dict[int, dict] = {}
    for line in path.open():
        e = json.loads(line)
        kind = e.get("type")
        if kind == "final":
            if e.get("replaces") is not None:
                cards.pop(e["replaces"], None)
            cards[e["id"]] = {"id": e["id"], "at": e.get("_at"),
                              "text": e["text"], "source": e["source"],
                              "target": e["targets"][0], "translation": "",
                              "merged": e.get("replaces") is not None}
        elif kind == "translation_delta" and e["id"] in cards:
            if e["target"] == cards[e["id"]]["target"]:
                cards[e["id"]]["translation"] += e["text"]
        elif kind == "translation_revised" and e["id"] in cards:
            text = (e.get("texts") or {}).get(cards[e["id"]]["target"])
            if text:
                cards[e["id"]]["translation"] = text
    return [cards[k] for k in sorted(cards)]


EXPORT_RE = re.compile(
    r"^\*\*\[(?P<at>[^\]]+)\](?: [^(]*)? \((?P<src>[A-Z]{2})\):\*\* (?P<text>.*)\n"
    r"→ \*\*\((?P<tgt>[A-Z]{2})\):\*\* (?P<tr>.*)$", re.M)


def load_export(path: Path) -> list[dict]:
    """The same cards from the app's own Export file: what a live session
    actually showed, which a replay can only approximate."""
    return [{"id": i, "at": m["at"], "text": m["text"],
             "source": m["src"].lower(), "target": m["tgt"].lower(),
             "translation": m["tr"], "merged": False}
            for i, m in enumerate(EXPORT_RE.finditer(path.read_text()), 1)]


def audit(cards: list[dict]) -> list[dict]:
    for c in cards:
        a, b = normalize_text(c["text"]), normalize_text(c["translation"])
        c["echo"] = bool(a and b) and (
            SequenceMatcher(None, a, b).ratio() >= ECHO_RATIO)
        lang, conf = detect_language_scored(c["text"],
                                            (c["source"], c["target"]))
        c["text_lang"], c["text_conf"] = lang, round(conf, 2)
        c["mislabeled"] = lang != c["source"] and conf >= SURE
        c["words"] = len(c["text"].split())
    return cards


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("events")
    p.add_argument("--show", type=int, default=15,
                   help="how many examples of each kind to print")
    p.add_argument("--json", help="write the audited cards to this file")
    args = p.parse_args()
    path = Path(args.events)
    cards = audit(load_export(path) if path.suffix == ".md"
                  else load_cards(path))
    n = len(cards)
    if not n:
        sys.exit("no cards in that event stream")
    echo = [c for c in cards if c["echo"]]
    wrong = [c for c in cards if c["mislabeled"]]
    both = [c for c in cards if c["echo"] and c["mislabeled"]]
    by_src = {s: sum(c["source"] == s for c in cards)
              for s in sorted({c["source"] for c in cards})}
    print(f"{n} cards on screen at the end  (labeled: {by_src})")
    print(f"  echo (translation repeats the heard text): {len(echo)} "
          f"({len(echo) / n:.1%})")
    print(f"  mislabeled (text reads as the other language): {len(wrong)} "
          f"({len(wrong) / n:.1%})")
    print(f"  both: {len(both)}")
    for s in by_src:
        w = [c for c in wrong if c["source"] == s]
        print(f"    labeled {s} but reads as the other: {len(w)}")
    short = [c for c in wrong if c["words"] <= 4]
    print(f"  mislabeled cards of 4 words or fewer: {len(short)}")
    for title, rows in (("ECHO", echo), ("MISLABELED, no echo",
                                         [c for c in wrong if not c["echo"]])):
        print(f"\n{title} (first {args.show})")
        for c in rows[:args.show]:
            print(f"  #{c['id']} @{c['at']}s [{c['source']}->{c['target']}] "
                  f"text reads {c['text_lang']} {c['text_conf']}")
            print(f"      heard: {c['text'][:110]}")
            print(f"      shown: {c['translation'][:110]}")
    if args.json:
        Path(args.json).write_text(json.dumps(cards, ensure_ascii=False,
                                              indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
