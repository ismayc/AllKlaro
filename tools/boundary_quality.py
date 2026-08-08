#!/usr/bin/env python3
"""Are the 5 s cap's boundaries worse than real pauses? Measure before cutting.

    uv run python tools/dump_transcripts.py --audio hour.wav --out hour.jsonl
    uv run python tools/boundary_quality.py --dump hour.jsonl

The idea this was built to test: `SOFT_MAX_SEC` cuts an utterance short, and
German is verb-final, so the cut lands before the verb and the card cannot be
translated. The fix would be to close on syntactic completion instead of on
elapsed time.

The premise did not survive the recording. Two things it got wrong:

1. The cap does not cut on a stopwatch. `split_at` holds the *last micro-pause*
   seen, and when the cap fires the VAD emits up to that pause (server.py's
   VadSession). The app already cuts at the speaker's own hesitation.
2. The VAD utterance is not the card. server.py merges an unfinished utterance
   into its continuation before anything is translated, so broken-ness measured
   on raw utterances counts fragments the pipeline already repairs.

Measured over 54 minutes of real conversation, 703 utterances, 522 of them
German. The cap fires on 76% of all splits — but its boundaries are no worse
than the ones a real 700 ms pause produces (--metric broken, German only):

                          soft_max   pause     p
      raw utterances       50.9%     47.9%    0.59
      merged cards         41.1%     38.7%    0.70

The ceiling is what settles it, with no significance test needed: the boundary
the idea wants to wait for is itself broken 38.7% of the time. Cards are
fragments because spontaneous speech is fragmentary, not because of the cap.
Closing on syntactic completion has ~2.5 pp of headroom and costs latency,
which is the one thing this pipeline cannot spare. Not built.

What the run does establish is that the existing merge earns its place: it
takes broken cards from 50.4% to 40.7%.

Two things to keep in mind before trusting a number out of here:

- The `pause` arm is small — 96 German utterances, 75 once merged — so the
  minimum detectable difference is 15.8 pp on raw utterances and 18.0 pp on
  cards. This rules out a large effect, not a small one, and `report()` prints
  the figure next to every comparison for that reason.
- The default `--metric broken` is the detector that was validated: judge or
  casing rule together reproduced 51 hand-labelled utterances at 86%
  agreement, precision 0.84, recall 0.93. The judge alone recalls only 0.75,
  because it reads the end of a chunk and not the start.
- That is also why `--merge lowercase` cannot be scored with it: the casing
  rule is both the intervention and half the detector, so the variant would
  be measured against itself. Pass `--metric ends` for that comparison and
  the tool insists on it. Under `ends` the variant does not improve endings
  (25.3% -> 29.2%, p=0.24) — merging moves a card's end to its continuation's
  end. It fixes starts, which `ends` cannot see and `broken` cannot fairly
  score, so that one stays a judgement call rather than a number.
"""
import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OLLAMA = "http://127.0.0.1:11434/api/generate"

# gemma3:12b cannot do this task — precision 0.30 against hand labels, it
# calls "It's because of the flood." verb-missing. qwen2.5:14b-instruct
# reproduces 51 hand-labelled utterances at 86% agreement (precision 0.84,
# recall 0.93) and matched qwen2.5:32b-instruct within noise at a quarter of
# the wall clock.
DEFAULT_MODEL = "qwen2.5:14b-instruct"

PROMPT = """You are a German syntax annotator. You get ONE utterance from a \
live speech transcript of a real conversation.

Decide whether it breaks off before the verb of its final clause.

German word order matters here:
- In a subordinate clause (weil, dass, wenn, ob, obwohl, damit, a relative \
pronoun ...) the finite verb comes LAST.
- In a main clause the finite verb is second, but the participle, the \
infinitive, or a separable prefix comes LAST.
Either of those trailing verbal elements counts as "the verb" below.

Answer with exactly one label:
- COMPLETE: every clause that was started has its verb. Also use this when \
no clause was started at all (an interjection, a single word, a bare noun \
phrase) -- then nothing is missing a verb.
- MISSING_VERB: a clause has been started (there is a subject, an object, or \
a subordinating conjunction) but its verb has NOT been said yet.
- MISSING_OTHER: it breaks off, but the verb is already there; what is \
missing is a noun, an object, or a complement.

Rules that matter more than they look:
- Ignore the final punctuation mark. This is an automatic transcript and it \
puts a period at the end of every chunk, including chunks that stop in the \
middle of a sentence. Judge the words, not the period.
- A trailing "..." does NOT by itself mean the verb is missing. Check \
whether a verbless clause is really in progress.
- Real conversation is mostly COMPLETE, and colloquial or elliptical German \
is still COMPLETE. Only answer MISSING_VERB if you can name the verb that \
is absent. If you cannot name it, the answer is not MISSING_VERB.

Examples:
"Ich habe das Buch gestern gelesen." -> COMPLETE (gelesen is there)
"Ich weiss nicht, ob er heute kommt." -> COMPLETE (kommt closes the ob-clause)
"Das macht dann auch keinen Sinn." -> COMPLETE (colloquial but whole)
"Ja, genau." -> COMPLETE (no clause was started)
"Er hat gesagt, dass wir morgen..." -> MISSING_VERB (dass-clause has no verb)
"Ein kleiner Ballon, der da irgendwie..." -> MISSING_VERB (relative clause \
has no verb)
"Ich habe das gestern in der Stadt..." -> MISSING_VERB (the participle after \
habe never comes)
"Ich kaufe mir morgen einen neuen..." -> MISSING_OTHER (kaufe is there, the \
noun is missing)
"Und dann gab es..." -> MISSING_OTHER (gab is there, the object is missing)

Utterance:
{text}

Reply with JSON only: {{"label": "...", "missing": "<the absent verb, or \
empty>"}}"""

LABELS = ("COMPLETE", "MISSING_VERB", "MISSING_OTHER")
BROKEN_LABELS = frozenset(("MISSING_VERB", "MISSING_OTHER"))


def starts_lowercase(text: str) -> bool:
    """True when a chunk opens mid-sentence, on casing alone.

    German capitalises every noun and every sentence start, so a chunk that
    opens with a lowercase letter did not start a sentence. Free, no model
    call, and it agreed with hand labels at precision 0.82.

    It has one blind spot worth knowing: Whisper occasionally emits a chunk
    with no casing at all, and then a real sentence start looks like a
    continuation. That was 2.3% of German cards over the hour.
    """
    head = text.lstrip(".,!? ")
    return bool(head) and head[0].isalpha() and head[0].islower()


def merge_cards(rows, lowercase_continues=False):
    """Replay server.py's fragment merge over dumped utterances.

    The server joins an utterance to the previous one when the previous did
    not finish and this one resumed within MERGE_GAP_SEC. That runs before
    translation, so the card -- not the VAD utterance -- is what a listener
    reads, and it is the only fair unit to score.

    `lowercase_continues` adds the sized-but-unbuilt variant: merge when THIS
    chunk opens lowercase, whatever punctuation the previous one ended with.
    Punctuation-only merging cannot see those, because the evidence sits on
    the current chunk and the rule only looks at the previous one.
    """
    import server as srv

    cards, prev = [], None
    for r in rows:
        text = (r.get("cleaned") or "").strip()
        if not text:
            continue
        source = r.get("detected_language")
        start = r["t_end"] - r["dur_sec"]
        gap = (start - prev["t_end"]) if prev else 99.0
        if (prev and prev["source"] == source
                and gap < srv.MERGE_GAP_SEC
                and len(prev["text"]) < srv.MERGE_MAX_CHARS
                and (not srv.looks_finished(prev["text"])
                     or (lowercase_continues and starts_lowercase(text)))):
            prev["text"] += " " + text
            prev["t_end"] = r["t_end"]
            prev["parts"] += 1
            # A merged card inherits the split that finally ended it.
            prev["split"] = r["split"]
            continue
        if prev:
            cards.append(prev)
        prev = {"i": len(cards) + 1, "text": text, "source": source,
                "split": r["split"], "t_end": r["t_end"], "parts": 1}
    if prev:
        cards.append(prev)
    return cards


def judge(text, model=DEFAULT_MODEL, host=OLLAMA, timeout=180):
    """Ask the local model whether an utterance breaks off before its verb."""
    body = json.dumps({
        "model": model,
        "prompt": PROMPT.format(text=text),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 60},
    }).encode()
    req = urllib.request.Request(host, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)["response"]
    try:
        got = json.loads(out)
        label = got.get("label", "")
    except (json.JSONDecodeError, AttributeError):
        m = re.search("|".join(LABELS), out or "")
        label = m.group(0) if m else ""
    return label if label in LABELS else "UNPARSED"


def wilson(k, n):
    """Wilson score interval. Normal approximation is not safe at these n."""
    if not n:
        return 0.0, (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, (100 * (centre - half), 100 * (centre + half))


def two_proportion_p(k1, n1, k2, n2):
    """Two-sided p for a difference in proportions, pooled."""
    if not n1 or not n2:
        return 1.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def is_broken(card, metric):
    """`ends` uses the judge alone; `broken` adds the casing rule.

    Keep them apart. The casing rule is half of `broken` AND the whole of the
    `lowercase` merge variant, so scoring that variant with `broken` measures
    the intervention against itself.
    """
    ends = card["label"] in BROKEN_LABELS
    if metric == "ends":
        return ends
    return ends or starts_lowercase(card["text"])


def report(cards, metric):
    arms = {}
    for c in cards:
        arms.setdefault(c["split"], []).append(c)
    k = sum(is_broken(c, metric) for c in cards)
    pct, ci = wilson(k, len(cards))
    print(f"\n  {len(cards)} cards, metric={metric}")
    print(f"  broken overall {k}/{len(cards)} = {pct:.1f}% "
          f"CI [{ci[0]:.1f}, {ci[1]:.1f}]")
    for split, rs in sorted(arms.items()):
        kk = sum(is_broken(c, metric) for c in rs)
        p, c = wilson(kk, len(rs))
        print(f"    {split:9} {kk:4}/{len(rs):4} = {p:5.1f}%  "
              f"CI [{c[0]:5.1f}, {c[1]:5.1f}]")
    a, b = arms.get("soft_max", []), arms.get("pause", [])
    if a and b:
        ka = sum(is_broken(c, metric) for c in a)
        kb = sum(is_broken(c, metric) for c in b)
        pa, pb = 100 * ka / len(a), 100 * kb / len(b)
        p = two_proportion_p(ka, len(a), kb, len(b))
        print(f"    cap minus pause {pa - pb:+.1f} pp   p={p:.3f}")
        # 80% power, alpha .05, worst case p=.5 -- the pause arm is small and
        # a null here is easy to over-read without this number next to it.
        mde = 100 * (1.96 + 0.84) * math.sqrt(
            0.25 * (1 / len(a) + 1 / len(b)))
        print(f"    smallest difference this many cards could detect: "
              f"{mde:.1f} pp")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True,
                   help="output of tools/dump_transcripts.py")
    p.add_argument("--merge", default="punctuation",
                   choices=("none", "punctuation", "lowercase"),
                   help="none: score raw VAD utterances. punctuation: the "
                        "rule server.py ships. lowercase: the sized variant.")
    p.add_argument("--metric", default="broken", choices=("ends", "broken"),
                   help="broken: judge or lowercase start, the detector that "
                        "was validated against hand labels. ends: judge only "
                        "-- required for --merge lowercase, which the broken "
                        "metric would be scoring against itself.")
    p.add_argument("--language", default="de",
                   help="only score this detected language; '' for all")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=OLLAMA)
    p.add_argument("--out", default="", help="write per-card labels here")
    a = p.parse_args(argv)

    rows = [json.loads(l) for l in open(a.dump)]
    if a.merge == "none":
        cards = [{"i": r["i"], "text": (r.get("cleaned") or "").strip(),
                  "split": r["split"], "source": r.get("detected_language"),
                  "parts": 1}
                 for r in rows if (r.get("cleaned") or "").strip()]
    else:
        cards = [dict(c, text=c["text"])
                 for c in merge_cards(rows, a.merge == "lowercase")]
    if a.language:
        cards = [c for c in cards if c["source"] == a.language]

    if a.merge == "lowercase" and a.metric == "broken":
        print("  refusing: --metric broken contains the casing rule that "
              "--merge lowercase acts on, so the comparison is circular.")
        return 2

    merged = sum(c["parts"] > 1 for c in cards)
    print(f"{len(rows)} utterances -> {len(cards)} cards "
          f"({merged} merged from 2+) via {a.merge} merge")

    for n, c in enumerate(cards, 1):
        c["label"] = judge(c["text"], a.model, a.host)
        print(f"{n:4}/{len(cards)} {c['split']:9} {c['label']:14} "
              f"{c['text'][-64:]}", flush=True)

    unparsed = sum(c["label"] == "UNPARSED" for c in cards)
    if unparsed:
        print(f"\n  {unparsed} cards the model did not label")
    report(cards, a.metric)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"model": a.model, "dump": a.dump, "merge": a.merge,
                       "cards": cards}, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
