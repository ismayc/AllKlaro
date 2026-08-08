#!/usr/bin/env python3
"""Find what Whisper writes when someone speaks dialect, so the lexicon can
key on that instead of on spellings it never produces.

    uv run python tools/dialect_mishearings.py --whisper full.json \\
        --transcript other-system.md

`dialects.txt` keys on `ick`, `kiek`, `wat`, `det`. Measured over 2277 German
word tokens of the real Berlin recording, **none of them ever appear**:
Whisper decodes toward standard orthography, so every unambiguous entry is
dormant on the audio path. The dialect is not absent from the audio, it is
absent from the transcript — it arrives as a *wrong standard word*.

## What this found, and what it did not

**Phonetic search alone does not work here, in either direction**, and that is
the main result rather than a caveat.

Filtering to words absent from the 368k-entry Wiktionary lexicon (4.4% of the
hour) yields five candidates, and all five are the *other language*: this is a
bilingual conversation, so `what`~`wat`, `next`~`net`, `hot`~`hoscht` are
English words being matched against German dialect forms. Zero real hits.

Dropping that filter (`--all-words`) is worse: `die`~`dit`, `ist`~`nischt`,
`auch`~`aach` — ordinary German, dozens of times each, drowning anything real.

The one genuine mis-hearing in the recording was found by **reading an
utterance whose translation was nonsense**, and confirmed because a second
decoder made the same mistake and the next sentence used the standard word:

    heard by both systems : "Ich kicke ja immer nicht.
                             Meistens *guckt* der Arvid nach."
    actually              : ick kieke ja immer nich  (kieken = gucken)
    both translated it as : kicking a ball

So the productive signal is **semantic implausibility**, not phonetic
distance — the mis-hearings that matter land on real words, which is exactly
why nothing downstream catches them. This tool is kept for `--show-context`
and as the record of what phonetic search is worth here, which is little.

**This proposes candidates; it does not edit the lexicon.** A wrong mapping is
worse than a missing one — it creates an error in ordinary speech rather than
failing to fix one — so every candidate needs reading before it is added.
"""
import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

WIKT = Path.home() / ".cache/allklaro/wikt_de.sqlite"
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Collapses the distinctions German ASR routinely gets wrong, so that a
# mis-heard "kick" and the intended "kiek" land on the same key.
_FOLD = [("sch", "s"), ("ck", "k"), ("chs", "ks"), ("ch", "h"), ("ph", "f"),
         ("qu", "kw"), ("tz", "z"), ("ss", "s"), ("ß", "s"), ("th", "t"),
         ("ie", "i"), ("ei", "ai"), ("eu", "oi"), ("äu", "oi"),
         ("ä", "e"), ("ö", "o"), ("ü", "u"), ("v", "f"), ("y", "i")]


def fold(word: str) -> str:
    w = word.lower()
    for a, b in _FOLD:
        w = w.replace(a, b)
    out = []                                   # collapse doubled letters
    for ch in w:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def german_words() -> set[str]:
    if not WIKT.exists():
        sys.exit(f"{WIKT} missing — build it with build_wiktionary_lexicon.py")
    con = sqlite3.connect(f"file:{WIKT}?mode=ro", uri=True)
    return {r[0] for r in con.execute("select word_lc from entries")}


def dialect_forms(path: Path) -> dict[str, str]:
    """token -> gloss, unambiguous German entries only (the dormant ones)."""
    import server as srv

    lex = srv.load_dialects().get("de", {})
    return {t: g for t, (g, amb, _f) in lex.items() if not amb}


def words_of(text: str) -> list[str]:
    return WORD_RE.findall(text)


def candidates(whisper_text: str, known: set[str], forms: dict[str, str],
               min_len=3, cutoff=0.80, skip_known=True):
    """Words that sound like a dialect form, ranked by how often they were
    heard.

    `skip_known=False` is the setting that actually works, and finding that
    out cost a run. The first version only considered words absent from
    Wiktionary, on the theory that a mis-hearing is usually not a real word.
    The opposite is true where it matters: the mis-hearings that change
    meaning are the ones that land on a REAL German word, because those are
    the ones no downstream check can catch. On the real recording both
    decoders wrote "gekickt" and "kicke" for Berlinerisch kieken (gucken) —
    kicken being an ordinary German verb, the unknown-word filter skipped
    them, and they were found by hand instead.

    Keeping known words in costs precision, which is why the cross-check
    against a second decoder matters more in this mode than the score does.
    """
    counts = Counter(w.lower() for w in words_of(whisper_text)
                     if len(w) >= min_len
                     and (not skip_known or w.lower() not in known))
    folded = {t: fold(t) for t in forms}
    out = []
    for word, n in counts.most_common():
        fw = fold(word)
        best, score = None, 0.0
        for term, ft in folded.items():
            s = 1.0 if ft == fw else SequenceMatcher(None, fw, ft).ratio()
            if s > score:
                best, score = term, s
        if best and score >= cutoff:
            out.append({"heard": word, "n": n, "dialect": best,
                        "gloss": forms[best], "score": round(score, 3)})
    return out


def contexts(text: str, word: str, width=48, limit=2):
    out = []
    for m in re.finditer(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
        out.append(text[max(0, m.start() - width):m.end() + width]
                   .replace("\n", " ").strip())
        if len(out) >= limit:
            break
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--whisper", required=True,
                   help="json from a whisper run, or a plain text file")
    p.add_argument("--transcript", help="another system's transcript, to cross-check")
    p.add_argument("--dialects", default="dialects.txt")
    # Deliberately loose. This generates candidates for a human to read, and
    # the known real case is marginal: Whisper's "kicke" against the lexicon's
    # "kiek" folds to kike/kik and scores 0.857, while the same utterance's
    # "kipp" scores 0.667. A tight cutoff would miss the very case that
    # started this.
    p.add_argument("--cutoff", type=float, default=0.80)
    p.add_argument("--show-context", action="store_true")
    p.add_argument("--all-words", action="store_true",
                   help="also consider real German words — where the "
                        "meaning-changing mis-hearings actually live")
    args = p.parse_args()

    raw = Path(args.whisper).read_text()
    text = json.loads(raw)["text"] if args.whisper.endswith(".json") else raw
    known = german_words()
    forms = dialect_forms(Path(args.dialects))
    toks = words_of(text)
    unknown = [w for w in toks if w.lower() not in known]
    print(f"{len(toks)} word tokens, {len(unknown)} not in Wiktionary "
          f"({len(unknown)/max(1,len(toks)):.1%}), {len(forms)} unambiguous "
          f"dialect forms to match against\n")

    hits = candidates(text, known, forms, cutoff=args.cutoff,
                      skip_known=not args.all_words)
    if not hits:
        print("No unknown word resembled a dialect form at this cutoff.")
    for h in hits:
        print(f"  heard {h['heard']!r} x{h['n']}  ~  {h['dialect']!r} "
              f"= {h['gloss']}   (similarity {h['score']})")
        if args.show_context:
            for c in contexts(text, h["heard"]):
                print(f"      …{c}…")

    if args.transcript:
        other = Path(args.transcript).read_text()
        # Only the source-language lines; a bilingual export carries both.
        de = "\n".join(ln for ln in other.splitlines()
                       if ln.startswith("**de:**"))
        oth = {w.lower() for w in words_of(de or other)}
        both = [h for h in hits if h["heard"] in oth]
        print(f"\n  {len(both)} of {len(hits)} were heard the SAME way by the "
              f"other system — those are the audio, not one decoder's quirk:")
        for h in both:
            print(f"    {h['heard']!r} ~ {h['dialect']!r}")
        only_other = sorted(w for w in oth
                            if w not in known and len(w) > 3)[:40]
        print(f"\n  words only the other system invented ({len(only_other)} "
              f"shown): {', '.join(only_other[:20])}")
    print("\n  Candidates only. A wrong mapping is worse than a missing one —")
    print("  it breaks ordinary speech instead of failing to fix dialect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
