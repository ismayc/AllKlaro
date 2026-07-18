"""Compile noun-gender lexicons from dictionary exports.

Usage:  uv run python build_gender_lexicon.py FILE...

Accepts, in any mix:
  - dict.cc tab exports (.zip or .txt), e.g. the EN->DE translation file
  - FreeDict TEI source tarballs (.src.tar.xz), e.g. eng-deu, deu-eng,
    eng-spa, spa-eng, deu-spa, spa-deu

Writes per-target lexicons to ~/.cache/allklaro/:
  de_noun_genders.tsv   english/spanish word -> German noun + gender
  es_noun_genders.tsv   english/german word  -> Spanish noun + gender

Rules that keep the data safe to inject into translation prompts:
  - Only same-spelling pairs are kept (compared ignoring case, hyphens, and
    accents, allowing one trailing -a/-o/-e: problem = problema). They are
    the words language models get gender-wrong, and the restriction rules
    out false friends — English "gift" must never surface Gift (= poison).
  - Words a dictionary lists with several genders (dict.cc "Margarita {m}
    {f}"), or whose gender differs across dictionaries, are dropped:
    both genders are in use, so there is nothing safe to teach.
  - Entries without a gender (eng-spa has none) borrow one from the other
    dictionaries' observations of the same target-language word — but only
    when every observation agrees.

Dictionary licenses differ (dict.cc is private-use only); the compiled
lexicons stay in ~/.cache and must never be committed to the repo.
"""
import io
import re
import sys
import tarfile
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path.home() / ".cache" / "allklaro"
TARGETS = {"de": "de_noun_genders.tsv", "es": "es_noun_genders.tsv"}
# Output-side maps: target-language noun -> gender, used to verify the
# model's *output* for impossible article/gender combinations.
OUTPUT_FILES = {"de": "de_output_genders.tsv", "es": "es_output_genders.tsv"}
TARGET_GENDERS = {"de": {"m", "f", "n"}, "es": {"m", "f"}}

LANG_CODES = {"eng": "en", "deu": "de", "spa": "es"}
# A single word, possibly hyphenated/apostrophed — no spaces or digits.
WORD_RE = re.compile(r"^[^\W\d_]+(?:[-'][^\W\d_]+)*$", re.UNICODE)

# dict.cc: "Podcast {m} [tags]" -> word + all gender markers.
DICTCC_GERMAN_RE = re.compile(
    r"^([A-Za-zÄÖÜäöüß'\-]+)((?: \{(?:m|f|n|pl)\})+)(?:\s|$)")
DICTCC_GENDER_RE = re.compile(r"\{(m|f|n|pl)\}")
DICTCC_ENGLISH_RE = re.compile(r"^([A-Za-z'\-]+)(?:\s*[<\[]|\s*$)")


def norm(word: str) -> str:
    """Spelling-comparison form: casefold, drop hyphens and accents."""
    decomposed = unicodedata.normalize("NFD", word.lower().replace("-", ""))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def loose_match(a: str, b: str) -> bool:
    """Same word across languages: exact, or one trailing -a/-o/-e added
    (problem/problema, map/mapa). Conservative on purpose."""
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    for x, y in ((na, nb), (nb, na)):
        if len(y) > 3 and y[:-1] == x and y[-1] in "aeo":
            return True
    return False


def norm_gender(raw: str) -> str | None:
    """'m' / 'masc' / 'fem' / 'neut' -> m/f/n; anything else -> None."""
    return raw[0] if raw and raw[0] in "mfn" else None


# ------------------------------------------------------------------- parsers
# Both parsers yield pairs: (lang_a, word_a, genders_a, lang_b, word_b,
# genders_b) where genders are frozensets (empty = the dictionary doesn't
# say; more than one = the dictionary explicitly allows several).


def parse_dictcc(lines, lang_a="en", lang_b="de"):
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or parts[2] != "noun":
            continue
        en_m = DICTCC_ENGLISH_RE.match(parts[0])
        de_m = DICTCC_GERMAN_RE.match(parts[1])
        if not en_m or not de_m:
            continue
        genders = {norm_gender(g)
                   for g in DICTCC_GENDER_RE.findall(de_m.group(2))}
        if None in genders:
            continue                   # {pl} entries carry no article
        yield (lang_a, en_m.group(1), frozenset(),
               lang_b, de_m.group(1), frozenset(genders))


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _genders_in(el):
    return frozenset(g for g in (norm_gender((gen.text or "").strip())
                                 for gen in el.iter() if _local(gen.tag) == "gen")
                     if g)


def parse_tei(fileobj, lang_a, lang_b):
    """FreeDict TEI, streaming. Handles the three gender placements seen in
    the wild: on the headword (deu-eng), on each translation cit (eng-deu),
    and at sense level for single-translation senses (spa-deu)."""
    for _, entry in ET.iterparse(fileobj):
        if _local(entry.tag) != "entry":
            continue
        orth, head_genders, noun = None, frozenset(), False
        translations = []              # (word, genders)
        for child in entry:
            tag = _local(child.tag)
            if tag == "form" and orth is None:
                for el in child.iter():
                    if _local(el.tag) == "orth" and orth is None:
                        orth = (el.text or "").strip()
                head_genders = _genders_in(child)
                noun = noun or any((el.text or "").strip().startswith("n")
                                   for el in child.iter()
                                   if _local(el.tag) == "pos")
            elif tag == "gramGrp":
                head_genders = head_genders | _genders_in(child)
                noun = noun or any((el.text or "").strip().startswith("n")
                                   for el in child.iter()
                                   if _local(el.tag) == "pos")
            elif tag == "sense":
                cits = [c for c in child
                        if _local(c.tag) == "cit" and c.get("type") == "trans"]
                sense_genders = frozenset()
                for el in child:       # sense-level gramGrp (spa-deu style)
                    if _local(el.tag) == "gramGrp":
                        sense_genders = sense_genders | _genders_in(el)
                quotes = []
                for cit in cits:
                    cit_genders = _genders_in(cit)
                    for q in cit:
                        if _local(q.tag) == "quote":
                            quotes.append(((q.text or "").strip(), cit_genders))
                for word, cit_genders in quotes:
                    genders = cit_genders
                    if not genders and len(quotes) == 1:
                        genders = sense_genders
                    translations.append((word, genders))
        if orth and (noun or head_genders):
            for word, genders in translations:
                yield (lang_a, orth, head_genders, lang_b, word, genders)
        entry.clear()                  # keep iterparse memory flat


# ------------------------------------------------------------------ building


def build_lexicons(pairs):
    """pairs -> {target: {key: (word, gender)}} with gender pooling."""
    pairs = list(pairs)
    pools = defaultdict(set)           # (lang, word.lower()) -> genders seen
    for la, wa, ga, lb, wb, gb in pairs:
        pools[(la, wa.lower())] |= ga
        pools[(lb, wb.lower())] |= gb

    found = {t: defaultdict(set) for t in TARGETS}
    for la, wa, ga, lb, wb, gb in pairs:
        for (tl, tw), (kl, kw) in (((la, wa), (lb, wb)),
                                   ((lb, wb), (la, wa))):
            if tl not in TARGETS or kl == tl:
                continue
            if not WORD_RE.match(tw) or not WORD_RE.match(kw):
                continue
            if not loose_match(tw, kw):
                continue
            genders = pools[(tl, tw.lower())] & TARGET_GENDERS[tl]
            if len(genders) != 1:
                continue               # unknown or genuinely both -> unusable
            found[tl][kw.lower()].add((tw, next(iter(genders))))

    lexicons = {}
    for target, keys in found.items():
        kept = {}
        for key, entries in keys.items():
            genders = {g for _, g in entries}
            if len(genders) == 1:      # spelling variants may differ; gender not
                word = sorted((w for w, _ in entries), key=len, reverse=True)[0]
                kept[key] = (word, genders.pop())
        lexicons[target] = kept
    return lexicons


def build_output_maps(pairs):
    """Every unambiguous target-language noun gender the dictionaries know:
    {target: {word.lower(): (display form, gender)}}. Unlike the source-keyed
    lexicons this needs no same-spelling pair — it describes the output side
    directly, so it also covers nouns that never appear in the source."""
    seen = {}
    for la, wa, ga, lb, wb, gb in pairs:
        for lang, word, genders in ((la, wa, ga), (lb, wb, gb)):
            if lang not in TARGETS or not genders or not WORD_RE.match(word):
                continue
            entry = seen.setdefault((lang, word.lower()),
                                    {"display": word, "genders": set()})
            entry["genders"] |= genders
            if word[:1].isupper() and not entry["display"][:1].isupper():
                entry["display"] = word    # prefer the capitalized form
    maps = {t: {} for t in TARGETS}
    for (lang, lower), entry in seen.items():
        genders = entry["genders"]
        if len(genders) == 1 and genders <= TARGET_GENDERS[lang]:
            maps[lang][lower] = (entry["display"], genders.pop())
    return maps


# -------------------------------------------------------------------- inputs


def detect_pair(name: str) -> tuple[str, str] | None:
    m = re.search(r"(eng|deu|spa)-(eng|deu|spa)", name)
    if not m or m.group(1) == m.group(2):
        return None
    return LANG_CODES[m.group(1)], LANG_CODES[m.group(2)]


def read_input(path: Path):
    """Yield pairs from one input file of either format."""
    if path.name.endswith(".tar.xz"):
        with tarfile.open(path, "r:xz") as tar:
            member = next(m for m in tar.getmembers()
                          if m.name.endswith(".tei"))
            langs = detect_pair(member.name)
            if not langs:
                print(f"skipping {path.name}: unknown language pair")
                return
            yield from parse_tei(tar.extractfile(member), *langs)
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".txt"))
            with z.open(name) as f:
                yield from parse_dictcc(
                    io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            yield from parse_dictcc(f)


def write_map(mapping, filename, label):
    if not mapping:                    # don't clobber a previous build
        print(f"{label}: no entries from these inputs, file untouched")
        return
    with open(OUT_DIR / filename, "w", encoding="utf-8") as f:
        for key in sorted(mapping):
            word, gender = mapping[key]
            f.write(f"{key}\t{word}\t{gender}\n")
    print(f"{label}: {len(mapping)} noun genders -> {OUT_DIR / filename}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    pairs = [p for arg in sys.argv[1:] for p in read_input(Path(arg))]
    lexicons = build_lexicons(pairs)
    output_maps = build_output_maps(pairs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for target, filename in TARGETS.items():
        write_map(lexicons.get(target, {}), filename, f"{target} source-keyed")
    for target, filename in OUTPUT_FILES.items():
        write_map(output_maps.get(target, {}), filename, f"{target} output-side")


if __name__ == "__main__":
    main()
