"""Compile a German noun-form analysis table from the german-nouns CSV
(Wiktionary-derived, https://github.com/gambolputty/german-nouns, CC BY-SA).

Usage:  uv run python build_noun_forms.py [path/to/nouns.csv | URL]

Downloads the CSV when no argument is given. Writes
~/.cache/allklaro/de_noun_forms.tsv with lines

    form<TAB>lemma<TAB>gender<TAB>codes

where codes is a comma list of <case><number> readings this exact spelling
can have: case n/g/d/a (nominativ/genitiv/dativ/akkusativ), number s/p.
E.g. "lehrer  Lehrer  m  ns,ds,as,np,gp,ap" — the missing dp says the
dative plural is "Lehrern", so "mit den Lehrer" is provably wrong.

The declension guard uses this to intersect determiner readings with noun
readings: an NP with no consistent (case, number, gender) reading is wrong
in every interpretation. Like the gender lexicons, the compiled table
stays in ~/.cache."""
import csv
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

CSV_URL = ("https://raw.githubusercontent.com/gambolputty/german-nouns/"
           "main/german_nouns/nouns.csv")
OUT_PATH = Path.home() / ".cache" / "allklaro" / "de_noun_forms.tsv"

CASES = {"nominativ": "n", "genitiv": "g", "dativ": "d", "akkusativ": "a"}
NUMBERS = {"singular": "s", "plural": "p"}
WORD_RE = re.compile(r"^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß-]*$")


def compile_forms(rows):
    """CSV rows -> {(form, lemma, gender): set of code strings}."""
    header = next(rows)
    idx = {name: i for i, name in enumerate(header)}
    slots = []                        # (code, [column indices])
    for case, c in CASES.items():
        for number, n in NUMBERS.items():
            cols = [i for name, i in idx.items()
                    if name.startswith(f"{case} {number}")]
            slots.append((c + n, cols))
    gender_cols = [i for name, i in idx.items() if name.startswith("genus")]

    analyses = defaultdict(set)
    for row in rows:
        if "Substantiv" not in row[idx["pos"]]:
            continue
        lemma = row[idx["lemma"]]
        if not WORD_RE.match(lemma):
            continue                  # affixes, multiword, lowercase entries
        genders = {row[i] for i in gender_cols
                   if i < len(row) and row[i] in ("m", "f", "n")}
        if not genders:
            continue
        for code, cols in slots:
            for i in cols:
                form = row[i].strip() if i < len(row) else ""
                if WORD_RE.match(form):
                    for g in genders:
                        analyses[(form.lower(), lemma, g)].add(code)
    return analyses


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else CSV_URL
    if re.match(r"https?://", src):
        print(f"Downloading {src} ...")
        with urllib.request.urlopen(src) as r:
            text = r.read().decode("utf-8")
        rows = csv.reader(text.splitlines())
    else:
        rows = csv.reader(open(src, encoding="utf-8"))
    analyses = compile_forms(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for (form, lemma, gender) in sorted(analyses):
            codes = ",".join(sorted(analyses[(form, lemma, gender)]))
            f.write(f"{form}\t{lemma}\t{gender}\t{codes}\n")
    forms = len({k[0] for k in analyses})
    print(f"{forms} forms / {len(analyses)} analyses -> {OUT_PATH}")


if __name__ == "__main__":
    main()
