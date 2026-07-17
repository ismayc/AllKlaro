"""Compile a German noun-gender lexicon from a dict.cc EN->DE export.

Usage:  uv run python build_gender_lexicon.py /path/to/english-to-german-dictionary.zip

Writes ~/.cache/allklaro/de_noun_genders.tsv with lines "english<TAB>German
noun<TAB>gender". Only same-spelling pairs (loanwords/cognates, compared
ignoring case and hyphens) are kept: they are the words language models get
gender-wrong, and the restriction rules out false friends — English "gift"
must never surface the German word Gift (= poison).

dict.cc data is licensed for PRIVATE USE ONLY. The compiled lexicon stays in
~/.cache and must never be committed or redistributed; this script (code
only) is safe to share.
"""
import io
import re
import sys
import zipfile
from pathlib import Path

OUT_PATH = Path.home() / ".cache" / "allklaro" / "de_noun_genders.tsv"

# "Podcast {m} [tags]" -> word + its gender markers. Entries carrying several
# markers ("Margarita {m} {f}") are genuinely both genders -> unusable.
GERMAN_RE = re.compile(r"^([A-Za-zÄÖÜäöüß'\-]+)((?: \{(?:m|f|n|pl)\})+)(?:\s|$)")
GENDER_RE = re.compile(r"\{(m|f|n|pl)\}")
ENGLISH_RE = re.compile(r"^([A-Za-z'\-]+)(?:\s*[<\[]|\s*$)")


def norm(word: str) -> str:
    return word.lower().replace("-", "")


def compile_lexicon(lines) -> dict[str, tuple[str, str]]:
    """english -> (German noun, gender); ambiguous-gender words dropped."""
    found: dict[str, set[tuple[str, str]]] = {}
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or parts[2] != "noun":
            continue
        en_m = ENGLISH_RE.match(parts[0])
        de_m = GERMAN_RE.match(parts[1])
        if not en_m or not de_m:
            continue
        english, german = en_m.group(1), de_m.group(1)
        genders = set(GENDER_RE.findall(de_m.group(2)))
        if genders - {"m", "f", "n"}:
            continue                      # plural-only entries carry no article
        if len(genders) != 1:
            continue                      # both genders valid -> nothing to teach
        if norm(english) != norm(german):
            continue                      # not the same word -> false-friend risk
        found.setdefault(english.lower(), set()).add((german, genders.pop()))
    kept = {}
    for english, entries in found.items():
        genders = {g for _, g in entries}
        if len(genders) == 1:             # conflicting genders are unusable
            # Prefer the hyphenated/canonical spelling if variants exist.
            german = sorted((w for w, _ in entries), key=len, reverse=True)[0]
            kept[english] = (german, genders.pop())
    return kept


def read_export(path: Path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".txt"))
            with z.open(name) as f:
                yield from io.TextIOWrapper(f, encoding="utf-8", errors="replace")
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            yield from f


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    lexicon = compile_lexicon(read_export(Path(sys.argv[1])))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for english in sorted(lexicon):
            german, gender = lexicon[english]
            f.write(f"{english}\t{german}\t{gender}\n")
    print(f"{len(lexicon)} same-spelling noun genders -> {OUT_PATH}")


if __name__ == "__main__":
    main()
