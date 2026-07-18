"""Compile tap-to-look-up dictionaries from Wiktionary via kaikki.org.

Usage:  uv run python build_wiktionary_lexicon.py [de] [es] [en]
        (no arguments = de es)

For each requested language this streams kaikki.org's machine-readable
extract of the English Wiktionary (CC BY-SA — unlike dict.cc, freely
redistributable with attribution) and compiles it into an SQLite database
that the server's /api/lookup endpoint reads:

    ~/.cache/allklaro/wikt_de.sqlite   etc.

Each row is one Wiktionary entry: word, part of speech, gender(s), IPA,
plural(s), up to five gloss senses (JSON array), and — for inflected forms
like "ging" — the lemma it belongs to, so a lookup can chain to the base
word ("first/third-person singular preterite of gehen" -> gehen).

The download streams straight through gzip into the parser; nothing but the
finished database touches disk. Sizes as of mid-2026: German ~95 MB
compressed, Spanish similar, English several times larger.
"""
import gzip
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

LANGS = {"de": "German", "es": "Spanish", "en": "English"}
URL_TEMPLATE = ("https://kaikki.org/dictionary/{name}/"
                "kaikki.org-dictionary-{name}.jsonl.gz")
OUT_DIR = Path.home() / ".cache" / "allklaro"

GENDER_TAGS = {"masculine": "m", "feminine": "f", "neuter": "n"}
# A form row tagged with any of these is not the citation plural we want to
# show ("Lehrern" is dative; "Tischchen" a diminutive).
NON_CITATION_TAGS = {"genitive", "dative", "accusative", "diminutive",
                     "table-tags", "inflection-template", "class"}
MAX_SENSES = 5
MAX_PLURALS = 2


def extract_entry(e, lang_code):
    """One kaikki JSON object -> a database row, or None if not wanted."""
    word = e.get("word")
    if not word or e.get("lang_code") != lang_code:
        return None
    pos = e.get("pos") or ""
    glosses, genders, lemma = [], [], ""
    for s in e.get("senses") or []:
        # Glosses are hierarchical, broad -> specific; keep the specific one.
        g = s.get("glosses") or s.get("raw_glosses") or []
        if g and g[-1] not in glosses:
            glosses.append(g[-1])
        for t in s.get("tags") or []:
            code = GENDER_TAGS.get(t)
            if code and code not in genders:
                genders.append(code)
        if not lemma:
            for ref in (s.get("form_of") or []) + (s.get("alt_of") or []):
                if ref.get("word"):
                    lemma = ref["word"]
                    break
    if not glosses:
        return None
    plurals = []
    if pos == "noun":
        for f in e.get("forms") or []:
            tags = f.get("tags") or []
            form = f.get("form") or ""
            if ("plural" in tags and form and form != "-"
                    and not NON_CITATION_TAGS.intersection(tags)
                    and form not in plurals):
                plurals.append(form)
    ipa = next((s["ipa"] for s in e.get("sounds") or [] if s.get("ipa")), "")
    return (word, word.lower(), pos, "/".join(genders), ipa,
            " / ".join(plurals[:MAX_PLURALS]),
            json.dumps(glosses[:MAX_SENSES], ensure_ascii=False), lemma)


def compile_lexicon(lines, lang_code, out_path):
    """Stream JSONL lines into a fresh SQLite lexicon; atomic replace."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    tmp_path.unlink(missing_ok=True)
    db = sqlite3.connect(tmp_path)
    db.execute("CREATE TABLE entries (word TEXT, word_lc TEXT, pos TEXT, "
               "gender TEXT, ipa TEXT, plural TEXT, senses TEXT, lemma TEXT)")
    kept = total = 0
    batch = []
    for line in lines:
        total += 1
        try:
            row = extract_entry(json.loads(line), lang_code)
        except json.JSONDecodeError:
            continue
        if row:
            batch.append(row)
            kept += 1
        if len(batch) >= 10_000:
            db.executemany("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?)", batch)
            batch.clear()
            print(f"\r  {total:,} entries scanned, {kept:,} kept",
                  end="", flush=True)
    db.executemany("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?)", batch)
    db.execute("CREATE INDEX idx_word_lc ON entries(word_lc)")
    db.commit()
    db.close()
    tmp_path.replace(out_path)
    print(f"\r  {total:,} entries scanned, {kept:,} kept -> {out_path}")
    return kept


def build(lang_code):
    name = LANGS[lang_code]
    url = URL_TEMPLATE.format(name=name)
    print(f"{name}: streaming {url}")
    with urllib.request.urlopen(url) as resp:
        lines = gzip.GzipFile(fileobj=resp)
        compile_lexicon(lines, lang_code, OUT_DIR / f"wikt_{lang_code}.sqlite")


if __name__ == "__main__":
    requested = sys.argv[1:] or ["de", "es"]
    bad = [c for c in requested if c not in LANGS]
    if bad:
        sys.exit(f"Unknown language(s) {bad} — choose from {list(LANGS)}")
    for code in requested:
        build(code)
