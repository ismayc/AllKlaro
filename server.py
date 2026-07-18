"""AllKlaro — local realtime German / Spanish / English conversation translator.

Pipeline: browser mic (or BlackHole loopback) -> WebSocket (16 kHz PCM) ->
VAD (Silero if available, energy fallback) -> mlx-whisper transcription ->
Ollama translation with conversation context (streamed back token by token).

Run:  uv run uvicorn server:app --host 127.0.0.1 --port 8710
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
import zlib
from difflib import SequenceMatcher
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("translator")

# ---------------------------------------------------------------- configuration

SAMPLE_RATE = 16000
FRAME_MS = 32                      # VAD analysis frame
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
PREROLL_FRAMES = 10                # ~320 ms of audio kept before speech onset
START_VOICED_FRAMES = 3            # frames above threshold to trigger speech
END_SILENCE_FRAMES = 22            # ~700 ms of silence ends the utterance
EARLY_SILENCE_FRAMES = 10          # ~320 ms: start transcribing speculatively
MIN_UTTERANCE_SEC = 0.4
MAX_UTTERANCE_SEC = 30.0           # hard force-flush, even mid-word
SOFT_MAX_SEC = 8.0                 # after this, split at the last micro-pause
MICRO_PAUSE_FRAMES = 6             # ~190 ms dip = natural split point in
                                   # continuous speech (videos, fast talkers)
PARTIAL_WINDOW_FRAMES = 375        # live partials look at the last ~12 s only
PARTIAL_INTERVAL_SEC = 1.5         # how often to emit live partial transcripts
HISTORY_TURNS = 6                  # recent exchanges fed to the translator
MERGE_GAP_SEC = 2.0                # resumed-within window for fragment merging
MERGE_MAX_CHARS = 300              # never grow merged utterances beyond this
ECHO_WINDOW_SEC = 6.0              # cross-channel duplicate suppression window
ECHO_MIN_CHARS = 16                # never dedupe short phrases ("Genau!") —
                                   # people legitimately repeat those

# A transcript that ends mid-sentence (no terminal punctuation) is a merge
# candidate — crucial for German, where the meaning-carrying verb comes last.
SENTENCE_END_RE = re.compile(r"[.!?…]['\")\]]?\s*$")

WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma3:12b"
GLOSSARY_PATH = Path(__file__).parent / "glossary.txt"
DIALECTS_PATH = Path(__file__).parent / "dialects.txt"
CORRECTIONS_PATH = Path(__file__).parent / "corrections.jsonl"
CORRECTION_EXAMPLES = 3            # retrieved corrections shown to the model
# Compiled by build_gender_lexicon.py from dict.cc / FreeDict exports. They
# live in ~/.cache (dict.cc data is private-use only — never commit it).
GENDER_LEXICON_PATHS = {
    "de": Path.home() / ".cache" / "allklaro" / "de_noun_genders.tsv",
    "es": Path.home() / ".cache" / "allklaro" / "es_noun_genders.tsv",
}
# Output-side maps (target-language noun -> gender) for verifying the
# model's own output; also built by build_gender_lexicon.py.
OUTPUT_GENDER_PATHS = {
    "de": Path.home() / ".cache" / "allklaro" / "de_output_genders.tsv",
    "es": Path.home() / ".cache" / "allklaro" / "es_output_genders.tsv",
}
# Per-form German noun readings (case/number), built by build_noun_forms.py
# from the Wiktionary-derived german-nouns CSV.
NOUN_FORMS_PATH = Path.home() / ".cache" / "allklaro" / "de_noun_forms.tsv"
# Tap-to-look-up dictionaries compiled from Wiktionary (kaikki.org) by
# build_wiktionary_lexicon.py; optional — /api/lookup says how to build one.
WIKTIONARY_PATHS = {
    code: Path.home() / ".cache" / "allklaro" / f"wikt_{code}.sqlite"
    for code in ("de", "en", "es")
}
LOOKUP_LIMIT = 6                   # entries returned per looked-up word
GENDER_NOTE_LIMIT = 8              # max per-sentence dictionary notes

VAD_BACKEND = os.environ.get("VAD_BACKEND", "silero")  # "silero" | "energy"
SILERO_URL = ("https://github.com/snakers4/silero-vad/raw/master/"
              "src/silero_vad/data/silero_vad.onnx")
SILERO_PATH = Path.home() / ".cache" / "allklaro" / "silero_vad.onnx"

# Whisper hallucinates these on silence/noise; drop them.
HALLUCINATION_RE = re.compile(
    r"^(thank you\.?|thanks for watching\.?|you\.?|bye\.?|"
    r"untertitel.*|"
    r"vielen dank\.?|"
    r"subt[ií]tulos.*|¡?gracias por ver.*|¡?gracias\.?|suscr[ií]bete.*|"
    r"copyright .*|amara\.org.*|\.+)$",
    re.IGNORECASE,
)

# mlx is not thread-safe across simultaneous calls -> serialize all transcription.
whisper_executor = ThreadPoolExecutor(max_workers=1)
STATIC_DIR = Path(__file__).parent / "static"


_glossary_cache = {"mtime": None, "lines": []}


def load_glossary() -> list[str]:
    """User-maintained terms in glossary.txt: `term` or `term = translation`.
    Reloaded automatically when the file changes."""
    try:
        mtime = GLOSSARY_PATH.stat().st_mtime
    except OSError:
        _glossary_cache.update(mtime=None, lines=[])
        return []
    if mtime != _glossary_cache["mtime"]:
        lines = [ln.strip() for ln in GLOSSARY_PATH.read_text().splitlines()]
        _glossary_cache.update(
            mtime=mtime,
            lines=[ln for ln in lines if ln and not ln.startswith("#")])
    return _glossary_cache["lines"]


def glossary_whisper_terms() -> str:
    """Source-side spellings only, to bias Whisper toward proper nouns."""
    return ", ".join(ln.split("=")[0].strip() for ln in load_glossary())


# ----------------------------------------------------------------- dialects

_dialects_cache = {"mtime": None, "map": {}}


def load_dialects() -> dict[str, tuple[str, bool]]:
    """dialects.txt: dialect token -> (standard gloss, ambiguous).

    Ambiguous entries ("? nett = net (nicht)") are real standard-German
    words too; they are only hinted when the sentence also contains an
    unambiguous dialect marker."""
    try:
        mtime = DIALECTS_PATH.stat().st_mtime
    except OSError:
        _dialects_cache.update(mtime=None, map={})
        return {}
    if mtime != _dialects_cache["mtime"]:
        entries = {}
        for line in DIALECTS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ambiguous = line.startswith("?")
            term, _, standard = line.lstrip("? ").partition("=")
            if term.strip() and standard.strip():
                entries[term.strip().lower()] = (standard.strip(), ambiguous)
        _dialects_cache.update(mtime=mtime, map=entries)
    return _dialects_cache["map"]


def dialect_notes(text: str, source: str) -> str | None:
    """A hint block when the German source contains dialect markers.

    Whisper mangles spoken dialect in meaning-changing ways ("net
    verstanne" -> "nett verstarne", which gemma then translates as
    'understood nicely' — an inversion). Whisper-side prompt biasing made
    transcripts WORSE in testing, so the fix lives here: name the likely
    intended forms and let the model translate the intended meaning.
    """
    if source != "de":
        return None
    lexicon = load_dialects()
    if not lexicon:
        return None
    hits, seen, marker = [], set(), False
    for token in re.findall(r"[^\W\d_]+", text.lower()):
        if token in seen:
            continue
        seen.add(token)
        entry = lexicon.get(token)
        if entry:
            hits.append((token, entry[0]))
            marker = marker or not entry[1]
    if not marker:                     # ambiguous words alone prove nothing
        return None
    listed = "; ".join(f'"{t}" = {std}' for t, std in hits[:10])
    return ("The German speaker is using regional dialect (e.g. Berlin or "
            "Hessian), and speech recognition may have mis-heard dialect "
            f"words. In this sentence: {listed}. Translate the intended "
            "standard meaning naturally.")


# ---------------------------------------------------------------- gender lexicon

_gender_caches: dict[str, dict] = {}   # target -> {"mtime", "map"}
ARTICLES = {"de": {"m": "der", "f": "die", "n": "das"},
            "es": {"m": "el", "f": "la"}}


def load_gender_lexicon(target: str) -> dict[str, tuple[str, str]]:
    """source word -> (target-language noun, article), mtime-cached."""
    path = GENDER_LEXICON_PATHS.get(target)
    articles = ARTICLES.get(target)
    if path is None or articles is None:
        return {}
    cache = _gender_caches.setdefault(target, {"mtime": None, "map": {}})
    try:
        mtime = path.stat().st_mtime
    except OSError:
        cache.update(mtime=None, map={})
        return {}
    if mtime != cache["mtime"]:
        entries = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[2] in articles:
                entries[parts[0]] = (parts[1], articles[parts[2]])
        cache.update(mtime=mtime, map=entries)
    return cache["map"]


def gender_notes(text: str, target: str) -> str | None:
    """Dictionary gender facts for source words that are target-language
    loanwords/cognates.

    Fires when translating into German or Spanish. The lexicons hold
    same-spelling pairs exclusively (caipirinha -> der Caipirinha,
    problem -> el problema), so a note can never push the model toward a
    false-friend word choice; words any dictionary lists with several
    genders (dict.cc "Margarita {m} {f}") were dropped at build time.
    """
    lexicon = load_gender_lexicon(target)
    if not lexicon:
        return None
    notes, seen = [], set()
    for word in re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", text.lower()):
        if len(word) < 3 or word in seen:
            continue
        seen.add(word)
        hit = lexicon.get(word)
        if hit:
            notes.append(f"{word} → {hit[1]} {hit[0]}")
            if len(notes) >= GENDER_NOTE_LIMIT:
                break
    if not notes:
        return None
    return ("Dictionary genders for nouns in this sentence — authoritative, "
            "overriding any general rule above: " + "; ".join(notes) + ".")


_output_caches: dict[str, dict] = {}   # target -> {"mtime", "map"}


def load_output_genders(target: str) -> dict[str, tuple[str, str]]:
    """target-language noun (lowercased) -> (display form, gender m/f/n)."""
    path = OUTPUT_GENDER_PATHS.get(target)
    if path is None:
        return {}
    cache = _output_caches.setdefault(target, {"mtime": None, "map": {}})
    try:
        mtime = path.stat().st_mtime
    except OSError:
        cache.update(mtime=None, map={})
        return {}
    if mtime != cache["mtime"]:
        entries = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[2] in "mfn":
                entries[parts[0]] = (parts[1], parts[2])
        cache.update(mtime=mtime, map=entries)
    return cache["map"]


# ------------------------------------------------------- agreement checking
#
# Every German determiner form is mapped to its complete set of possible
# (case, number, gender) readings — gender None means "any" (plural).
# An NP is flagged only when NO determiner reading is consistent with any
# reading of the noun form (from the Wiktionary paradigm table, or a
# conservative fallback synthesized from the gender lexicon). Wrong in
# every interpretation = provably wrong; anything else passes.

def _derword_readings(ending):
    return {
        "e": {("n", "s", "f"), ("a", "s", "f"),
              ("n", "p", None), ("a", "p", None)},
        "er": {("n", "s", "m"), ("d", "s", "f"), ("g", "s", "f"),
               ("g", "p", None)},
        "es": {("n", "s", "n"), ("a", "s", "n"),
               ("g", "s", "m"), ("g", "s", "n")},
        "en": {("a", "s", "m"), ("d", "p", None)},
        "em": {("d", "s", "m"), ("d", "s", "n")},
    }[ending]


DET_READINGS = {
    "der": _derword_readings("er"), "die": _derword_readings("e"),
    "den": _derword_readings("en"), "dem": _derword_readings("em"),
    "des": {("g", "s", "m"), ("g", "s", "n")},
    "das": {("n", "s", "n"), ("a", "s", "n")},
    # Contractions fix case AND gender.
    "am": _derword_readings("em"), "im": _derword_readings("em"),
    "beim": _derword_readings("em"), "vom": _derword_readings("em"),
    "zum": _derword_readings("em"), "zur": {("d", "s", "f")},
    "ins": {("a", "s", "n")}, "ans": {("a", "s", "n")},
}
for _end in ("e", "er", "es", "en", "em"):
    DET_READINGS["dies" + _end] = _derword_readings(_end)
    # "jeder" has no plural readings.
    DET_READINGS["jed" + _end] = {r for r in _derword_readings(_end)
                                  if r[1] == "s"}

# ein-words (mixed declension) by ending; the article "ein" itself has no
# plural, kein-/possessives add plural readings.
EIN_READINGS = {
    "": {("n", "s", "m"), ("n", "s", "n"), ("a", "s", "n")},
    "e": {("n", "s", "f"), ("a", "s", "f")},
    "en": {("a", "s", "m")},
    "em": {("d", "s", "m"), ("d", "s", "n")},
    "er": {("d", "s", "f"), ("g", "s", "f")},
    "es": {("g", "s", "m"), ("g", "s", "n")},
}
POSS_PLURAL = {"e": {("n", "p", None), ("a", "p", None)},
               "en": {("d", "p", None)}, "er": {("g", "p", None)}}
# Definite forms double as relative pronouns after a comma ("der Mann, dem
# Frauen vertrauen" — dem refers to Mann, Frauen is the subject).
RELATIVE_FORMS = {"der", "die", "das", "den", "dem", "des"}

ES_IMPOSSIBLE = {"el": "f", "un": "f", "la": "m", "una": "m"}
# After these article forms, mixed adjective declension allows exactly one
# ending set ("ein schöne Tag" is impossible in every reading).
DE_ADJ_ENDINGS = {
    "das": ("e",), "dem": ("en",), "des": ("en",),
    "ein": ("er", "es"), "eine": ("e",), "einen": ("en",),
    "einem": ("en",), "einer": ("en",), "eines": ("en",),
    "kein": ("er", "es"), "keinem": ("en",), "keines": ("en",),
    "beim": ("en",), "vom": ("en",), "zum": ("en",), "zur": ("en",),
    "am": ("en",), "im": ("en",), "ins": ("e",), "ans": ("e",),
}

# ------------------------------------------------------ prepositional case
#
# These prepositions govern case unconditionally, so case IS computable
# here — no parsing needed. Two-way prepositions (in/an/auf/über/unter/
# neben/vor/hinter/zwischen) are deliberately absent: their case depends
# on motion semantics ("in die Schule gehen" vs "in der Schule sein"),
# which no regex can see. Genitive prepositions also accept the colloquial
# dative that dominates spoken German.

DE_PREP_CASE = {
    "aus": "d", "bei": "d", "mit": "d", "nach": "d", "seit": "d",
    "von": "d", "zu": "d", "gegenüber": "d",
    "durch": "a", "für": "a", "gegen": "a", "ohne": "a", "um": "a",
    "während": "gd", "wegen": "gd", "trotz": "gd",
    "anstatt": "gd", "statt": "gd",
}
CASE_NAMES = {"n": "nominative", "a": "accusative",
              "d": "dative", "g": "genitive"}
DE_DEF_ART = {
    "n": {"m": "der", "f": "die", "n": "das"},
    "a": {"m": "den", "f": "die", "n": "das"},
    "d": {"m": "dem", "f": "der", "n": "dem"},
    "g": {"m": "des", "f": "der", "n": "des"},
}
DE_EIN_ENDINGS = {
    "n": {"m": "", "f": "e", "n": ""},
    "a": {"m": "en", "f": "e", "n": ""},
    "d": {"m": "em", "f": "er", "n": "em"},
    "g": {"m": "es", "f": "er", "n": "es"},
}
DE_POSS_RE = re.compile(r"^(ein|kein|mein|dein|sein|ihr|unser|euer|eur)"
                        r"(e|en|em|er|es)?$")
# Bare "sein"/"ihr" are verb/pronoun homonyms ("mit ihr Deutsch üben" =
# with her, not her German) — only their inflected forms are safely
# possessive determiners.
DE_BARE_AMBIGUOUS = {"sein", "ihr"}

DE_PREP_NP_RE = re.compile(
    r"\b(?i:(gegenüber|während|anstatt|wegen|trotz|statt|durch|gegen|nach"
    r"|ohne|seit|aus|bei|mit|von|für|um|zu))\s+"
    r"([a-zäöüß]+)\s+(?:([a-zäöüß-]+)\s+)?(?:([a-zäöüß-]+)\s+)?"
    r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüß-]*)")


def det_readings(detl: str):
    """All (case, number, gender) readings of a determiner form, or None
    for words we can't classify (pronouns, quantifiers, bare sein/ihr)."""
    if detl in DET_READINGS:
        return DET_READINGS[detl]
    m = DE_POSS_RE.match(detl)
    if not m or detl in DE_BARE_AMBIGUOUS:
        return None
    ending = m.group(2) or ""
    readings = set(EIN_READINGS.get(ending, ()))
    if m.group(1) != "ein":            # kein/possessives have plurals too
        readings |= POSS_PLURAL.get(ending, set())
    return readings or None


_noun_forms_cache = {"mtime": None, "map": {}}


def load_noun_forms() -> dict[str, tuple]:
    """form (lowercased) -> ((lemma, gender, frozenset of case+number
    codes), ...) from the compiled Wiktionary paradigm table."""
    try:
        mtime = NOUN_FORMS_PATH.stat().st_mtime
    except OSError:
        _noun_forms_cache.update(mtime=None, map={})
        return {}
    if mtime != _noun_forms_cache["mtime"]:
        entries: dict[str, list] = {}
        for line in NOUN_FORMS_PATH.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 4:
                entries.setdefault(parts[0], []).append(
                    (parts[1], parts[2], frozenset(parts[3].split(","))))
        _noun_forms_cache.update(
            mtime=mtime, map={k: tuple(v) for k, v in entries.items()})
    return _noun_forms_cache["map"]


def _noun_analyses(form: str):
    """Possible (lemma, gender, readings) of a noun form. Falls back to the
    gender lexicon with conservative assumptions: the bare lemma covers the
    singular (genitive only for feminines — m/n need -s), and any plural
    except the dative, which requires -n or -s."""
    entries = load_noun_forms().get(form.lower())
    if entries:
        return entries
    fallback = load_output_genders("de").get(form.lower())
    if not fallback:
        return None
    display, g = fallback
    codes = {"ns", "as", "ds"} | ({"gs"} if g == "f" else set())
    codes |= {"np", "ap", "gp"}
    if form.lower().endswith(("n", "s")):
        codes.add("dp")
    return ((display, g, frozenset(codes)),)


def _compatible(readings, entries) -> bool:
    """Is any determiner reading consistent with any noun reading?"""
    return any(
        c + n in codes and (n == "p" or g is None or g == g2)
        for c, n, g in readings
        for _, g2, codes in entries)


def _gender_options(entries, form):
    """(display, genders) for messaging an incompatible NP. Prefers entries
    whose lemma is the form itself (homograph noise like SM/SMS otherwise
    leaks in); more than two candidate genders means the message would be
    mush, so callers skip flagging then."""
    exact = [e for e in entries if e[0].lower() == form.lower()]
    pool = exact or entries
    return pool[0][0], sorted({g for _, g, _ in pool})


def _wanted_det(detl, cases, gender):
    """The determiner form(s) that would be correct, for the message."""
    if detl in RELATIVE_FORMS or detl in ("das",):
        return " or ".join(sorted({f'"{DE_DEF_ART[c][gender]}"'
                                   for c in cases}))
    m = DE_POSS_RE.match(detl)
    if m:
        stem = "euer" if m.group(1) == "eur" else m.group(1)
        return " or ".join(sorted({f'"{stem + DE_EIN_ENDINGS[c][gender]}"'
                                   for c in cases}))
    return None
# Measure/quantity constructions carry their own agreement ("ein bisschen
# Ruhe" is correct despite Ruhe being feminine) — skip the whole phrase.
MEASURE_WORDS = {"paar", "bisschen", "wenig", "viel", "etwas", "mehr", "poco"}
UNDECLINED_ADJS = {"rosa", "lila", "prima", "super", "extra", "klasse",
                   "orange", "beige"}
# Words between determiner and noun that end like adjectives but aren't
# ("Das werden gute Freunde", "das heute Abend") — their presence means the
# "determiner" is really a pronoun, so the whole match is skipped.
NON_ADJ_MIDS = {"werden", "waren", "wurde", "wurden", "würde", "würden",
                "haben", "hatten", "hätte", "hätten", "sollen", "wollen",
                "können", "müssen", "dürfen", "mögen", "möchten",
                "heute", "morgen", "gestern", "immer", "wieder", "gerne",
                "leider", "lieber", "eher", "weiter", "außer", "über",
                "unter", "hinter"}


def _plausible_adj(word: str) -> bool:
    """Could this be an attributive adjective inside an NP? Attributive
    adjectives are always declined, so they end -e/-er/-es/-en/-em; verbs,
    adverbs, and nested determiners mean we're not looking at one NP."""
    return (word.endswith(("e", "er", "es", "en", "em"))
            and word not in NON_ADJ_MIDS
            and word not in DET_READINGS
            and not DE_POSS_RE.match(word))
AGREEMENT_EXCEPTIONS = {("ein", "uhr")}   # "um ein Uhr" (telling time)
GENDER_NAMES = {"m": "masculine", "f": "feminine", "n": "neuter"}
NOM_ARTICLE = {"de": {"m": "der", "f": "die", "n": "das"},
               "es": {"m": "el", "f": "la"}}

def _det_alternation() -> str:
    forms = set(DET_READINGS)
    for stem in ("ein", "kein", "mein", "dein", "sein", "ihr", "unser",
                 "euer", "eur"):
        for end in ("", "e", "en", "em", "er", "es"):
            forms.add(stem + end)
    return "|".join(sorted(forms, key=len, reverse=True))


DE_NP_RE = re.compile(
    r"\b(?i:(" + _det_alternation() + r"))\s+"
    r"(?:([a-zäöüß-]+)\s+)?(?:([a-zäöüß-]+)\s+)?"
    r"([A-ZÄÖÜ][A-Za-zÄÖÜäöüß-]*)")
ES_NP_RE = re.compile(
    r"\b(?i:(el|la|una|un))\s+(?:([a-záéíóúñü-]+)\s+)?([a-záéíóúñü-]+)")


def agreement_issues(text: str, target: str) -> list[str]:
    """Provably-wrong article/adjective agreement in translated text, as
    human-readable facts suitable for a corrective prompt. Empty when the
    output map is missing or nothing matches."""
    genders = load_output_genders(target)
    if target == "de":
        if not genders and not load_noun_forms():
            return []
    elif not genders:
        return []
    issues = []
    if target == "de":
        flagged_spans = []
        # Pass 1: a case-governing preposition pins the case; the NP must
        # have a consistent reading within that case.
        for m in DE_PREP_NP_RE.finditer(text):
            prep, det, noun = m.group(1), m.group(2), m.group(5)
            mids = [w for w in (m.group(3), m.group(4)) if w]
            detl = det.lower()
            if ((detl, noun.lower()) in AGREEMENT_EXCEPTIONS
                    or any(w in MEASURE_WORDS for w in mids)
                    or detl in MEASURE_WORDS
                    or not all(_plausible_adj(w) for w in mids)):
                continue
            if (detl in RELATIVE_FORMS
                    and text[:m.start()].rstrip().endswith((",", ";"))):
                continue               # ", mit dem …" — relative pronoun
            readings = det_readings(detl)
            entries = _noun_analyses(noun)
            if readings is None or entries is None:
                continue
            cases = DE_PREP_CASE[prep.lower()]
            filtered = {r for r in readings if r[0] in cases}
            if filtered and _compatible(filtered, entries):
                continue
            display, noun_genders = _gender_options(entries, noun)
            if len(noun_genders) > 2:
                continue               # message would be mush — skip
            wants = sorted({w for g in noun_genders
                            for w in [_wanted_det(detl, cases, g)] if w})
            gender_txt = " or ".join(GENDER_NAMES[g] for g in noun_genders)
            case_names = "/".join(CASE_NAMES[c] for c in sorted(cases))
            issues.append(
                f'"{prep} {det} {noun}" is wrong — "{prep.lower()}" takes '
                f'the {case_names} case and {display} is {gender_txt}'
                + (f", so it must be {' or '.join(wants)} {noun}."
                   if wants else "."))
            flagged_spans.append((m.start(), m.end()))
        # Pass 2: case unknown — flag NPs with NO consistent reading in
        # any (case, number), via determiner/noun-paradigm intersection.
        for m in DE_NP_RE.finditer(text):
            if any(s <= m.start() < e for s, e in flagged_spans):
                continue
            det, noun = m.group(1), m.group(4)
            detl = det.lower()
            mids = [w for w in (m.group(2), m.group(3)) if w]
            if ((detl, noun.lower()) in AGREEMENT_EXCEPTIONS
                    or any(w in MEASURE_WORDS for w in mids)
                    or not all(_plausible_adj(w) for w in mids)):
                continue
            before = text[:m.start()].rstrip()
            if detl in RELATIVE_FORMS and (
                    before.endswith((",", ";"))
                    or re.search(r"[,;]\s*[a-zäöüß]+$", before)):
                continue   # "…, den Frauen vertrauen", "…, mit dem Frauen …"
            readings = det_readings(detl)
            entries = _noun_analyses(noun)
            if readings is None or entries is None:
                continue
            if not _compatible(readings, entries):
                display, noun_genders = _gender_options(entries, noun)
                if len(noun_genders) <= 2:
                    gender_txt = " or ".join(GENDER_NAMES[g]
                                             for g in noun_genders)
                    correct = "/".join(DE_DEF_ART["n"][g]
                                       for g in noun_genders)
                    issues.append(
                        f'"{det} {noun}" is wrong — {display} is '
                        f'{gender_txt}: {correct} {display}.')
            elif (mids and detl in DE_ADJ_ENDINGS
                    and mids[-1] not in UNDECLINED_ADJS
                    and not any(mids[-1].endswith(e)
                                for e in DE_ADJ_ENDINGS[detl])):
                endings = " or ".join(f"-{e}" for e in DE_ADJ_ENDINGS[detl])
                issues.append(
                    f'"{det} {mids[-1]} {noun}" is wrong — after '
                    f'"{detl}" the adjective must end in {endings}.')
    elif target == "es":
        for m in ES_NP_RE.finditer(text):
            art = m.group(1).lower()
            # Spanish adjectives are postnominal, so the word right after
            # the article is usually the noun — try it first ("la problema
            # es" must check "problema", not "es").
            words = [w for w in (m.group(2), m.group(3)) if w]
            if any(w in MEASURE_WORDS for w in words):
                continue
            entry, noun = None, None
            for noun in words:
                entry = genders.get(noun.lower())
                if entry:
                    break
            if not entry:
                continue
            display, g = entry
            # "el agua / un arma" are correct for stressed a- feminines.
            if art in ("el", "un") and g == "f" and noun[:1] in "aáh":
                continue
            if g in ES_IMPOSSIBLE[art]:
                issues.append(
                    f'"{m.group(1)} {noun}" is wrong — {display} is '
                    f'{GENDER_NAMES[g]}: {NOM_ARTICLE["es"][g]} {display}.')
    return issues


# ------------------------------------------------------------------ corrections

_corrections_cache = {"mtime": None, "items": []}


def load_corrections() -> list[dict]:
    """User-corrected translations from corrections.jsonl, mtime-cached.

    Repeated corrections of the same utterance keep only the newest one, so
    a re-edit supersedes rather than contradicts the earlier attempt.
    """
    try:
        mtime = CORRECTIONS_PATH.stat().st_mtime
    except OSError:
        _corrections_cache.update(mtime=None, items=[])
        return []
    if mtime != _corrections_cache["mtime"]:
        latest: dict[tuple, dict] = {}
        for line in CORRECTIONS_PATH.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (isinstance(item, dict) and item.get("text")
                    and item.get("corrected")):
                continue
            key = (item.get("source"), item.get("target"),
                   item["text"].strip())
            latest.pop(key, None)          # re-insert so newest is last
            latest[key] = item
        _corrections_cache.update(mtime=mtime, items=list(latest.values()))
    return _corrections_cache["items"]


def save_correction(item: dict) -> None:
    with CORRECTIONS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower(), re.UNICODE)
            if len(w) >= 3}


def relevant_corrections(text: str, source: str, target: str,
                         k: int = CORRECTION_EXAMPLES) -> list[dict]:
    """The k stored corrections most lexically similar to `text`, same
    direction only. Word-overlap retrieval — no embeddings needed at this
    scale, and a loosely related example is harmless as a few-shot pair."""
    words = content_words(text)
    scored = []
    for idx, c in enumerate(load_corrections()):
        if c.get("source") != source or c.get("target") != target:
            continue
        overlap = len(words & content_words(c["text"]))
        if overlap:
            scored.append((overlap, idx, c))
    scored.sort()                          # score, then file order (oldest first)
    return [c for _, _, c in scored[-k:]]


def is_degenerate(text: str) -> bool:
    """Whisper repetition loops ("ninninnin…", a phrase echoed 30×) compress
    absurdly well; real speech doesn't."""
    stripped = re.sub(r"\s+", "", text)
    if len(stripped) < 60:
        return False
    ratio = len(stripped) / len(zlib.compress(stripped.encode("utf-8")))
    return ratio > 4.0


def collapse_repeats(text: str) -> str:
    """Collapse short units repeated 8+ times ("L-L-L-L…" → "L-"), rescuing
    the real speech in a segment that ends in a repetition loop."""
    return re.sub(r"(.{1,12}?)\1{7,}", r"\1", text)


def _clean_segment(raw: str) -> str | None:
    """A segment's usable text, or None if it is a repetition artifact."""
    if is_degenerate(raw):
        return None                    # long-unit loop (repeated phrases)
    collapsed = collapse_repeats(raw)
    if len(collapsed) < 0.4 * len(raw) and len(collapsed.strip()) < 30:
        return None                    # mostly loop, nothing left worth keeping
    return collapsed


def clean_transcript(result: dict) -> str:
    """Drop low-confidence and degenerate segments (music, jingles, repetition
    loops) the hallucination blocklist has never seen, using Whisper's own
    signals plus text-level repetition checks."""
    segments = result.get("segments")
    if not segments:
        return (_clean_segment(result.get("text", "").strip()) or "").strip()
    kept = []
    for s in segments:
        if (s.get("no_speech_prob", 0.0) > 0.6
                and s.get("avg_logprob", 0.0) < -1.0):
            continue
        if s.get("compression_ratio", 0.0) > 2.4:
            continue                   # Whisper's own repetitiveness signal
        text = _clean_segment(s.get("text", ""))
        if text is not None:
            kept.append(text)
    return "".join(kept).strip()


def transcribe(audio: np.ndarray, language: str | None,
               prompt: str | None = None) -> dict:
    import mlx_whisper

    return mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=WHISPER_REPO,
        language=language,
        initial_prompt=prompt,
        # A ladder (not a fixed 0.0) lets Whisper retry a segment at higher
        # temperature when the greedy decode degenerates into a repetition loop.
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.5,
        condition_on_previous_text=False,
    )


# ------------------------------------------------------------------ VAD scorers


class EnergyScorer:
    """Frame is voiced when RMS exceeds an adaptive noise-floor threshold."""

    def __init__(self):
        self.noise = 100.0

    def __call__(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        voiced = rms > max(250.0, self.noise * 3.0)
        if not voiced:
            self.noise = 0.95 * self.noise + 0.05 * rms
        return voiced


_silero_session = None  # onnxruntime session, loaded at startup if available


CONTEXT_SAMPLES = 64  # Silero v5 prepends this much of the previous chunk


class SileroScorer:
    """Neural VAD: frame is voiced when Silero's speech probability > 0.5.

    Handles both the v5 onnx interface (single `state` tensor, and each
    512-sample chunk prefixed with the last 64 samples of the previous one)
    and the older h/c interface, detected from the model's declared inputs.
    """

    def __init__(self, session):
        self.session = session
        names = [i.name for i in session.get_inputs()]
        self.v5 = "state" in names
        if self.v5:
            self.state = np.zeros((2, 1, 128), dtype=np.float32)
            self.context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        else:
            self.h = np.zeros((2, 1, 64), dtype=np.float32)
            self.c = np.zeros((2, 1, 64), dtype=np.float32)
        self.sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.active = False

    def __call__(self, frame: np.ndarray) -> bool:
        x = frame.astype(np.float32) / 32768.0
        if self.v5:
            with_ctx = np.concatenate([self.context, x])[None, :]
            prob, self.state = self.session.run(
                None, {"input": with_ctx, "state": self.state, "sr": self.sr})
            self.context = x[-CONTEXT_SAMPLES:]
        else:
            prob, self.h, self.c = self.session.run(
                None, {"input": x[None, :], "sr": self.sr,
                       "h": self.h, "c": self.c})
        p = float(np.asarray(prob).ravel()[0])
        # Hysteresis: harder to start speech than to continue it, so quiet
        # word-endings don't chop utterances mid-sentence.
        voiced = p > (0.35 if self.active else 0.5)
        self.active = voiced
        return voiced


def load_silero():
    """Download (once) and load the Silero VAD onnx model; None on any failure."""
    global _silero_session
    if VAD_BACKEND != "silero":
        log.info("VAD backend forced to energy (VAD_BACKEND=%s)", VAD_BACKEND)
        return
    try:
        import onnxruntime

        if not SILERO_PATH.exists():
            SILERO_PATH.parent.mkdir(parents=True, exist_ok=True)
            log.info("Downloading Silero VAD model ...")
            r = httpx.get(SILERO_URL, follow_redirects=True, timeout=60)
            r.raise_for_status()
            SILERO_PATH.write_bytes(r.content)
        session = onnxruntime.InferenceSession(
            str(SILERO_PATH), providers=["CPUExecutionProvider"])
        # Smoke-test the interface before committing to it.
        SileroScorer(session)(np.zeros(FRAME_SAMPLES, dtype=np.int16))
        _silero_session = session
        log.info("Silero VAD ready.")
    except Exception as exc:
        log.warning("Silero VAD unavailable (%s); using energy VAD.", exc)


def make_scorer():
    return SileroScorer(_silero_session) if _silero_session else EnergyScorer()


class VadSession:
    """Speech segmenter over int16 16 kHz frames; voicing decided by `scorer`."""

    def __init__(self, scorer=None, end_silence_frames: int = END_SILENCE_FRAMES):
        self.scorer = scorer or EnergyScorer()
        self.end_silence = end_silence_frames
        self.preroll: list[np.ndarray] = []
        self.speech: list[np.ndarray] = []
        self.in_speech = False
        self.voiced_run = 0
        self.silence_run = 0
        self.split_at = None  # frame index of the last micro-pause boundary
        self.early_event = None   # audio ready for speculative transcription
        self.speculating = False  # a speculation is plausibly still valid

    def feed(self, frame: np.ndarray) -> np.ndarray | None:
        """Feed one frame; returns a finished utterance (float32) or None."""
        voiced = self.scorer(frame)

        if not self.in_speech:
            self.preroll.append(frame)
            if len(self.preroll) > PREROLL_FRAMES:
                self.preroll.pop(0)
            self.voiced_run = self.voiced_run + 1 if voiced else 0
            if self.voiced_run >= START_VOICED_FRAMES:
                self.in_speech = True
                self.speech = list(self.preroll)
                self.preroll = []
                self.silence_run = 0
            return None

        self.speech.append(frame)
        if voiced:
            self.silence_run = 0
            self.speculating = False  # speech resumed; speculation is stale
        else:
            self.silence_run += 1
        if (not voiced and self.silence_run == MICRO_PAUSE_FRAMES
                and len(self.speech) * FRAME_MS / 1000 >= MIN_UTTERANCE_SEC):
            self.split_at = len(self.speech)
        seconds = len(self.speech) * FRAME_MS / 1000
        voiced_sec = seconds - self.silence_run * FRAME_MS / 1000
        if (not voiced and self.silence_run == EARLY_SILENCE_FRAMES
                and self.end_silence > EARLY_SILENCE_FRAMES
                and voiced_sec >= MIN_UTTERANCE_SEC):
            # The pause might become final: hand out the audio now so
            # transcription can run while we wait out the rest of the pause.
            self.early_event = (np.concatenate(self.speech)
                                .astype(np.float32) / 32768.0)
            self.speculating = True
        if self.silence_run >= self.end_silence or seconds > MAX_UTTERANCE_SEC:
            utterance = np.concatenate(self.speech)
            self.in_speech = False
            self.speech = []
            self.voiced_run = 0
            self.split_at = None
            self.speculating = False
            if seconds - self.silence_run * FRAME_MS / 1000 < MIN_UTTERANCE_SEC:
                return None
            return utterance.astype(np.float32) / 32768.0
        if seconds > SOFT_MAX_SEC and self.split_at:
            # Continuous speech (a video, a fast talker) never yields a full
            # pause; emit up to the last natural micro-pause and keep going.
            utterance = np.concatenate(self.speech[:self.split_at])
            self.speech = self.speech[self.split_at:]
            self.split_at = None
            return utterance.astype(np.float32) / 32768.0
        return None

    def current_audio(self) -> np.ndarray | None:
        if not self.in_speech or len(self.speech) < 32:  # need ~1 s for stability
            return None
        window = self.speech[-PARTIAL_WINDOW_FRAMES:]
        return np.concatenate(window).astype(np.float32) / 32768.0


# --------------------------------------------------------------------- startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    def _load():
        load_silero()
        log.info("Prewarming whisper model %s ...", WHISPER_REPO)
        transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), language="en")
        log.info("Whisper ready.")

    asyncio.get_event_loop().run_in_executor(whisper_executor, _load)
    yield


app = FastAPI(lifespan=lifespan)


# ------------------------------------------------------------------- endpoints


class NoCacheStaticFiles(StaticFiles):
    """Static files that browsers must revalidate on every load.

    Without this, iOS home-screen web apps keep replaying a cached app.js
    long after the server has newer code — features silently "disappear"
    until the cache happens to expire. no-cache still allows ETag 304s,
    so repeat loads stay cheap on the LAN."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


@app.get("/")
async def index():
    """index.html with cache-busted asset URLs: __BUILD__ becomes the newest
    static-file mtime. Browsers do revalidate the page on navigation but can
    keep its subresources cached (observed on iOS: new HTML, old CSS/JS); a
    changed ?v= forces a fresh fetch the moment the HTML mentions it."""
    html = (STATIC_DIR / "index.html").read_text()
    stamp = max(int(p.stat().st_mtime)
                for p in STATIC_DIR.iterdir() if p.is_file())
    return HTMLResponse(html.replace("__BUILD__", str(stamp)),
                        headers={"Cache-Control": "no-cache"})


app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")

CERT_PATH = Path(__file__).parent / "certs" / "cert.pem"


@app.get("/cert")
async def cert():
    """Phone mode's self-signed certificate, for a one-time install on the
    phone. Until iOS fully trusts the cert, its home-screen icon fetcher
    (which runs outside the Safari session that accepted the warning)
    rejects the connection and falls back to a lettered tile."""
    if not CERT_PATH.exists():
        return JSONResponse(
            {"error": "No certificate — start phone mode first."},
            status_code=404)
    return FileResponse(CERT_PATH, media_type="application/x-x509-ca-cert",
                        filename="allklaro.pem")


def ollama_client() -> httpx.AsyncClient:
    """Factory so tests can swap in a mock transport."""
    return httpx.AsyncClient(base_url=OLLAMA_URL)


@app.get("/api/models")
async def models():
    """Installed Ollama models for the UI dropdown."""
    try:
        async with ollama_client() as client:
            r = await client.get("/api/tags", timeout=5)
            r.raise_for_status()
            installed = r.json().get("models", [])
    except Exception as exc:
        log.warning("Ollama unreachable: %s", exc)
        return {"models": [], "default": DEFAULT_MODEL,
                "error": f"Cannot reach Ollama at {OLLAMA_URL} — is `ollama serve` running?"}
    # Skip embedding models; they can't translate.
    kept = [m for m in installed if "embed" not in m["name"]]
    return {"models": [m["name"] for m in kept], "default": DEFAULT_MODEL,
            # Sizes let the UI suggest the smallest model as the fast draft.
            "sizes": {m["name"]: m.get("size", 0) for m in kept}}


def build_markdown(items: list[dict], summary: str = "") -> str:
    """Render a finished conversation as Markdown for export."""
    lines = ["# Conversation transcript", ""]
    for item in items:
        time = item.get("time", "")
        speaker = item.get("speaker", "")
        source = item.get("source", "?").upper()
        who = f" {speaker}" if speaker else ""
        lines.append(f"**[{time}]{who} ({source}):** {item.get('text', '')}")
        translations = item.get("translations")
        if not isinstance(translations, dict):
            translations = {item.get("target", "?"): item.get("translation", "")}
        for target, text in translations.items():
            lines.append(f"→ **({target.upper()}):** {text}")
        lines.append("")
    if summary:
        lines += ["## Summary", "", summary, ""]
    return "\n".join(lines)


@app.post("/api/export")
async def export(payload: dict):
    items = payload.get("items", [])
    if not isinstance(items, list):
        items = []
    summary = payload.get("summary", "")
    if not isinstance(summary, str):
        summary = ""
    return {"markdown": build_markdown(items, summary)}


@app.post("/api/correction")
async def correction(payload: dict):
    """Persist a user-edited translation for future few-shot retrieval."""
    source, target = payload.get("source"), payload.get("target")
    text = payload.get("text")
    corrected = payload.get("corrected")
    if (source not in LANG_NAMES or target not in LANG_NAMES
            or source == target
            or not isinstance(text, str) or not text.strip()
            or not isinstance(corrected, str) or not corrected.strip()):
        return {"error": "Invalid correction."}
    model_translation = payload.get("model_translation")
    save_correction({
        "source": source, "target": target, "text": text.strip(),
        "corrected": corrected.strip(),
        "model_translation": model_translation.strip()
        if isinstance(model_translation, str) else "",
    })
    return {"ok": True, "count": len(load_corrections())}


@app.post("/api/translate")
async def translate_api(payload: dict):
    """Stateless one-shot translation for external callers — built for an
    iOS Shortcut ("Translate with AllKlaro" in the share sheet), so reading
    a WhatsApp message needs no app switch. Same pipeline as typed input,
    minus the conversation context."""
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return {"error": "No text to translate."}
    text = text.strip()[:2000]
    mode = payload.get("mode") or "auto-de-en"
    model = payload.get("model") or DEFAULT_MODEL
    flavor = payload.get("de_flavor")
    flavor = flavor if flavor in FLAVOR_NOTES else ""
    pair = mode_pair(mode)
    detected = detect_language(text, pair) if pair else (lang_hint(mode) or "de")
    source, targets = resolve_targets(mode, detected)
    translations = {}
    for target in targets:
        out = await translate_once(text, source, target, model, flavor=flavor)
        if out is None:
            return {"error": "Translation failed — are Ollama and the model "
                             f"{model!r} available?"}
        if not (target == "de" and flavor):  # guard assumes standard German
            out, _ = await enforce_agreement(text, source, target, model,
                                             None, out)
        translations[target] = out
    return {"source": source, "target": targets[0],
            "translation": translations[targets[0]],
            "translations": translations}


# ---------------------------------------------------------- word lookup

_wikt_conns: dict[str, sqlite3.Connection] = {}


def wiktionary_conn(lang: str) -> sqlite3.Connection | None:
    conn = _wikt_conns.get(lang)
    if conn is None and WIKTIONARY_PATHS[lang].exists():
        conn = sqlite3.connect(WIKTIONARY_PATHS[lang], check_same_thread=False)
        _wikt_conns[lang] = conn
    return conn


def wiktionary_entries(conn, word: str, limit: int = LOOKUP_LIMIT) -> list[dict]:
    """All entries whose spelling matches `word` case-insensitively;
    exact-case matches first (Tisch the noun before tisch- anything),
    then base words before inflected forms."""
    rows = conn.execute(
        "SELECT word, pos, gender, ipa, plural, senses, lemma FROM entries "
        "WHERE word_lc = ? ORDER BY (word = ?) DESC, (lemma = '') DESC LIMIT ?",
        (word.lower(), word, limit)).fetchall()
    return [{"word": w, "pos": pos, "gender": g, "ipa": ipa, "plural": pl,
             "senses": json.loads(senses), "lemma": lemma}
            for w, pos, g, ipa, pl, senses, lemma in rows]


@app.get("/api/lookup")
async def lookup(word: str = "", lang: str = "de"):
    """Dictionary data for a long-pressed word in the feed."""
    word = word.strip()
    if lang not in WIKTIONARY_PATHS or not word or len(word) > 64:
        return {"error": "Invalid lookup."}
    conn = wiktionary_conn(lang)
    if conn is None:
        return {"error": f"No {lang} dictionary built yet — run: "
                         f"uv run python build_wiktionary_lexicon.py {lang}"}
    entries = wiktionary_entries(conn, word)
    # Inflected forms chain to their base word ("ging" -> gehen), so the
    # popup shows the lemma's meaning under the form's description.
    fetched = {word.lower()}
    for e in list(entries):
        lemma = e["lemma"]
        if lemma and lemma.lower() not in fetched:
            fetched.add(lemma.lower())
            entries.extend(le for le in wiktionary_entries(conn, lemma, limit=2)
                           if not le["lemma"])
    return {"entries": entries[:LOOKUP_LIMIT + 2]}


SUMMARY_PROMPT = (
    "You summarize bilingual spoken conversations for a language learner. "
    "The user message is a conversation transcript, one utterance per line, "
    "prefixed with its language. Write, in English:\n"
    "1. **Summary** — the main points, decisions, and follow-ups.\n"
    "2. **Vocabulary** — up to 10 useful words or phrases that appeared in "
    "the non-English language, each with its English translation, chosen for "
    "review value.\n"
    "Output only these two sections in Markdown."
)


@app.post("/api/summarize")
async def summarize(payload: dict):
    items = payload.get("items", []) or []
    model = payload.get("model") or DEFAULT_MODEL
    convo = "\n".join(f"[{i.get('source', '?').upper()}] {i.get('text', '')}"
                      for i in items if isinstance(i, dict) and i.get("text"))
    if not convo:
        return {"error": "Nothing to summarize yet."}
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SUMMARY_PROMPT},
                     {"role": "user", "content": convo}],
        "stream": False,
        "options": {"temperature": 0.2},
        "keep_alive": "60m",
    }
    if any(k in model.lower() for k in ("qwen3", "deepseek-r1", "gpt-oss")):
        body["think"] = False
    try:
        async with ollama_client() as client:
            r = await client.post("/api/chat", json=body, timeout=300)
            if r.status_code != 200:
                return {"error": f"Ollama error: {r.text[:200]}"}
            return {"summary": r.json().get("message", {}).get("content", "").strip()}
    except Exception as exc:
        log.warning("summarize failed: %s", exc)
        return {"error": f"Cannot reach Ollama at {OLLAMA_URL}."}


# ------------------------------------------------------------------ translation

LANG_NAMES = {"de": "German", "en": "English", "es": "Spanish"}

# Target-language grammar notes appended to the system prompt. Generic
# "be careful with grammar" instructions measurably do nothing (gemma3:12b
# still writes "der Margarita"); stating the concrete rule fixes it — the
# model lacks the lexical fact, not the diligence. Keep these static per
# direction so Ollama's prefix cache survives.
GRAMMAR_NOTES = {
    "de": ("German grammar notes: names of drinks and cocktails ending in "
           "-a (Margarita, Sangria, Piña Colada) are feminine — die "
           "Margarita, eine Sangria — unless a dictionary note below says "
           "otherwise. Decline articles and adjectives to match the noun's "
           "gender."),
}


# Optional German output flavor (WhatsApp friends write Berlinerisch and
# Hessisch; replying in kind is half the fun). Static per connection and
# inserted before the dynamic prompt parts, so prefix caching survives.
FLAVOR_NOTES = {
    "berlin": (
        "Write the German translation in casual Berlin dialect "
        "(Berlinerisch), the way a Berliner actually talks: ick (ich), "
        "dit/det (das), wat (was), ooch (auch), keen/keene (kein/keine), "
        "nüscht (nichts), jut (gut), j- for g- (jehen, jenau, Jeld). "
        "Dialect flavor, not parody — keep it readable and never change "
        "the meaning."),
    "hessian": (
        "Write the German translation in casual Hessian dialect "
        "(Hessisch), the way people around Frankfurt talk: isch (ich), "
        "net (nicht), aach (auch), ebbes (etwas), gell as a tag question, "
        "gude as a greeting, babbeln (reden), -sch for -ch (isch, disch). "
        "Dialect flavor, not parody — keep it readable and never change "
        "the meaning."),
    "worms": (
        "Write the German translation in Wormser Platt, the city dialect "
        "of Worms (Rheinhessisch, Rhine Franconian — its signature word "
        "is nää for nein): nää (nein — always nää, never nee), isch "
        "(ich), net (nicht), aach (auch), ebbes (etwas), mer (wir), alla "
        "(well then / los), redde/babbeln (reden), gugge (schauen), Woi "
        "(Wein), Grumbeere (Kartoffeln), "
        "-scht for -st (bischt, hoscht), dropped final -n on verbs (mer "
        "mache, se gehe), -che diminutives. Dialect flavor, not parody — "
        "keep it readable and never change the meaning."),
}


def translation_messages(text: str, source: str, target: str,
                         history: list[dict] | None = None,
                         flavor: str | None = None) -> list[dict]:
    """Static system prompt + history as chat turns.

    Keeping the system prompt constant and prepending history as normal
    user/assistant pairs lets Ollama's prefix cache skip re-processing
    everything but the new sentence. History entries from the opposite
    direction of the same pair are flipped (their translation becomes the
    "user" side), so both halves of the conversation provide context.
    """
    src, tgt = LANG_NAMES[source], LANG_NAMES[target]
    system = (
        f"You are a professional simultaneous interpreter for a live spoken "
        f"conversation between a {src} speaker and a {tgt} speaker. Translate "
        f"each user message from {src} to {tgt}. Output ONLY the {tgt} "
        f"translation - no explanations, no quotation marks, no notes. "
        f"Preserve the speaker's tone and register. Spoken language may "
        f"contain fillers or small transcription errors; translate the "
        f"intended meaning naturally. Earlier exchanges are shown for "
        f"context; use them to resolve pronouns and topic."
    )
    if target in GRAMMAR_NOTES:
        system += "\n\n" + GRAMMAR_NOTES[target]
    if target == "de" and flavor in FLAVOR_NOTES:
        system += "\n\n" + FLAVOR_NOTES[flavor]
    glossary = load_glossary()
    if glossary:
        system += ("\n\nGlossary — use these exact translations where "
                   "relevant:\n" + "\n".join(glossary))
    # Dynamic additions go last so the static prefix above stays cacheable.
    note = gender_notes(text, target)
    if note:
        system += "\n\n" + note
    dialect = dialect_notes(text, source)
    if dialect:
        system += "\n\n" + dialect
    # User-corrected translations similar to this sentence are shown as
    # few-shot pairs. The system prompt must mark them as authoritative:
    # unlabeled example turns read as mere history and gemma3 ignores their
    # word choices (verified against the real model).
    examples = relevant_corrections(text, source, target)
    if examples:
        system += (
            f"\n\nThe first {len(examples)} exchange(s) below are "
            f"translations the user has manually corrected. They are "
            f"authoritative: when the same words or expressions come up "
            f"again, reuse the corrected terminology and phrasing exactly."
        )
    msgs = [{"role": "system", "content": system}]
    for ex in examples:
        msgs.append({"role": "user", "content": ex["text"]})
        msgs.append({"role": "assistant", "content": ex["corrected"]})
    for h in history or []:
        if h.get("source") == source and h.get("target") == target:
            pair = (h["text"], h["translation"])
        elif h.get("source") == target and h.get("target") == source:
            pair = (h["translation"], h["text"])  # flipped: reverse direction
        else:
            continue
        msgs.append({"role": "user", "content": pair[0]})
        msgs.append({"role": "assistant", "content": pair[1]})
    msgs.append({"role": "user", "content": text})
    return msgs


async def safe_send(ws: WebSocket, payload: dict) -> bool:
    """Send on a socket that may already be closed; returns False if it was."""
    try:
        await ws.send_json(payload)
        return True
    except Exception:
        return False


async def stream_translation(ws: WebSocket, uid: int, text: str, source: str,
                             target: str, model: str,
                             history: list[dict] | None = None,
                             flavor: str | None = None) -> str | None:
    """Streams deltas to the client; returns the full translation, or None on error.

    The caller sends the closing `translation_done` message (with metrics).
    """
    payload = {
        "model": model,
        "messages": translation_messages(text, source, target, history, flavor),
        "stream": True,
        "options": {"temperature": 0.0},
        "keep_alive": "60m",
    }
    # Disable chain-of-thought on reasoning models so tokens are the translation.
    if any(k in model.lower() for k in ("qwen3", "deepseek-r1", "gpt-oss")):
        payload["think"] = False

    collected: list[str] = []
    try:
        async with ollama_client() as client:
            async with client.stream("POST", "/api/chat",
                                     json=payload, timeout=120) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode()
                    await safe_send(ws, {"type": "error", "id": uid,
                                         "message": f"Ollama error: {body[:200]}"})
                    return None
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        collected.append(delta)
                        if not await safe_send(ws, {"type": "translation_delta",
                                                    "id": uid, "target": target,
                                                    "text": delta}):
                            return None  # client left; stop burning tokens
                    if chunk.get("done"):
                        break
        return "".join(collected)
    except httpx.ConnectError:
        await safe_send(ws, {"type": "error", "id": uid,
                             "message": f"Cannot reach Ollama at {OLLAMA_URL} — "
                                        "is `ollama serve` running?"})
    except Exception as exc:
        log.exception("translation failed")
        await safe_send(ws, {"type": "error", "id": uid,
                             "message": f"Translation failed: {exc}"})
    return None


async def translate_once(text: str, source: str, target: str, model: str,
                         history: list[dict] | None = None,
                         revise: tuple[str, str] | None = None,
                         flavor: str | None = None) -> str | None:
    """One-shot, non-streaming translation; None on any failure.

    Used for the behind-the-scenes refinement pass, which replaces the fast
    draft wholesale — streaming deltas would rewrite text mid-read. With
    `revise=(candidate, issues)` the model is instead asked to correct its
    own earlier translation, with the offending facts spelled out.
    """
    messages = translation_messages(text, source, target, history, flavor)
    if revise:
        candidate, issues = revise
        messages.append({"role": "assistant", "content": candidate})
        messages.append({"role": "user", "content": (
            f"Your translation contains grammar errors: {issues} "
            f"Output ONLY the corrected {LANG_NAMES[target]} translation "
            f"of that same sentence.")})
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0},
        "keep_alive": "60m",
    }
    if any(k in model.lower() for k in ("qwen3", "deepseek-r1", "gpt-oss")):
        body["think"] = False
    try:
        async with ollama_client() as client:
            r = await client.post("/api/chat", json=body, timeout=120)
            if r.status_code != 200:
                log.warning("refinement failed: %s", r.text[:200])
                return None
            return r.json().get("message", {}).get("content", "").strip() or None
    except Exception as exc:
        # The draft is already on screen; a failed refinement is not an error
        # worth interrupting the conversation for.
        log.warning("refinement failed: %s", exc)
        return None


# Optional second opinion from a local LanguageTool server (Java).
# Opt-in: install with `uv sync --extra lt` and run with ALLKLARO_LT=1.
LT_ENABLED = os.environ.get("ALLKLARO_LT") == "1"
LT_LANGS = {"de": "de-DE", "es": "es"}
_lt_tools: dict = {}


def _lt_tool(target: str):
    if target not in _lt_tools:
        try:
            import language_tool_python
            _lt_tools[target] = language_tool_python.LanguageTool(
                LT_LANGS[target])
            log.info("LanguageTool ready for %s", target)
        except Exception as exc:
            log.warning("LanguageTool unavailable (%s); disabled for %s",
                        exc, target)
            _lt_tools[target] = None
    return _lt_tools[target]


def languagetool_issues(text: str, target: str) -> list[str]:
    """Grammar-category LanguageTool findings (blocking — run in executor).
    Filtered to ruleIssueType 'grammar' for precision; style/typography
    opinions must not trigger re-translations."""
    if not LT_ENABLED or target not in LT_LANGS:
        return []
    tool = _lt_tool(target)
    if tool is None:
        return []
    try:
        matches = tool.check(text)
    except Exception as exc:
        log.warning("LanguageTool check failed: %s", exc)
        return []
    issues = []
    for m in matches:
        # German agreement rules ship as 'uncategorized' — accept them by
        # rule id; otherwise only grammar-typed findings (never style).
        agreementish = any(k in (m.rule_id or "")
                           for k in ("AGREEMENT", "KONGRUENZ"))
        if m.rule_issue_type != "grammar" and not agreementish:
            continue
        bad = text[m.offset:m.offset + m.error_length]
        fix = f' (suggestion: "{m.replacements[0]}")' if m.replacements else ""
        issues.append(f'"{bad}": {m.message}{fix}')
    return issues[:4]


async def _combined_issues(text: str, target: str) -> list[str]:
    issues = agreement_issues(text, target)
    if LT_ENABLED:
        issues += await asyncio.get_running_loop().run_in_executor(
            None, languagetool_issues, text, target)
    return issues


async def enforce_agreement(text: str, source: str, target: str, model: str,
                            context: list[dict] | None,
                            candidate: str) -> tuple[str, bool]:
    """Returns (final text, changed). If the candidate contains impossible
    article/gender combinations (or LanguageTool findings, when enabled),
    re-ask once with the facts stated; the retry is used only if it
    verifies clean — otherwise keep the original."""
    issues = await _combined_issues(candidate, target)
    if not issues:
        return candidate, False
    log.info("agreement retry (%s): %s", target, " ".join(issues))
    fixed = await translate_once(text, source, target, model, context,
                                 revise=(candidate, " ".join(issues)))
    if (fixed and fixed != candidate
            and not await _combined_issues(fixed, target)):
        return fixed, True
    return candidate, False


# ------------------------------------------------------------------- directions


def resolve_direction(mode: str, detected: str) -> tuple[str, str]:
    """Returns (source, target) language codes.

    Modes: "auto-<a>-<b>" translates within the pair, direction driven by the
    detected language (unrecognized detections default to a -> b); "<src>-<tgt>"
    forces a direction. "auto" is a legacy alias for "auto-de-en".
    """
    if mode == "auto":
        mode = "auto-de-en"
    parts = mode.split("-")
    if parts[0] == "auto" and len(parts) == 3:
        a, b = parts[1], parts[2]
        if a in LANG_NAMES and b in LANG_NAMES:
            return (b, a) if detected == b else (a, b)
    elif len(parts) == 2:
        src, tgt = parts
        if src in LANG_NAMES and tgt in LANG_NAMES and src != tgt:
            return src, tgt
    return "de", "en"  # unrecognized mode string


def resolve_targets(mode: str, detected: str) -> tuple[str, list[str]]:
    """Like resolve_direction, plus multi-target modes "<src>-<t1>+<t2>"."""
    if "+" in mode:
        src, _, rest = mode.partition("-")
        targets = rest.split("+")
        if (src in LANG_NAMES and len(set(targets)) == len(targets)
                and all(t in LANG_NAMES and t != src for t in targets)):
            return src, targets
        return "de", ["en"]
    source, target = resolve_direction(mode, detected)
    return source, [target]


def lang_hint(mode: str) -> str | None:
    """Whisper language hint: pinned for forced directions, free for auto."""
    src = mode.split("-")[0]
    return src if src in LANG_NAMES else None


def mode_pair(mode: str) -> tuple[str, str] | None:
    """The (a, b) pair of an "auto-a-b" mode; None for forced directions."""
    m = "auto-de-en" if mode == "auto" else mode
    parts = m.split("-")
    if (parts[0] == "auto" and len(parts) == 3
            and all(p in LANG_NAMES for p in parts[1:])):
        return parts[1], parts[2]
    return None


def normalize_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


# Typed text skips Whisper, so auto modes need their own language detection.
STOPWORDS = {
    "de": {"der", "die", "das", "und", "ist", "nicht", "ich", "du", "wir",
           "ihr", "sie", "es", "ein", "eine", "einen", "dem", "den", "mit",
           "für", "auf", "haben", "hat", "war", "sind", "bitte", "danke",
           "schon", "noch", "auch", "aber", "oder", "wenn", "wie", "wo",
           "was", "warum", "kann", "können", "möchte", "geht", "gut"},
    "en": {"the", "and", "is", "are", "was", "were", "i", "you", "we",
           "they", "it", "a", "an", "to", "of", "in", "on", "with", "for",
           "have", "has", "had", "please", "thanks", "this", "that", "what",
           "why", "how", "where", "when", "not", "can", "could", "would",
           "like", "good"},
    "es": {"el", "la", "los", "las", "y", "es", "no", "yo", "tú", "usted",
           "un", "una", "unos", "unas", "de", "en", "con", "para", "por",
           "que", "qué", "cómo", "dónde", "cuándo", "gracias", "está",
           "están", "ser", "estar", "pero", "si", "como", "puedo", "quiero",
           "bien", "muy"},
}
CHAR_HINTS = {"de": "äöüß", "es": "ñ¿¡áéíóú"}


def detect_language(text: str, candidates: tuple[str, str] = ("de", "en")) -> str:
    """Stopword/charset scorer for typed text; ties go to the first candidate."""
    lowered = text.lower()
    words = re.findall(r"[^\W\d_]+", lowered)
    best, best_score = candidates[0], -1
    for lang in candidates:
        score = sum(1 for w in words if w in STOPWORDS.get(lang, ()))
        score += sum(2 for c in lowered if c in CHAR_HINTS.get(lang, ""))
        if score > best_score:
            best, best_score = lang, score
    return best


def is_echo(a: str, b: str) -> bool:
    """True when two normalized transcripts are near-duplicates — the same
    speech captured on both channels (mic picked up the call speakers)."""
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() > 0.85


SPEAKERS = {0: "you", 1: "them"}


def split_tagged(data: bytes) -> tuple[int, bytes]:
    """Audio frames: even length = untagged channel 0; odd = 1 tag byte + PCM."""
    if len(data) % 2:
        tag = data[0] if data[0] in SPEAKERS else 0
        return tag, data[1:]
    return 0, data


# ------------------------------------------------------------------- websocket


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()
    channels: dict[int, dict] = {}   # tag -> {"vad": VadSession, "residual": ...}
    history: deque = deque(maxlen=HISTORY_TURNS)
    last_final: dict[str, str] = {}  # language -> last final text (whisper prompt)
    mode = "auto-de-en"
    model = DEFAULT_MODEL
    draft_model = None               # fast first-pass model; None = single-pass
    de_flavor = ""                   # "" standard | "berlin" | "hessian"
    corrected_uids: set[int] = set()  # user edits that refinement must not undo
    pause_frames = END_SILENCE_FRAMES
    uid = 0
    last_partial = 0.0
    busy = False                     # a transcription is in flight
    # Last finalized utterance, for merging sentence fragments cut mid-pause.
    prev = None                      # {"uid","text","source","speaker","t_end"}
    recent_finals: deque = deque(maxlen=8)  # for cross-channel echo dedupe

    def channel(tag: int) -> dict:
        if tag not in channels:
            channels[tag] = {"vad": VadSession(make_scorer(), pause_frames),
                             "residual": np.zeros(0, dtype=np.int16)}
        return channels[tag]

    def whisper_prompt() -> str | None:
        parts = []
        terms = glossary_whisper_terms()
        if terms:
            parts.append(terms + ".")
        hint = lang_hint(mode)
        if hint and last_final.get(hint):
            parts.append(last_final[hint])
        return " ".join(parts)[:400] or None

    def auto_pair() -> tuple[str, str] | None:
        return mode_pair(mode)

    async def handle_utterance(audio: np.ndarray, my_uid: int, speaker: str,
                               spec_task=None):
        nonlocal busy, prev
        busy = True
        t0 = loop.time()
        try:
            if spec_task is not None:
                result = await spec_task  # transcription began during the pause
            else:
                result = await loop.run_in_executor(
                    whisper_executor, transcribe, audio, lang_hint(mode),
                    whisper_prompt())
            detected = result.get("language", "de")
            pair = auto_pair()
            if pair and detected not in pair:
                # Whisper picked a language outside the active pair (e.g. Dutch
                # for German speech) — the decode itself is wrong. Redo it
                # pinned to the pair's primary language.
                result = await loop.run_in_executor(
                    whisper_executor, transcribe, audio, pair[0],
                    whisper_prompt())
                detected = pair[0]
            t1 = loop.time()
            text = clean_transcript(result)
            if not text or HALLUCINATION_RE.match(text):
                await safe_send(ws, {"type": "discard", "id": my_uid})
                return
            norm = normalize_text(text)
            if len(norm) >= ECHO_MIN_CHARS and any(
                    r["speaker"] != speaker and t0 - r["t"] < ECHO_WINDOW_SEC
                    and is_echo(norm, r["norm"]) for r in recent_finals):
                # Same speech heard on the other channel moments ago: the mic
                # picked up the call audio (or vice versa). Drop the echo.
                await safe_send(ws, {"type": "discard", "id": my_uid})
                return
            source, targets = resolve_targets(mode, detected)

            # Merge with the previous utterance when it was cut mid-sentence
            # (no terminal punctuation) and this one resumed right after —
            # rejoins German verb-final clauses split by a short pause.
            replaces = None
            gap = (t0 - len(audio) / SAMPLE_RATE - prev["t_end"]) if prev else 99
            if (prev and prev["speaker"] == speaker and prev["source"] == source
                    and gap < MERGE_GAP_SEC
                    and len(prev["text"]) < MERGE_MAX_CHARS
                    and not SENTENCE_END_RE.search(prev["text"])):
                text = prev["text"] + " " + text
                replaces = prev["uid"]
                if history and history[-1].get("uid") == replaces:
                    history.pop()

            last_final[source] = text[-200:]
            prev = {"uid": my_uid, "text": text, "source": source,
                    "speaker": speaker, "t_end": t0}
            recent_finals.append({"norm": normalize_text(text),
                                  "speaker": speaker, "t": t0})
            final_msg = {"type": "final", "id": my_uid, "text": text,
                         "source": source, "target": targets[0],
                         "targets": targets, "speaker": speaker}
            if replaces is not None:
                final_msg["replaces"] = replaces
            await safe_send(ws, final_msg)
            await run_translations(my_uid, text, source, targets, t0, t1)
        except Exception as exc:
            log.exception("transcription failed")
            await safe_send(ws, {"type": "error", "id": my_uid,
                                 "message": f"Transcription failed: {exc}"})
        finally:
            busy = False

    async def run_translations(my_uid, text, source, targets, t0, t1):
        # Two-tier translation: stream the fast draft model first so text
        # appears immediately, then re-translate with the main model
        # behind the scenes and swap in its (better) answer. Either way,
        # the final text passes the declension guard (enforce_agreement)
        # and is corrected once when it trips.
        draft = draft_model if draft_model and draft_model != model else None
        context = list(history)
        translations = await asyncio.gather(
            *(stream_translation(ws, my_uid, text, source, t,
                                 draft or model, context, de_flavor)
              for t in targets))
        if all(t is not None for t in translations):
            history.append({"uid": my_uid, "source": source,
                            "target": targets[0], "text": text,
                            "translation": translations[0]})
            await safe_send(ws, {
                "type": "translation_done", "id": my_uid,
                "refining": bool(draft),
                "transcribe_ms": int((t1 - t0) * 1000),
                "translate_ms": int((loop.time() - t1) * 1000)})
            texts = {}
            for t, streamed in zip(targets, translations):
                candidate = streamed
                if draft:
                    refined = await translate_once(text, source, t,
                                                   model, context,
                                                   flavor=de_flavor)
                    if refined:
                        candidate = refined
                if t == "de" and de_flavor:
                    # The declension guard assumes standard German; dialect
                    # forms ("dit Haus", "keene") would trip it and get
                    # "corrected" back to Hochdeutsch.
                    final = candidate
                else:
                    final, _ = await enforce_agreement(text, source, t, model,
                                                       context, candidate)
                if final != streamed:
                    texts[t] = final
            # The final text becomes the conversation context — unless
            # the user already corrected this utterance.
            if targets[0] in texts and my_uid not in corrected_uids:
                for h in history:
                    if h.get("uid") == my_uid:
                        h["translation"] = texts[targets[0]]
            # With a draft, always sent (even empty) so the UI can clear
            # the "refining…" hint; single-pass sends only real changes.
            if draft or texts:
                await safe_send(ws, {"type": "translation_revised",
                                     "id": my_uid, "texts": texts})

    async def handle_text(text, my_uid):
        """Typed input: same translation pipeline, no audio machinery —
        no Whisper, no merging, no echo dedupe, no busy flag."""
        t0 = loop.time()
        try:
            pair = auto_pair()
            detected = (detect_language(text, pair) if pair
                        else lang_hint(mode) or "de")
            source, targets = resolve_targets(mode, detected)
            await safe_send(ws, {"type": "final", "id": my_uid, "text": text,
                                 "source": source, "target": targets[0],
                                 "targets": targets, "speaker": "you"})
            await run_translations(my_uid, text, source, targets, t0, t0)
        except Exception as exc:
            log.exception("typed translation failed")
            await safe_send(ws, {"type": "error", "id": my_uid,
                                 "message": f"Translation failed: {exc}"})

    async def maybe_partial():
        nonlocal last_partial, busy
        now = loop.time()
        # Never let a partial queue in front of real work: skip while an
        # utterance is being handled or a speculation is in flight.
        if busy or now - last_partial < PARTIAL_INTERVAL_SEC:
            return
        if any(c["vad"].speculating for c in channels.values()):
            return
        audio = next((c["vad"].current_audio() for c in channels.values()
                      if c["vad"].in_speech), None)
        if audio is None:
            return
        last_partial = now
        busy = True
        try:
            result = await loop.run_in_executor(
                whisper_executor, transcribe, audio, lang_hint(mode),
                whisper_prompt())
            text = clean_transcript(result)
            if text and not HALLUCINATION_RE.match(text):
                await safe_send(ws, {"type": "partial", "text": text})
        except Exception:
            log.exception("partial transcription failed")
        finally:
            busy = False

    try:
        while True:
            message = await ws.receive()
            if message.get("text") is not None:
                try:
                    cfg = json.loads(message["text"])
                except (json.JSONDecodeError, TypeError):
                    log.warning("ignoring malformed control message")
                    continue
                if isinstance(cfg, dict) and cfg.get("type") == "config":
                    mode = cfg.get("mode", mode)
                    model = cfg.get("model", model)
                    if "draft_model" in cfg:  # "" means draft pass off
                        draft_model = cfg["draft_model"] or None
                    if "de_flavor" in cfg:  # "" means standard German
                        de_flavor = (cfg["de_flavor"]
                                     if cfg["de_flavor"] in FLAVOR_NOTES else "")
                    try:
                        pause_ms = int(cfg.get("pause_ms", pause_frames * FRAME_MS))
                        pause_frames = max(200, min(2000, pause_ms)) // FRAME_MS
                    except (TypeError, ValueError):
                        pass
                    for c in channels.values():
                        c["vad"].end_silence = pause_frames
                    log.info("config: mode=%s model=%s pause=%dms",
                             mode, model, pause_frames * FRAME_MS)
                elif isinstance(cfg, dict) and cfg.get("type") == "text":
                    typed = cfg.get("text")
                    if isinstance(typed, str) and typed.strip():
                        uid += 1
                        asyncio.create_task(
                            handle_text(typed.strip()[:2000], uid))
                elif isinstance(cfg, dict) and cfg.get("type") == "correction":
                    # An edited translation also fixes the live context, so
                    # follow-up utterances build on the corrected phrasing.
                    corrected = cfg.get("corrected")
                    if isinstance(corrected, str) and corrected.strip():
                        corrected_uids.add(cfg.get("id"))
                        for h in history:
                            if (h.get("uid") == cfg.get("id")
                                    and h.get("target") == cfg.get("target")):
                                h["translation"] = corrected.strip()
                continue
            data = message.get("bytes")
            if data is None:
                break
            tag, pcm = split_tagged(data)
            ch = channel(tag)
            samples = np.frombuffer(pcm, dtype=np.int16)
            ch["residual"] = np.concatenate([ch["residual"], samples])
            while len(ch["residual"]) >= FRAME_SAMPLES:
                frame = ch["residual"][:FRAME_SAMPLES]
                ch["residual"] = ch["residual"][FRAME_SAMPLES:]
                vad = ch["vad"]
                utterance = vad.feed(frame)
                if vad.early_event is not None:
                    # The pause just hit ~320 ms: transcribe speculatively so
                    # the result is ready if the pause turns out to be final.
                    early = vad.early_event
                    vad.early_event = None
                    ch["spec"] = {"len": len(early),
                                  "task": loop.run_in_executor(
                                      whisper_executor, transcribe, early,
                                      lang_hint(mode), whisper_prompt())}
                if utterance is not None:
                    spec = ch.pop("spec", None)
                    expected = (spec["len"] + (vad.end_silence
                                - EARLY_SILENCE_FRAMES) * FRAME_SAMPLES
                                if spec else -1)
                    spec_task = spec["task"] if len(utterance) == expected else None
                    uid += 1
                    await safe_send(ws, {"type": "segment_start", "id": uid,
                                         "speaker": SPEAKERS[tag]})
                    asyncio.create_task(
                        handle_utterance(utterance, uid, SPEAKERS[tag],
                                         spec_task))
            if any(c["vad"].in_speech for c in channels.values()):
                asyncio.create_task(maybe_partial())
    except WebSocketDisconnect:
        pass
