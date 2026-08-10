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
import threading
import time
import zlib
from difflib import SequenceMatcher
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np

import voiceprint
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
# 5.0, not the original 8.0, and not lower: measured, bracketed A-B-A on the
# real slice. Accumulation was 65% of first-word lag at p50, and the split
# points to cut sooner already existed — `split_at` is updated at every
# micro-pause and was then ignored until 8 s had elapsed.
#
# Against the fixed-cost stub, on the chunk-count-invariant metric (lag at
# speech-run starts, which a lower cap cannot inflate by manufacturing more
# utterances): 8.0 -> 9092 ms, 6.0 -> 8631 ms, 5.0 -> 6827 ms, 4.0 -> 6417 ms.
# 6.0 does not clear its own 366 ms control spread; 5.0 takes 2.3 s of the
# 2.8 s available at 4.0, and 4.0 pays steeply for the last 400 ms.
#
# Confirmed against live Ollama, which contradicted the expected cost. More,
# shorter chunks did not congest it: translate p50 fell 3000 -> 1744 ms and
# its worst case 14.7 s -> 4.4 s, because a shorter chunk is a shorter prompt.
# Live first-word lag fell 10444 -> 6932 ms, a larger win than the stub showed.
#
# `spec:none` went 6 -> 20 live, which looked like the cap's real cost. It is
# mostly not a cost at all. Every spec:none is a soft_max split (pause splits
# are 19/19 and 15/15 hits), and `specs_shed` is 0 in both arms, so nothing was
# lost to the backlog. Of the 20, thirteen are chunks *longer* than the cap:
# speech ran past 5 s with no dip, so the first micro-pause set `split_at` and
# fired the cut on the same frame, at MICRO_PAUSE_FRAMES. There is no dead time
# there for a speculation to have filled — it would submit the same audio at
# the same instant. The other seven cut at a stale `split_at` left by a 6-9
# frame dip; those do leave dead time (p50 1.45 s against a 0.68 s decode), but
# filling it means speculating at every dip, and that is the redundant decoding
# the backlog gates exist to shed. `queue_ms` p50 is 0 in every group, so the
# whole mechanism is worth a few hundred ms per utterance at best. Refines also
# shed more (37 -> 49), but refine_ms sits at its 10 s timeout in both arms, so
# that is shedding work already failing. See tests/test_vad.py for the geometry.
SOFT_MAX_SEC = float(os.environ.get("ALLKLARO_SOFT_MAX_SEC", "5.0"))
MICRO_PAUSE_FRAMES = 6             # ~190 ms dip = natural split point in
                                   # continuous speech (videos, fast talkers)
PARTIAL_WINDOW_FRAMES = 190        # live partials look at the last ~6 s only
PARTIAL_INTERVAL_SEC = 2.0         # how often to emit live partial transcripts

# Backlog thresholds — see docs/findings/real-conversation-pace.md.
# Measured on a real 54-minute conversation: the pipeline decodes ~2.7x more
# audio than exists (rolling partials + speculations), which puts the single
# Whisper thread at ~1.11x realtime against a 1.0x budget. An 11% overload does
# not degrade, it queues without bound: the queue reached 72 and first-word lag
# 91 s. All of that surplus is *optional* work, so shedding it by backlog depth
# restores the budget without ever dropping a real utterance.
# PARTIAL_MAX_QUEUE was 1 for one measured run: the queue is legitimately
# non-empty most of the time even when healthy (89 of 94 chunks at depth <= 5),
# so a threshold of 1 shed 2672 partials and left the screen blank while people
# were still talking — trading one failure for another. It wants to say "we are
# behind", not "we are working".
SPEC_MAX_QUEUE = 2                 # no speculative decodes past this depth
PARTIAL_MAX_QUEUE = 3              # no live partials past this depth
REFINE_MAX_QUEUE = 2               # skip the second-pass refine past this depth
# The refine pass is a background improvement to text the user can already
# read, so it must never be able to stall the pipeline. Measured: pointing
# draft and main at two different Ollama models made every refine take >120 s
# (model swapping), and because lag was clocked after it, cards that appeared
# in ~13 s were reported as 125 s. Bound it, and drop it once the utterance is
# old enough that better wording no longer matters.
# Overridable for the same reason SOFT_MAX_SEC is: the timeout *censors* the
# thing it is measuring. The traces could only ever say "> 10 s", never how
# much more, so the pass looked unsalvageable when it was not.
#
# Uncensored at 60 s on the real slice: the median attempted refine is 4.6 s,
# and the ones a 10 s ceiling was killing need 14-17 s (max 16.9 s). Nothing
# came close to 60 s, so these are slow refines, not hung ones — and killing
# them threw away work that was nearly done.
#
# 20 s is set on that mechanism, NOT on a demonstrated lag win. Bracketed
# A-B-A on demo4, which is what a single pair could not show:
#   arm       landed   killed         first-word lag p50
#   A1 10 s   19       6 (24% of att) 11798 ms
#   B  20 s   22       2 ( 8% of att) 11296 ms
#   A2 10 s   10       9 (47% of att) 14355 ms
# The two identical control arms disagree by 23 points of kill-rate and 2556 ms
# of lag, so the control spread swallows almost everything: landed refines
# (+7.5 vs a spread of 9) and lag (-1781 vs 2556) are both INSIDE noise. Only
# kill-rate clears, and barely (27 vs 23) — which is close to tautological,
# since a higher ceiling mechanically kills fewer things that exceed it.
# So: this setting is justified by "a refine at 14-17 s is nearly finished and
# throwing it away wastes the capacity already spent", not by any measured
# improvement. Nothing here shows it costs anything either. Do not quote a lag
# benefit for it, and note the rig drifts *within* a session (A2 is much worse
# than A1), so arm order matters and even paired arms are not safe.
REFINE_TIMEOUT_SEC = float(os.environ.get("ALLKLARO_REFINE_TIMEOUT_SEC", "20"))
REFINE_MAX_AGE_SEC = 20.0
# ...and the same for the translation backlog. `whisper_pending` says nothing
# about how many utterances are waiting on Ollama, which is the queue the next
# card actually sits in once transcription keeps up. 2 = one translating and
# one waiting; past that, a background rewrite of text already on screen is
# worth less than the card nobody has seen yet.
REFINE_MAX_IN_FLIGHT = int(os.environ.get("ALLKLARO_REFINE_MAX_IN_FLIGHT", "2"))
# The running gist shown at the top of the screen. It is *folded* forward --
# each refresh sees the previous gist plus only what was said since -- so its
# cost is flat over a 54-minute call instead of growing with the transcript.
# Everything here exists to stop it becoming a second refine pass: measured on
# the real recording, an ungated background job on this Ollama loses to the
# translation backlog on most utterances and takes capacity from the cards
# someone is actually waiting for. One refresh a minute, never while the
# pipeline is behind, and bounded when it runs.
GIST_INTERVAL_SEC = float(os.environ.get("ALLKLARO_GIST_INTERVAL_SEC", "60"))
GIST_MAX_QUEUE = 2                 # no refresh past this Whisper depth
GIST_MAX_IN_FLIGHT = 2             # ...or this translation backlog
GIST_TIMEOUT_SEC = 20.0            # longer than a refine: nobody is waiting
GIST_MAX_LINES = 60                # utterances folded in one refresh
GIST_MAX_PENDING = 200             # backlog kept if refreshes keep failing
# A reconnect hands the gist back from the client, which is the only party that
# survives one. Bounded because it re-enters the fold prompt: the gist is three
# short lines by design, so anything near this is already malformed, and an
# unbounded string here would be a client-controlled prompt of any length.
GIST_SEED_MAX_CHARS = 2000
# Yielding to the backlog is right, but yielding *forever* is not a policy, it
# is the feature not existing. Measured on 240 s of the real recording at
# 25:00: `in_flight` sat at 3-6 for the whole slice and touched 2 only during
# the drain, so with the idle gate alone the gist never refreshed once. Past
# this long without one, take the turn anyway — a summary that never appears
# is worse than one utterance's wait, and at one fold a minute this costs a
# fraction of what the per-utterance refine pass does.
GIST_MAX_STALE_SEC = float(os.environ.get("ALLKLARO_GIST_MAX_STALE_SEC", "180"))
# The on-demand "improve this card" tap. This is the refine pass with its two
# handicaps removed: no backlog to yield to, and no deadline to beat. Both of
# those exist because the refine competes with the card nobody has read yet —
# and neither applies when the user has asked for this one and is watching it.
# The offline comparison measured the main model's advantage under exactly
# these conditions, which is why the tap is where that advantage is reachable.
IMPROVE_MAX_IN_FLIGHT = 2          # taps served at once; the rest are told to wait
# Generous rather than absent: nobody is racing this, but a wedged Ollama must
# not leave the button spinning for the rest of the conversation.
IMPROVE_TIMEOUT_SEC = float(os.environ.get("ALLKLARO_IMPROVE_TIMEOUT_SEC", "90"))
# "What did they just say?" — the same tap widened from one card to a window.
# Item 12 is the reason it exists: after the item 8 merge, 41% of cards are
# still fragments, because merging only reaches across a 2 s gap and a thought
# routinely spans more. A fragment cannot be translated correctly alone, which
# item 8 already established; this rejoins a whole stretch on demand and
# translates it as one passage.
#
# Which cards make up the window is decided in `recapWindow` in app.js, since
# only the client knows when each card reached the screen — deliberately not
# mirrored here, because a second copy of "15 seconds" is a number that drifts.
# The server sees the joined passage and bounds it, having no way to count
# cards in it and no reason to trust a client about length.
RECAP_MAX_CHARS = 2000             # a bounded prompt even for a fast talker
RECAP_TIMEOUT_SEC = float(os.environ.get("ALLKLARO_RECAP_TIMEOUT_SEC", "90"))
# A visible break when the voice changes, and deliberately nothing more — no
# names, no identities, no diarization. `SPEAKERS` below is a channel tag, so
# with one microphone carrying a room every card reads "you"; this at least
# says where one person stopped and another started.
#
# OFF BY DEFAULT, because measured on the real recording it does not work.
#
# On 6 synthetic voices it looked strong: same-speaker p90 0.161 against
# different-speaker p10 0.295, no overlap, precision ~1.0. On the real
# 54-minute conversation it carries almost no speaker information at all.
#
# The test that settles it uses the pipeline's own split reasons. A `soft_max`
# cut means the 5 s cap fired while someone was still talking, so the next
# utterance continues the SAME speaker; a `pause` cut is 700 ms of silence,
# where a turn actually changes. If the detector worked, continuations would
# score far lower than pauses. Over the full hour (534 continuations, 171
# pauses) at this threshold:
#
#     marked across a continuation (same speaker)  33.7%
#     marked across a pause        (turn may change) 35.1%
#
# The same number. One in three marks would land mid-sentence. No threshold
# fixes that — at 0.60 it is 11.6% against 16.4%, still overlapping, and by
# then it marks almost nothing. Synthetic voices flattered it because TTS is
# internally far more consistent, and mutually far more distinct, than three
# people round one microphone at varying distance and vocal effort.
#
# Kept rather than deleted: the measurement is the useful part, and the
# machinery is what a real speaker-embedding model would plug into. Set
# ALLKLARO_VOICE_MARKS=1 to see it anyway.
VOICE_MARKS_ON = os.environ.get("ALLKLARO_VOICE_MARKS", "") == "1"
VOICE_CHANGE_DIST = float(os.environ.get("ALLKLARO_VOICE_CHANGE_DIST", "0.35"))
HISTORY_TURNS = 6                  # recent exchanges fed to the translator
MERGE_GAP_SEC = 2.0                # resumed-within window for fragment merging
# 500, up from 300 (2026-08-09): the reference batch transcript's paragraphs
# run to a median of 51 words, and at 300 chars the cap was the binding
# refusal 39 times over the real hour. 500 admits a ~75-word card — the
# reference's p90 — while still bounding the retranslation prompt.
MERGE_MAX_CHARS = 500              # never grow merged utterances beyond this
# A chunk this short is an interjection — "Ja.", "Mm-hmm.", "No, no." — and
# gets absorbed into the live card instead of becoming one (see the merge
# site). Sized at 3 words: 26% of all cards over the real hour were ≤3 words,
# and Whisper's language detection on them is a coin toss, so each one both
# WAS a card and broke the chain for the speaker's real continuation — the
# double damage behind most of the count gap vs the batch reference.
ABSORB_MAX_WORDS = 3
# How long a merge waits for its predecessor's in-flight translation before
# giving up and retranslating the whole card. Dense speech emits a chunk
# every 2-4 s and a draft translate takes 1-2.5 s, so roughly half of merge
# links arrive before their base is finished — and falling back to a full
# retranslate there is what made the O(k²) storm survive the stitch: each
# fallback finished later, which made the next link miss its base too, and
# within a minute nothing stitched at all (measured: translate p50 8.7 s in
# the FIRST two minutes of the 62-min replay). Waiting instead keeps every
# link O(tail): the wait overlaps work already in flight, so the listener's
# cost is predecessor-completion + one small translate.
CHAIN_WAIT_SEC = 10.0
ECHO_WINDOW_SEC = 6.0              # cross-channel duplicate suppression window
ECHO_MIN_CHARS = 16                # never dedupe short phrases ("Genau!") —
                                   # people legitimately repeat those
STATS_INTERVAL_SEC = 0.5           # how often the pipeline overlay is updated
# Forcing a direction pins Whisper's language, so the *other* language in a
# bilingual conversation gets decoded as the forced one and comes out as a
# repetition loop ("…die Füße starete, die Füße starete"). has_phrase_loop now
# catches that text, which stops it reaching the screen but turns the utterance
# into a silent discard — the speech is still lost, just quietly. When a decode
# this size is rejected outright, decode it once more with the language free.
# Sized so ordinary silence and one-word noise never trigger a second decode:
# only a substantial decode that cleaning threw away entirely qualifies.
FORCED_REDO_MIN_CHARS = 40

# A transcript that ends mid-sentence (no terminal punctuation) is a merge
# candidate — crucial for German, where the meaning-carrying verb comes last.
SENTENCE_END_RE = re.compile(r"[.!?…]['\")\]]?\s*$")
# ...but an ellipsis is not a full stop, it is Whisper saying the speaker was
# still going. Measured over 254 real utterances: 58 (23%) end this way and 50
# of those 58 were `soft_max` splits — cut at a micro-pause mid-speech, not at
# the end of a thought. Counting them as finished stranded every one as its own
# card ("Von Gottbergs hießen die, wo man da als…" / "als Küchenbammser
# gearbeitet hat."), each translated with no sight of the other half. Only 28
# utterances end in no punctuation at all, so this is the *majority* of the
# merge opportunity, not an edge case.
ELLIPSIS_END_RE = re.compile(r"(\.\.\.|…)['\")\]]?\s*$")
# ...and a question is the opposite of both: it is finished AND it hands the
# turn to someone else, so it closes the merge window (see yields_turn).
QUESTION_END_RE = re.compile(r"\?['\")\]]?\s*$")


def looks_finished(text: str) -> bool:
    """True when an utterance reads as a completed sentence.

    Kept separate from the regex because "ends with punctuation" and "is
    finished" are not the same question, and the difference is where the
    fragment merging lives.
    """
    return bool(SENTENCE_END_RE.search(text)) and not ELLIPSIS_END_RE.search(text)


def continues_previous(text: str) -> bool:
    """True when this utterance opens mid-sentence, on casing alone.

    German capitalises every noun and every sentence start, so a chunk opening
    with a lowercase letter did not begin a sentence. English capitalises
    sentence starts too, so the signal holds there, just less strongly.

    This is the half of the evidence `looks_finished` structurally cannot see.
    That rule inspects the PREVIOUS utterance, and Whisper regularly ends a
    fragment with a full stop — "…in dem Bereich des Gartens." — while the
    verb arrives in the next chunk: "ausgestreut, damit das Wasser abläuft."
    Nothing about the first chunk says it was unfinished. The second one says
    it, and says it for free.

    Measured against 51 hand-labelled utterances: precision 0.82. Its blind
    spot is a chunk Whisper emits with no casing at all, which was 2.3% of
    German cards over the real hour and merges one card too many when it hits.
    """
    head = text.lstrip(".,!? ")
    return bool(head) and head[0].isalpha() and head[0].islower()


def yields_turn(text: str) -> bool:
    """True when an utterance ends by handing the conversation to someone
    else — in this recording, that means it ends with a question.

    This is the one systematic blind spot of `flowed_on`. A `soft_max` split
    normally guarantees the same speaker is continuing, but conversational
    volleys hand over faster than the ~190 ms micro-pause the VAD needs to
    see a boundary — so a question and its answer, two people, arrive inside
    one continuous speech run. Measured over the real hour: the shipped rule
    made 12 merges across a `?` boundary without casing evidence, ALL of
    them via `flowed_on`, and reading all 12 finds question→answer
    handovers every time ("Wie lange seid ihr unterwegs?" ‖ "Wir sind eine
    Woche nur weg."). Refusing them costs 352 → 364 cards (+3.4%) and
    removes 12 real two-speaker splices — the defect a batch transcription
    service avoids by cutting at semantic boundaries it can see with the
    whole recording in hand.

    A lowercase continuation still overrules this: Whisper sometimes puts a
    mid-sentence question mark ("oder?") in front of a clause that carries
    on, and casing is direct evidence the sentence did not end.
    """
    return bool(QUESTION_END_RE.search(text))


def flowed_on(split_reason: str | None) -> bool:
    """True when the previous utterance was cut mid-flow, not at a real pause.

    A `soft_max` split is the cap firing on continuous speech: the VAD emits up
    to the last micro-pause because the speaker never stopped long enough to
    end the utterance (VadSession). Whoever was talking is still talking, so
    the next chunk is the same person carrying on. That makes it the one
    same-speaker signal available here without diarization, which is a closed
    decision — and a same-speaker signal is exactly what merging needs, since
    the risk of joining two utterances is joining two people.

    This is what `looks_finished` and `continues_previous` both structurally
    miss. Both read the text, and Whisper punctuates and capitalises a mid-flow
    cut exactly as it would a finished sentence: "…die Klimaanlage lief." then
    "Und darunter ist so, da haben sie Kies…". Nothing in either string says
    the speaker never stopped. The split reason says it, and says it for free —
    it is already on `meta` for the stats dump.

    Measured over the real 54-minute recording, 703 utterances, against the
    batch service's 216 segments of ~39 words:

                                        cards   /min   mean words
          no merging                      703   12.9         10.7
          punctuation only                503    9.3         15.0
          + casing (was shipped)          438    8.1         17.2
          + this rule                     228    4.2         33.0
          batch service (target)          216    4.0         39.0

    Merging on elapsed time alone gets closer still (211 cards, 39 median
    words) and is the wrong trade: sampled cards splice a question onto its
    answer ("How did you do it? No. No, I don't do red eyes anymore.") and
    German onto English, because the batch service segments by turn and a
    clock cannot tell one. Under this rule 7 of 8 sampled multi-utterance
    cards are single-speaker passages.

    The cost is retranslation, not latency: a merge replaces the card and
    re-translates the grown text, so first paint is unchanged but merge
    operations over the hour go 265 -> 475. That is ~79% more translate work
    on the stage that is already this machine's bottleneck.
    """
    return split_reason == "soft_max"

WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
# Overridable so a benchmark can point the translation stage at a
# fixed-latency stub (tools/fake_ollama.py). Measured on the real recording,
# the live server's own spread — translate p50 2.9 s to 7.7 s across runs of
# identical code — is wider than any pipeline change worth making, so timing
# work needs a translation stage that costs the same every time.
OLLAMA_URL = os.environ.get("ALLKLARO_OLLAMA_URL", "http://127.0.0.1:11434")
# Pinned on every request, because the server's default is not ours to trust:
# Ollama.app injects OLLAMA_CONTEXT_LENGTH into `ollama serve` from its own
# settings UI, and at its 131072 the KV cache per model is large enough that
# draft and main evict each other — which presents as every card draft-only
# and every ✨/⏪/gist call timing out silently (PROGRESS.md item 15), or as
# translate p50 collapsing 1.1 s → 7.5 s over ~12 min of continuous speech
# (item 16). A request that says num_ctx gets a runner sized to num_ctx, so
# this holds whatever the app's setting is.
#
# 16384 is sized by the largest prompt in the app, the whole-session summary:
# a full hour of this conversation is ~8k words ≈ 11-12k tokens, so an hour
# fits with room for the reply. Everything else here — translate with 6 turns
# of history, refine, gist step, recap — is under ~1k tokens. Same value
# everywhere on purpose: one value means one runner allocation, never a
# mid-session reload because two call sites disagreed about the size.
OLLAMA_NUM_CTX = int(os.environ.get("ALLKLARO_NUM_CTX", "16384"))
# ALLKLARO_MODEL also lets the integration suite benchmark any model:
#   RUN_INTEGRATION=1 ALLKLARO_MODEL=<name> uv run pytest tests/test_integration.py
DEFAULT_MODEL = os.environ.get("ALLKLARO_MODEL", "gemma3:12b")
GLOSSARY_PATH = Path(__file__).parent / "glossary.txt"
DIALECTS_PATH = Path(__file__).parent / "dialects.txt"
# Words the listener already knows. Every other lever tried on lag divides the
# same work up differently; this is the only one that makes there be less of
# it — and for someone learning the language it is the better product anyway,
# since translating a sentence you understood removes the reason to practise.
#
# Measured over the real 54-minute recording: 42% of segments are three words
# or fewer ("Ja.", "Oh!", "Bei mir sind 18.") and 43% are fully covered by the
# 300 commonest words in the conversation. Roughly half the load on the
# bottleneck is spent on sentences the listener did not need.
#
# Off unless the file exists. One lowercase word per line, `#` comments.
KNOWN_WORDS_PATH = Path(os.environ.get(
    "ALLKLARO_KNOWN_WORDS", Path(__file__).parent / "known_words.txt"))
# Never skip a long utterance, however familiar its words: a sentence can be
# built entirely of known words and still say something the listener would not
# assemble in time. The cap is on the *sentence*, not the vocabulary.
KNOWN_SKIP_MAX_WORDS = int(os.environ.get("ALLKLARO_KNOWN_MAX_WORDS", "8"))
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

# ------------------------------------------------------------------- tracing
#
# One JSON line per finalized utterance, so "it can't keep up with a fast
# talker" becomes a number. The per-card latency in the UI only covers a
# chunk's own transcribe+translate, which stays reassuringly small even while
# the *first* words of an 8 s chunk age off-screen; these records carry the
# whole story instead: how long the audio sat accumulating, how much work was
# already queued on the single Whisper thread, and why the chunk was cut.
# `tools/trace_report.py` summarizes the file. ALLKLARO_TRACE=off disables it.
TRACE_PATH = os.environ.get("ALLKLARO_TRACE", "/tmp/allklaro-trace.jsonl")

# Transcriptions queued or running on the single Whisper thread. Depth, not
# duration, is what a fast talker actually creates.
whisper_pending = 0


def trace(record: dict) -> None:
    """Append one trace record. Instrumentation must never break a session."""
    if not TRACE_PATH or TRACE_PATH == "off":
        return
    try:
        with open(TRACE_PATH, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        log.debug("trace write failed", exc_info=True)


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


def load_dialects() -> dict[str, dict[str, tuple[str, bool, frozenset | None]]]:
    """dialects.txt: language -> {token -> (gloss, ambiguous, flavors)}.

    A "[es]"-style line switches the language section (the file starts in
    German for backward compatibility). Ambiguous entries ("? nett = net
    (nicht)") are real standard words too; they are only hinted when the
    sentence also contains an unambiguous dialect marker, or when the user
    has selected the dialect they belong to.

    A trailing "[hessian worms]" names those dialects; None means the entry
    applies to all of them. The distinction only bites for ambiguous entries:
    hinting a Rhine-Hessian reading of "mehr" to someone who told us they are
    listening to a Berliner would be a new error, not a fix."""
    try:
        mtime = DIALECTS_PATH.stat().st_mtime
    except OSError:
        _dialects_cache.update(mtime=None, map={})
        return {}
    if mtime != _dialects_cache["mtime"]:
        entries: dict[str, dict] = {}
        lang = "de"
        for line in DIALECTS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            section = re.fullmatch(r"\[(\w{2})\]", line)
            if section:
                lang = section.group(1)
                continue
            ambiguous = line.startswith("?")
            flavors = None
            tag = re.search(r"\[([\w\s,]+)\]\s*$", line)
            if tag:
                flavors = frozenset(tag.group(1).replace(",", " ").split())
                line = line[:tag.start()].rstrip()
            term, _, standard = line.lstrip("? ").partition("=")
            if term.strip() and standard.strip():
                entries.setdefault(lang, {})[term.strip().lower()] = (
                    standard.strip(), ambiguous, flavors)
        _dialects_cache.update(mtime=mtime, map=entries)
    return _dialects_cache["map"]


# ------------------------------------------------------------- known words

_known_cache = {"mtime": None, "words": frozenset()}


def load_known_words() -> frozenset:
    """The listener's own vocabulary, mtime-cached like the other lexicons."""
    try:
        mtime = KNOWN_WORDS_PATH.stat().st_mtime
    except OSError:
        _known_cache.update(mtime=None, words=frozenset())
        return frozenset()
    if mtime != _known_cache["mtime"]:
        words = {ln.strip().lower()
                 for ln in KNOWN_WORDS_PATH.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.startswith("#")}
        _known_cache.update(mtime=mtime, words=frozenset(words))
    return _known_cache["words"]


def is_already_understood(text: str, source: str) -> bool:
    """True when this utterance is not worth spending the bottleneck on.

    Deliberately conservative, because the two errors are not symmetric: a
    needless translation costs a little time, while a skipped one the listener
    actually needed costs them the sentence. So it must be short, *entirely*
    covered by the known list, and in the language being learned — an English
    utterance is not something a German learner is practising.
    """
    known = load_known_words()
    if not known or source != "de":
        return False
    words = re.findall(r"[^\W\d_]+", text.lower())
    if not words or len(words) > KNOWN_SKIP_MAX_WORDS:
        return False
    return all(w in known for w in words)


def dialect_markers(text: str, source: str,
                    flavor: str | None = None) -> list[str]:
    """Dialect words in `text`, lowercased, for colouring the source on screen.

    Unambiguous entries only, and this is the whole design. The ambiguous ones
    are ordinary standard words — measured over 2267 German tokens of the real
    recording, the *only* lexicon hits were 14 ambiguous ones ("mehr" ×10,
    "des" ×4), every one of them ordinary speech and none of them Berlinerisch.
    Colouring those would paint plain German red on a recording containing no
    detectable dialect at all.

    The consequence is worth stating rather than discovering later: because
    Whisper normalises dialect to standard orthography, this stays dark on the
    audio path. It earns its place on typed input, where the spelling survives.

    A selected `flavor` narrows the lexicon to that dialect; without one, any
    unambiguous marker counts.
    """
    lexicon = load_dialects().get(source)
    if not lexicon:
        return []
    found, seen = [], set()
    for token in re.findall(r"[^\W\d_]+", text.lower()):
        if token in seen:
            continue
        seen.add(token)
        entry = lexicon.get(token)
        if not entry:
            continue
        _gloss, ambiguous, flavors = entry
        if ambiguous:
            continue
        if flavor and flavors and flavor not in flavors:
            # Marking a Rhine-Hessian form while the user is listening to a
            # Berliner would be a new error, not a feature.
            continue
        found.append(token)
    return found


def dialect_notes(text: str, source: str,
                  asserted: str | None = None) -> str | None:
    """A hint block when the source text contains dialect markers.

    Whisper mangles spoken dialect in meaning-changing ways ("net
    verstanne" -> "nett verstarne", which gemma then translates as
    'understood nicely' — an inversion). Whisper-side prompt biasing made
    transcripts WORSE in testing, so the fix lives here: name the likely
    intended forms and let the model translate the intended meaning.

    `asserted` means the user has *selected* this dialect rather than us
    inferring it from spelling. That changes what the ambiguous entries are
    worth. Normally a word like "nett" proves nothing on its own — it is
    ordinary German — so it is only glossed alongside an unambiguous marker
    like "nochemol". But speech never supplies those markers: Whisper
    normalises dialect to standard orthography, and over 514 word tokens of
    the real recording not one unambiguous marker appeared, which is why
    this whole hint was dormant on the audio path. Once the speaker's
    dialect is asserted, an ambiguous hit is worth reporting on its own —
    hedged, because "nett" really can just mean nice.
    """
    intros = {
        "de": ("The German speaker is using regional dialect (e.g. Berlin, "
               "Hessian, or Rhine-Hessian), and speech recognition may have "
               "mis-heard dialect words."),
        "es": ("The Spanish speaker is using regional or colloquial forms "
               "(e.g. Mexican or Catalonia Spanish)."),
    }
    if source not in intros:
        return None
    lexicon = load_dialects().get(source)
    if not lexicon:
        return None
    hits, seen, marker = [], set(), False
    for token in re.findall(r"[^\W\d_]+", text.lower()):
        if token in seen:
            continue
        seen.add(token)
        entry = lexicon.get(token)
        if not entry:
            continue
        hits.append((token, *entry))
        marker = marker or not entry[1]
    if marker:
        # An unambiguous marker vouches for the whole sentence, so every hit
        # is worth glossing — including the ambiguous ones, whatever dialect
        # they are filed under. This is the typed-input path.
        listed = "; ".join(f'"{t}" = {g}' for t, g, _, _ in hits[:10])
        return (f"{intros[source]} In this sentence: {listed}. Translate the "
                "intended standard meaning naturally.")
    # No marker: only a selected dialect can justify a hint, and then only
    # for entries belonging to *that* dialect.
    eligible = [(t, g) for t, g, _, flavors in hits
                if not flavors or asserted in flavors]
    if not (asserted and eligible):
        return None                    # ambiguous words alone prove nothing
    listed = "; ".join(f'"{t}" = {g}' for t, g in eligible[:10])
    # Asserted dialect, but every hit is a word that is ordinary standard
    # language too. Say so and let context decide — flipping "nett" to "nicht"
    # in "das war nett von dir" would be its own inversion.
    lang = LANG_NAMES.get(source, source)
    return (f"{intros[source]} These words are also ordinary {lang}, so read "
            f"whichever fits: {listed}. If the dialect reading is the one "
            f"that makes sense here, translate that meaning — a mis-heard "
            f"negation is the common case.")


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


def has_phrase_loop(text: str, n: int = 3, times: int = 3) -> bool:
    """A short phrase repeated 3+ times — the loop shape the other two checks
    are both blind to.

    `is_degenerate` wants a compression ratio above 4.0 and `collapse_repeats`
    wants 8+ repeats of a unit under 12 chars. A ~17-char phrase repeated
    three times clears neither. Measured on the real recording, forcing German
    onto English speech produced "…die Füße starete, die sich so starete, die
    Füße starete, die Füße starete." — compression 1.33 against the 4.0
    threshold, and Whisper's own compression_ratio 1.57 against its 2.4. It
    was emitted verbatim.

    Counted over words rather than characters so punctuation and spacing
    cannot hide the repeat. Validated against the 51 real utterances of the
    Berlin slice: zero flagged, so this does not touch ordinary speech —
    including genuine emphasis like "Ja, ja, ja", which repeats a single word
    and never a 3-word phrase.
    """
    words = re.findall(r"\w+", text.lower())
    if len(words) < n * times:
        return False
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    # The gram must be a *varied* phrase. One word repeated is emphasis or
    # onomatopoeia, and five of them yield three copies of an all-identical
    # gram: over the full recording that dropped "…dann kommt das immer tüt
    # tüt tüt tüt tüt, dann tropft das da runter." — a coherent German
    # sentence about a gutter dripping. Runs of a single token are already
    # handled upstream by is_degenerate and collapse_repeats.
    return any(len(set(g)) > 1 and grams.count(g) >= times for g in set(grams))


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
    # After collapsing, not before: a tail of "L-L-L-…" is 100 repeated tokens
    # to a word-level check, so testing the raw text would throw away the real
    # speech in front of a loop that collapse_repeats can rescue.
    if has_phrase_loop(collapsed):
        return None                    # loop that survived collapsing
    return collapsed


# Parakeet writes its unknown-token literal straight into the text, and unlike
# Whisper it has no temperature ladder or compression threshold to fall back on
# when a decode degenerates. Seen live on the 25:00 slice: 390 consecutive
# `<unk>` with no spaces between them, held on screen for seconds. Every
# repetition filter missed it — they split on whitespace, so all 1950
# characters were a single "word".
UNK_TOKEN_RE = re.compile(r"(?:<unk>)+")


def clean_partial(text: str) -> str:
    """Usable live-partial text; "" when the fast decode produced nothing real.

    The fast path cannot use `clean_transcript` — that reads Whisper's segment
    structure and its per-segment confidences, which Parakeet does not produce
    — so the repetition checks have to be applied here too, or they only ever
    protected the finals.
    """
    return (_clean_segment(UNK_TOKEN_RE.sub(" ", text)) or "").strip()


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


# --------------------------------------------------- partial-pass ASR (fast)
#
# Live partials used to run on the same Whisper thread as finals, re-decoding
# a rolling ~6 s window from scratch every 2 s. Measured over a real
# conversation that redundancy was ~36% of everything the one thread decoded —
# against a deficit of only ~11%, which is why the queue grew without bound.
#
# Parakeet is an RNN-T/TDT model built for incremental decoding rather than
# sliding-window re-transcription, and it is ~12x faster here on the exact
# window AllKlaro decodes (measured 2026-08-06 on the same audio, both warm:
# 884 ms Whisper vs 71 ms Parakeet for 6 s of German; 822 ms vs 60 ms for
# Spanish, with identical transcripts apart from punctuation).
#
# Finals and speculations stay on Whisper. A speculation *becomes* the final
# transcript when the pause turns out to be real (see handle_utterance), so
# it is not optional work and must not be downgraded.
PARTIAL_ASR_REPO = os.environ.get("ALLKLARO_PARTIAL_ASR",
                                  "mlx-community/parakeet-tdt-0.6b-v3")
# Partials get their own worker: the whole point is that they no longer wait
# behind finals. Both models still share the GPU, but at ~70 ms a partial is
# no longer meaningful contention.
partial_executor = ThreadPoolExecutor(max_workers=1)
_parakeet = None
_parakeet_unavailable = False
_parakeet_lock = threading.Lock()


def load_parakeet():
    """Load the partial-pass model once. None when unavailable, which makes
    partials fall back to Whisper — degraded pacing, never a broken app."""
    global _parakeet, _parakeet_unavailable
    if _parakeet is not None or _parakeet_unavailable:
        return _parakeet          # fast path, no lock once settled
    # Callers are serialized by partial_executor today, but a second worker
    # (or a direct call) would otherwise start a duplicate multi-second load.
    with _parakeet_lock:
        if _parakeet is not None or _parakeet_unavailable:
            return _parakeet
        try:
            from parakeet_mlx import from_pretrained

            _parakeet = from_pretrained(PARTIAL_ASR_REPO)
            log.info("Partial-pass ASR ready (%s).", PARTIAL_ASR_REPO)
        except Exception as exc:
            _parakeet_unavailable = True
            log.warning("Partial ASR unavailable (%s); partials fall back to "
                        "Whisper and will compete with finals.", exc)
    return _parakeet


def transcribe_partial(audio: np.ndarray) -> str | None:
    """Transcribe a live-partial window. Returns the text, or None to mean
    "no fast model here, use Whisper" — never raises for that case."""
    model = load_parakeet()
    if model is None:
        return None
    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel

    # current_audio() already hands us float32 normalized to +-1.
    mel = get_logmel(mx.array(audio), model.preprocessor_config)
    results = model.generate(mel)
    return results[0].text.strip() if results else ""


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


def submit_transcribe(loop, audio: np.ndarray, language: str | None,
                      prompt: str | None = None, timing: dict | None = None):
    """Queue a transcription on the single Whisper thread, keeping count.

    Every caller goes through here — final utterances and speculations
    compete for the same worker, and `whisper_pending` is the only place
    that competition is visible. (Live partials have their own worker; see
    transcribe_partial.)

    Pass `timing` to learn how the wall time split. The caller can only see
    submit-to-result, which is queue wait AND decode together — and reading
    that number as "decoding" is exactly how a 73 s wait for 1.9 s of work
    once got attributed to a slow model.
    """
    global whisper_pending
    whisper_pending += 1
    t_submit = time.perf_counter()

    def job():
        t_start = time.perf_counter()
        try:
            return transcribe(audio, language, prompt)
        finally:
            if timing is not None:
                now = time.perf_counter()
                timing["queue_ms"] = int((t_start - t_submit) * 1000)
                timing["decode_ms"] = int((now - t_start) * 1000)

    fut = loop.run_in_executor(whisper_executor, job)
    fut.add_done_callback(_transcribe_finished)
    return fut


def _transcribe_finished(_fut) -> None:
    global whisper_pending
    whisper_pending -= 1


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
        # Why the last utterance was cut: "pause" (the speaker stopped),
        # "soft_max" (continuous speech, cut at a micro-pause), or "hard_max"
        # (cut mid-word). A run dominated by soft_max is the app being
        # outpaced — that is the case worth measuring.
        self.split_reason = None

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
            hard = seconds > MAX_UTTERANCE_SEC
            self.in_speech = False
            self.speech = []
            self.voiced_run = 0
            self.split_at = None
            self.speculating = False
            if seconds - self.silence_run * FRAME_MS / 1000 < MIN_UTTERANCE_SEC:
                return None
            self.split_reason = "hard_max" if hard else "pause"
            return utterance.astype(np.float32) / 32768.0
        if seconds > SOFT_MAX_SEC and self.split_at:
            # Continuous speech (a video, a fast talker) never yields a full
            # pause; emit up to the last natural micro-pause and keep going.
            utterance = np.concatenate(self.speech[:self.split_at])
            self.speech = self.speech[self.split_at:]
            self.split_at = None
            self.split_reason = "soft_max"
            return utterance.astype(np.float32) / 32768.0
        return None

    def current_audio(self) -> np.ndarray | None:
        if not self.in_speech or len(self.speech) < 32:  # need ~1 s for stability
            return None
        window = self.speech[-PARTIAL_WINDOW_FRAMES:]
        return np.concatenate(window).astype(np.float32) / 32768.0


def spec_expected_len(spec: dict, vad: "VAD") -> int:
    """How long the finished utterance must be for the in-flight speculation
    to be a transcript of it. Any other length means the speculation decoded
    a different span and its result has to be thrown away.

    There are two ways an utterance can end, and they need different sums.

    A `pause` split keeps every frame of the trailing silence, so the
    utterance is the speculated audio plus the silence still to come.

    A `soft_max` split is the one that used to be written off. It emits
    `speech[:split_at]`, cut at the last micro-pause — and a speculation is
    launched at EARLY_SILENCE_FRAMES of silence while `split_at` was set at
    MICRO_PAUSE_FRAMES of that same run, exactly (EARLY - MICRO) frames
    earlier. So when the split is at the speculation's own micro-pause, the
    speculated audio is the emitted chunk plus 128 ms of trailing silence:
    the same words. Requiring exactly that difference is what keeps it
    honest — a split at a *later* micro-pause makes the utterance longer
    than the speculation (words the speculation never saw), and an *earlier*
    one makes the gap bigger than one silence run.

    Measured on the real 54-minute recording before this existed: all 15
    misses in a 240 s slice were soft_max, and all 18 hits were pause. Each
    miss threw away a ~2 s decode on the one Whisper thread.
    """
    if vad.split_reason == "soft_max":
        return spec["len"] - (EARLY_SILENCE_FRAMES
                              - MICRO_PAUSE_FRAMES) * FRAME_SAMPLES
    return spec["len"] + (vad.end_silence
                          - EARLY_SILENCE_FRAMES) * FRAME_SAMPLES


# --------------------------------------------------------------------- startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    def _load():
        load_silero()
        log.info("Prewarming whisper model %s ...", WHISPER_REPO)
        transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), language="en")
        log.info("Whisper ready.")

    def _load_partial():
        # On its own worker so the (larger) Whisper load doesn't gate it, and
        # warmed here because the first real call compiles the MLX graph —
        # otherwise the first partial of a conversation pays for it.
        if load_parakeet() is not None:
            transcribe_partial(np.zeros(SAMPLE_RATE, dtype=np.float32))

    loop = asyncio.get_event_loop()
    loop.run_in_executor(whisper_executor, _load)
    loop.run_in_executor(partial_executor, _load_partial)
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
    flavors = {
        lang: (payload.get(f"{lang}_flavor")
               if payload.get(f"{lang}_flavor") in FLAVOR_NOTES[lang] else "")
        for lang in FLAVOR_NOTES
    }
    address = payload.get("address")
    address = address if address in VALID_ADDRESS else ""
    pair = mode_pair(mode)
    pinned = payload.get("source")           # caller overriding detection
    confidence = None
    if pair and pinned in pair:
        detected = pinned
    elif pair:
        detected, confidence = detect_language_scored(text, pair)
    else:
        detected = lang_hint(mode) or "de"
    source, targets = resolve_targets(mode, detected)
    translations = {}
    for target in targets:
        flavor = flavors.get(target, "")
        out = await translate_once(text, source, target, model, flavor=flavor,
                                   heard_flavor=flavors.get(source, ""),
                                   address=address)
        if out is None:
            return {"error": "Translation failed — are Ollama and the model "
                             f"{model!r} available?"}
        if not (target == "de" and flavor):  # guard assumes standard German
            out, _ = await enforce_agreement(text, source, target, model,
                                             None, out, address)
        translations[target] = out
    # "display" is a ready-captioned version for Shortcuts to show as-is.
    caption = f"🗣️ AllKlaro ({source.upper()} → {targets[0].upper()}):"
    return {"source": source, "target": targets[0],
            "translation": translations[targets[0]],
            "translations": translations,
            # How sure the detector was, so a Shortcut can re-ask with an
            # explicit "source" when the answer looks like a coin flip.
            "confidence": None if confidence is None else round(confidence, 2),
            "display": f"{caption}\n{translations[targets[0]]}"}


@app.get("/api/translate")
async def translate_api_get(text: str = "", mode: str = "", source: str = "",
                            de_flavor: str = "", es_flavor: str = "",
                            model: str = "", address: str = ""):
    """GET twin of the POST endpoint, so the whole pipeline can be
    smoke-tested from a phone browser's address bar:
    https://<mac-ip>:8710/api/translate?text=Hallo — separates "server
    unreachable" from "Shortcut built wrong" when debugging."""
    return await translate_api({"text": text, "mode": mode, "source": source,
                                "de_flavor": de_flavor,
                                "es_flavor": es_flavor, "model": model,
                                "address": address})


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


GIST_PROMPT = (
    "You keep a running gist of a live bilingual conversation for someone "
    "following it through translation. You are given the gist so far (it may "
    "be empty) and the lines spoken since, one per line, prefixed with their "
    "language. Rewrite the gist so it covers the whole conversation, "
    "including the new lines.\n"
    "Format: one to three lines, each starting with \"- \". Nothing else — no "
    "preamble, no heading, no closing remark. Asking for \"bullets\" is not "
    "enough on its own; gemma3:12b answers with a paragraph unless the line "
    "format is spelled out.\n"
    "Write in English, under 20 words a line. Say what is being discussed and "
    "any decision or follow-up.\n"
    "Keep the concrete details already in the gist — names, places, dates, "
    "numbers, decisions. Do not generalise them away: \"fly to Hawaii in "
    "March\" must not become \"their trip\". Folding a summary into a summary "
    "loses specifics unless it is told not to; measured at 50% loss of a "
    "named destination after a single fold without this instruction.\n"
    "Three lines total, not three new ones — replace and merge rather than "
    "append, or the list grows every minute.\n"
    "Summarise the lines; never copy them through, and never repeat the "
    "\"[DE]\" / \"[EN]\" language tags in your answer.\n"
    "Speech recognition makes "
    "mistakes, so ignore lines that are garbled or repeat a word over and "
    "over rather than trying to make sense of them."
)


def remember_for_gist(pending: list[dict], uid: int, source: str, text: str,
                      replaces: int | None = None) -> None:
    """Queue one utterance for the next fold.

    `replaces` is the uid of a fragment this utterance was merged with. The
    fragment's words are a prefix of this one, so keeping both would show the
    gist the same half-sentence twice — the same reason the translator pops it
    off `history`.
    """
    if replaces is not None and pending and pending[-1].get("uid") == replaces:
        pending.pop()
    pending.append({"uid": uid, "source": source, "text": text})
    # If folds keep failing, keep the newest rather than growing without bound:
    # an old gist plus recent lines still describes the call, and a 54-minute
    # conversation must not sit in memory waiting for an Ollama that is wedged.
    del pending[:-GIST_MAX_PENDING]


# The fold is shown the transcript as "[DE] …" / "[EN] …" so it knows who is
# speaking which language, and on a long conversation the model eventually
# copies a line through verbatim, tags and all, instead of summarising it.
# Observed on fold 6 of a six-fold run. The prompt asks it not to; this makes
# sure, because the tags are meaningless to someone reading the panel.
LANG_TAG_RE = re.compile(r"\[(?:DE|EN|ES)\]\s*", re.I)


def strip_lang_tags(text: str) -> str:
    return LANG_TAG_RE.sub("", text or "").strip()


def gist_messages(previous: str, lines: list[str]) -> list[dict]:
    """The fold step's prompt: the gist so far, plus what was said since."""
    body = ("Gist so far:\n" + (previous.strip() or "(nothing yet)")
            + "\n\nLines spoken since:\n" + "\n".join(lines))
    return [{"role": "system", "content": GIST_PROMPT},
            {"role": "user", "content": body}]


async def fold_gist(previous: str, lines: list[str], model: str) -> str | None:
    """One rolling-summary step; None on any failure, leaving the old gist up.

    Deliberately shaped like translate_once rather than /api/summarize: this
    runs inside a live session competing with the translator, so a failure
    has to be survivable rather than reported.
    """
    if not lines:
        return None
    body = {
        "model": model,
        "messages": gist_messages(previous, lines),
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": OLLAMA_NUM_CTX},
        "keep_alive": "60m",
    }
    if any(k in model.lower() for k in ("qwen3", "deepseek-r1", "gpt-oss")):
        body["think"] = False
    try:
        async with ollama_client() as client:
            r = await client.post("/api/chat", json=body,
                                  timeout=GIST_TIMEOUT_SEC)
            if r.status_code != 200:
                log.warning("gist failed: %s", r.text[:200])
                return None
            answer = r.json().get("message", {}).get("content", "").strip()
            return strip_lang_tags(answer) or None
    except Exception as exc:
        log.warning("gist failed: %s", exc)
        return None


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
        "options": {"temperature": 0.2, "num_ctx": OLLAMA_NUM_CTX},
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


# Optional output flavors (WhatsApp friends write Berlinerisch or Mexican
# Spanish; replying in kind is half the fun). Static per connection and
# inserted before the dynamic prompt parts, so prefix caching survives.
FLAVOR_NOTES: dict[str, dict[str, str]] = {"de": {}, "es": {}}
FLAVOR_NOTES["de"] = {
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
# The same selection, read the other way round: if you are writing replies in
# Berlinerisch you are listening to a Berliner. `dialect_notes` below tries to
# *infer* that from the text, and on speech it cannot — Whisper normalises
# dialect to standard orthography, so the unambiguous markers it looks for
# never arrive. Measured over 514 word tokens of the real Berlin recording:
# zero unambiguous markers, so the hint could not fire once. The selected
# flavor is the missing signal, and it is a user assertion rather than a guess.
#
# Static per connection, so this goes in the cacheable prefix, not with the
# per-sentence additions.
HEARD_DIALECT_NOTES: dict[str, dict[str, str]] = {"de": {}, "es": {}}
_HEARD_INTRO = ("You are translating transcribed speech from a {name} "
                "speaker. Speech recognition writes {short} in standard "
                "spelling and mis-hears its words in ways that change "
                "meaning — {example} Where a sentence only makes sense once "
                "such a word is restored, translate the intended meaning; "
                "where the text already reads as plain {lang}, translate it "
                "exactly as written and do not read dialect into it.")
HEARD_DIALECT_NOTES["de"] = {
    "berlin": _HEARD_INTRO.format(
        name="Berlin dialect (Berlinerisch)", short="Berlinerisch", lang="German",
        example=("ick (ich), dit/det (das), wat (was), ooch (auch), "
                 "keen/keene (kein/keine), nüscht (nichts), jut (gut) and "
                 "j- for g- (jehen, jenau) are the forms behind the "
                 "mis-hearings, and negation is the usual casualty.")),
    "hessian": _HEARD_INTRO.format(
        name="Hessian (Hessisch)", short="Hessisch", lang="German",
        example=('"net verstanne" (nicht verstanden) is transcribed as '
                 '"nett verstarne" and then translated as "understood '
                 'nicely" — the exact opposite. Also isch (ich), aach '
                 "(auch), ebbes (etwas), babbeln (reden), -sch for -ch.")),
    "worms": _HEARD_INTRO.format(
        name="Wormser Platt (Rheinhessisch)", short="Wormser Platt", lang="German",
        example=('"net" (nicht) arriving as "nett" inverts a negation. Also '
                 "nää (nein), isch (ich), aach (auch), ebbes (etwas), mer "
                 "(wir), gugge (schauen), -scht for -st (bischt, hoscht), "
                 "and dropped final -n on verbs (mer mache).")),
}
HEARD_DIALECT_NOTES["es"] = {
    "mexico": _HEARD_INTRO.format(
        name="Mexican Spanish", short="it", lang="Spanish",
        example=("ahorita, ¿mande?, platicar, chamba, órale and qué padre "
                 "are colloquial rather than literal — ahorita is not "
                 '"right now" and chido is not a proper noun.')),
    "barcelona": _HEARD_INTRO.format(
        name="Barcelona Spanish", short="it", lang="Spanish",
        example=("vale, tío/tía, guay, currar and plegar (from Catalan) are "
                 "colloquial rather than literal.")),
}

FLAVOR_NOTES["es"] = {
    "mexico": (
        "Write the Spanish translation in casual Mexican Spanish: ustedes "
        "(never vosotros), celular (not móvil), computadora (not "
        "ordenador), carro (not coche), jugo (not zumo), platicar "
        "(charlar), chamba (trabajo, colloquial), ahorita, ¿mande? for a "
        "polite 'what?', and órale / qué padre / chido where the tone "
        "fits. Example: \"That's really cool!\" → \"¡Está bien chido!\". "
        "Natural Mexican phrasing, not a caricature — never change the "
        "meaning."),
    "barcelona": (
        "Write the Spanish translation the way people speak Spanish in "
        "Barcelona (Peninsular Spanish with a Catalan tinge): vosotros "
        "for informal plural (podéis, tenéis), móvil (not celular), "
        "ordenador (not computadora), coche (not carro), zumo (not "
        "jugo), vale for okay, tío/tía as informal address, guay (cool), "
        "currar (trabajar, colloquial), plegar (to finish work, from "
        "Catalan). Example: \"Okay, see you guys later!\" → \"¡Vale, "
        "hasta luego, tíos!\". Natural and readable, not parody — never "
        "change the meaning."),
}


# How to render "you" in the target language. English collapses du/Sie/ihr
# (and tú/usted/ustedes) into one word, so the model can only guess unless
# the user pins it. Static per connection -> prefix-cache friendly.
VALID_ADDRESS = ("informal", "formal", "plural")
ADDRESS_NOTES = {
    "de": {
        "informal": ('Address the listener as informal singular "du" '
                     '(dich/dir/dein), never "Sie".'),
        "formal": ('Address the listener as formal "Sie" (Ihnen/Ihr), '
                   'never "du" or "ihr".'),
        "plural": ('"You" addresses several people here: use informal '
                   'plural "ihr" (euch/euer), never "du" or "Sie".'),
    },
    "es": {
        "informal": ('Address the listener as informal singular "tú" '
                     '(te/ti/tu), never "usted".'),
        "formal": ('Address the listener as formal "usted" (le/lo/su), '
                   'never "tú".'),
        "plural": ('Every "you" addresses a group: conjugate for "ustedes" '
                   '(third-person plural — pueden, tienen, tengan, su). '
                   'Example: "Can you help me?" → "¿Me pueden ayudar?". '
                   'Singular forms like "puedes" or "podría" are wrong here.'),
    },
}


def translation_messages(text: str, source: str, target: str,
                         history: list[dict] | None = None,
                         flavor: str | None = None,
                         address: str | None = None,
                         heard_flavor: str | None = None,
                         guard_language: bool = False) -> list[dict]:
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
    flavor_note = FLAVOR_NOTES.get(target, {}).get(flavor or "")
    if flavor_note:
        system += "\n\n" + flavor_note
    heard_note = HEARD_DIALECT_NOTES.get(source, {}).get(heard_flavor or "")
    if heard_note:
        system += "\n\n" + heard_note
    address_note = ADDRESS_NOTES.get(target, {}).get(address or "")
    if target == "es" and address == "plural" and flavor == "barcelona":
        # Barcelona style overrides the Latin American plural default.
        address_note = ('Every "you" addresses a group: use Peninsular '
                        '"vosotros" with second-person plural verbs '
                        '(podéis, tenéis, tengáis). Example: "Can you '
                        'help me?" → "¿Me podéis ayudar?".')
    if address_note:
        system += "\n\n" + address_note
    glossary = load_glossary()
    if glossary:
        system += ("\n\nGlossary — use these exact translations where "
                   "relevant:\n" + "\n".join(glossary))
    # Dynamic additions go last so the static prefix above stays cacheable.
    note = gender_notes(text, target)
    if note:
        system += "\n\n" + note
    dialect = dialect_notes(text, source,
                            asserted=heard_flavor if heard_note else None)
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
    if guard_language and (note or dialect or heard_note):
        # Draft-tier calls only. The restoration hints make a small model
        # flip tasks: qwen2.5:7b answered the Berlin note + a "gekickt"
        # gloss with the *corrected German sentence* instead of a
        # translation, and the merge stitch replayed that untranslated base
        # into every successor card. The guard is last because that is the
        # position that fixed qwen — and it must NOT reach the main model:
        # appended anywhere, in any wording tried, it made gemma3 stop
        # uncrossing the "nett verstarne" negation, and gemma never flipped
        # tasks in the first place.
        system += (f"\n\nThese hints only decide which meaning to pick. "
                   f"Your reply is still ONLY the {tgt} translation — never "
                   f"the corrected or restored {src} wording.")
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
                             flavor: str | None = None,
                             address: str | None = None,
                             heard_flavor: str | None = None,
                             guard_language: bool = False) -> str | None:
    """Streams deltas to the client; returns the full translation, or None on error.

    The caller sends the closing `translation_done` message (with metrics).
    """
    payload = {
        "model": model,
        "messages": translation_messages(text, source, target, history,
                                         flavor, address, heard_flavor,
                                         guard_language),
        "stream": True,
        "options": {"temperature": 0.0, "num_ctx": OLLAMA_NUM_CTX},
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


_prewarmed: set[str] = set()


async def prewarm_model(model: str) -> None:
    """Load a model into Ollama once, off the critical path.

    Fire-and-forget: a failure here only means the first refine pays the load,
    which is the behaviour without it.
    """
    if not model or model in _prewarmed:
        return
    _prewarmed.add(model)
    try:
        async with ollama_client() as client:
            await client.post("/api/chat", json={
                "model": model, "stream": False, "keep_alive": "60m",
                "messages": [{"role": "user", "content": "ok"}],
                "options": {"num_predict": 1, "temperature": 0.0,
                            "num_ctx": OLLAMA_NUM_CTX}},
                timeout=120)
        log.info("Prewarmed translation model %s", model)
    except Exception as exc:
        _prewarmed.discard(model)
        log.warning("Prewarm of %s failed (%s); first refine will pay the load",
                    model, exc)


async def translate_once(text: str, source: str, target: str, model: str,
                         history: list[dict] | None = None,
                         revise: tuple[str, str] | None = None,
                         flavor: str | None = None,
                         address: str | None = None,
                         heard_flavor: str | None = None) -> str | None:
    """One-shot, non-streaming translation; None on any failure.

    Used for the behind-the-scenes refinement pass, which replaces the fast
    draft wholesale — streaming deltas would rewrite text mid-read. With
    `revise=(candidate, issues)` the model is instead asked to correct its
    own earlier translation, with the offending facts spelled out.
    """
    messages = translation_messages(text, source, target, history,
                                    flavor, address, heard_flavor)
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
        "options": {"temperature": 0.0, "num_ctx": OLLAMA_NUM_CTX},
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
                            context: list[dict] | None, candidate: str,
                            address: str | None = None) -> tuple[str, bool]:
    """Returns (final text, changed). If the candidate contains impossible
    article/gender combinations (or LanguageTool findings, when enabled),
    re-ask once with the facts stated; the retry is used only if it
    verifies clean — otherwise keep the original."""
    issues = await _combined_issues(candidate, target)
    if not issues:
        return candidate, False
    log.info("agreement retry (%s): %s", target, " ".join(issues))
    fixed = await translate_once(text, source, target, model, context,
                                 revise=(candidate, " ".join(issues)),
                                 address=address)
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


def forced_pair(mode: str) -> tuple[str, str] | None:
    """The (src, tgt) of a plain forced mode like "de-en"; None otherwise.

    The mirror of `mode_pair`: that one names the languages an *auto* mode
    chooses between, this one names the direction a forced mode pins. Multi-
    target modes ("de-en+es") are deliberately excluded — there is no single
    language to fall back to, so they keep the pinned decode.
    """
    if "+" in mode or mode == "auto":
        return None
    parts = mode.split("-")
    if len(parts) == 2:
        src, tgt = parts
        if src in LANG_NAMES and tgt in LANG_NAMES and src != tgt:
            return src, tgt
    return None


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
#
# py3langid's character n-gram model does the actual work; the word lists
# below are its tie-breaking vote and the fallback when the package is
# missing. They used to be the whole detector, and that is what made short
# greetings unusable: "Happy birthday!" contains no listed word in any
# language, so every candidate scored zero and the tie fell through to the
# pair's first language — German, and with a dialect selected, loudly so.
# Measured over the 248 phrase/pair cases in tests/fixtures/detect_phrases.py:
# the scorer this replaced got 77.0%, these lists on their own 84.3%, and the
# two signals together 98.8%.
STOPWORDS = {
    "de": {"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen",
           "einem", "einer", "eines", "und", "oder", "aber", "denn",
           "sondern", "wenn", "weil", "dass", "ob", "als", "wie", "wo",
           "warum", "ich", "du", "er", "sie", "es", "wir", "ihr", "mich",
           "mir", "dich", "dir", "ihm", "ihn", "uns", "euch", "ihnen",
           "sich", "mein", "meine", "dein", "deine", "sein", "unser",
           "ist", "sind", "war", "waren", "bin", "bist", "seid", "haben",
           "hat", "habe", "hatte", "hatten", "wird", "werden", "wurde",
           "wurden", "kann", "können", "könnte", "muss", "müssen",
           "möchte", "will", "wollen", "soll", "sollen", "darf", "nicht",
           "kein", "keine", "nichts", "nie", "immer", "noch", "schon",
           "auch", "nur", "sehr", "mehr", "wieder", "mit", "ohne", "für",
           "von", "zu", "zum", "zur", "aus", "bei", "nach", "über",
           "unter", "vor", "auf", "an", "im", "am", "hier", "dort",
           "heute", "morgen", "gestern", "jetzt", "dann", "bitte",
           "danke", "ja", "nein", "gut", "geht", "gibt", "bis", "etwas",
           "viel", "viele"},
    "en": {"the", "a", "an", "and", "or", "but", "if", "because", "that",
           "which", "who", "what", "when", "where", "why", "how", "i",
           "you", "he", "she", "it", "we", "they", "me", "him", "her",
           "us", "them", "my", "your", "his", "its", "our", "their", "is",
           "are", "was", "were", "be", "been", "being", "am", "have",
           "has", "had", "do", "does", "did", "done", "will", "would",
           "can", "could", "should", "must", "may", "might", "not", "no",
           "never", "always", "still", "already", "only", "very", "more",
           "again", "just", "with", "without", "for", "from", "to", "of",
           "at", "by", "on", "in", "into", "out", "up", "down", "over",
           "under", "about", "here", "there", "today", "tomorrow",
           "yesterday", "now", "then", "please", "thanks", "thank", "yes",
           "ok", "good", "this", "these", "those", "some", "any", "all",
           "each", "let", "get", "got", "see", "know"},
    "es": {"el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
           "y", "o", "pero", "porque", "que", "si", "cuando", "donde",
           "como", "quien", "cual", "yo", "tú", "él", "ella", "nosotros",
           "ustedes", "ellos", "me", "te", "se", "nos", "le", "les", "mi",
           "mis", "su", "sus", "nuestro", "es", "son", "era", "eran",
           "soy", "eres", "somos", "está", "están", "estoy", "estamos",
           "ser", "estar", "he", "has", "ha", "hemos", "han", "haber",
           "hay", "tengo", "tiene", "tienen", "tener", "no", "nunca",
           "siempre", "ya", "todavía", "solo", "muy", "más", "también",
           "tan", "con", "sin", "para", "por", "de", "en", "a", "al",
           "del", "desde", "hasta", "sobre", "entre", "aquí", "allí",
           "hoy", "mañana", "ayer", "ahora", "entonces", "favor",
           "gracias", "sí", "bien", "este", "esta", "esto", "ese", "esa",
           "eso", "usted", "puedo", "quiero", "vamos"},
}
# Characters each language actually writes. One settles the answer only when
# exactly one candidate in the pair writes it: "ü" proves German against
# English, but proves nothing against Spanish ("pingüino", "bilingüe").
NATIVE_CHARS = {"de": "äöüß", "es": "ñ¿¡ü"}
# Weaker, because English borrows them too ("café", "naïve").
SOFT_CHARS = {"es": "áéíóú"}
# How the two signals are combined: the model wins unless it is hedging
# below MODEL_CEILING and the word lists lean the other way by at least
# LEXICAL_MARGIN ("Und sonst?" — one German function word, model 0.96 sure
# it is English). Loosening them further starts losing collision-heavy
# sentences ("The war was over in nineteen forty five", which carries three
# German function words); both were swept against the fixture corpus.
MODEL_CEILING = 0.97
LEXICAL_MARGIN = 1
# Below this the UI marks the card's language chip as a guess worth checking.
# It flags about a tenth of the fixture phrases — often enough to be worth
# glancing at, rare enough that the marking still means something.
UNSURE_BELOW = 0.75

_WORD_RE = re.compile(r"[^\W\d_]+")
_langid_identifier = None
_langid_lock = threading.Lock()


def _lexical_vote(text: str, candidates: tuple[str, ...]) -> tuple[dict, str | None]:
    """(function-word hits per candidate, the one language whose own
    orthography appears here — None if neither or both do)."""
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    hits = {}
    for lang in candidates:
        hits[lang] = sum(1 for w in words if w in STOPWORDS.get(lang, ()))
        hits[lang] += sum(0.5 for c in lowered if c in SOFT_CHARS.get(lang, ""))
    marked = [lang for lang in candidates
              if any(c in NATIVE_CHARS.get(lang, "") for c in lowered)]
    return hits, marked[0] if len(marked) == 1 else None


def _langid_vote(text: str, candidates: tuple[str, ...]) -> tuple[str | None, float]:
    """py3langid restricted to the active pair — the restriction is what
    makes it accurate on two-word phrases. None when the package is absent,
    so the app still runs (on the word lists alone) without it."""
    global _langid_identifier
    try:
        from py3langid.langid import MODEL_FILE, LanguageIdentifier
    except ImportError:  # pragma: no cover - exercised by the fallback test
        return None, 0.0
    try:
        with _langid_lock:   # set_languages mutates shared model state
            if _langid_identifier is None:
                _langid_identifier = LanguageIdentifier.from_pickled_model(
                    MODEL_FILE, norm_probs=True)
            _langid_identifier.set_languages(list(candidates))
            lang, prob = _langid_identifier.classify(text)
        return lang, float(prob)
    except Exception:
        log.exception("language model failed; falling back to word lists")
        return None, 0.0


def detect_language_scored(text: str,
                           candidates: tuple[str, str] = ("de", "en"),
                           ) -> tuple[str, float]:
    """Language of typed text, plus 0..1 confidence in that answer."""
    hits, proven = _lexical_vote(text, candidates)
    if proven:
        return proven, 1.0
    lang, prob = _langid_vote(text, candidates)
    if lang is None or lang not in candidates:
        # No model: the word lists decide, ties to the first candidate.
        best = max(hits.values())
        top = next(c for c in candidates if hits[c] == best)
        return top, (0.7 if best >= LEXICAL_MARGIN else 0.4)
    other = next(c for c in candidates if c != lang)
    if prob < MODEL_CEILING and hits[other] - hits[lang] >= LEXICAL_MARGIN:
        return other, 0.7
    if hits[lang] > hits[other]:
        prob = max(prob, 0.9)       # both signals agree
    elif hits[other] > hits[lang]:
        prob = min(prob, 0.7)       # they disagree and the model just wins
    return lang, prob


def detect_language(text: str, candidates: tuple[str, str] = ("de", "en")) -> str:
    return detect_language_scored(text, candidates)[0]


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
    superseded: set[int] = set()     # uids replaced by a merge mid-translation:
                                     # their run skips history and the refine —
                                     # the card is gone and the merged
                                     # successor stitched their draft already
    mode = "auto-de-en"
    model = DEFAULT_MODEL
    draft_model = None               # fast first-pass model; None = single-pass
    de_flavor = ""                   # "" standard | "berlin" | "hessian" | "worms"
    es_flavor = ""                   # "" standard | "mexico" | "barcelona"
    address = ""                     # "" auto | "informal" | "formal" | "plural"

    def flavor_for(target: str) -> str:
        return {"de": de_flavor, "es": es_flavor}.get(target, "")
    gist_on = True                   # running gist pinned above the feed
    gist = ""                        # the gist as last folded
    gist_pending: list[dict] = []    # utterances not yet folded into it
    gist_busy = False                # a fold is in flight
    # Started now, not at 0.0: against a monotonic clock the interval would
    # already be satisfied, and the first gist would summarize one utterance.
    last_gist = loop.time()
    corrected_uids: set[int] = set()  # user edits that refinement must not undo
    improving: set[int] = set()       # cards being re-translated on demand
    recapping = False                 # one "what did they just say?" at a time
    last_voice: dict[int, dict] = {}  # per channel: the last describable voice
    pause_frames = END_SILENCE_FRAMES
    uid = 0
    last_partial = 0.0
    busy = False                     # a transcription is in flight
    partial_busy = False             # ...specifically a live partial
    # Last finalized utterance, for merging sentence fragments cut mid-pause.
    prev = None                      # {"uid","text","source","speaker","t_end"}
    recent_finals: deque = deque(maxlen=8)  # for cross-channel echo dedupe
    # --- pipeline instrumentation (see trace() and the "stats" message) ---
    stats_on = False                 # client wants the live pipeline overlay
    in_flight = 0                    # utterances being transcribed/translated
    partials_skipped = 0             # partials dropped because the pipe was busy
    specs_shed = 0                   # speculations skipped because of backlog
    refines_shed = 0                 # refine passes skipped because of backlog
    # ...split, because one number could not answer the question it was there
    # to answer: a gate skip and a timeout are opposite failures — the first
    # spent nothing, the second spent the whole ceiling — and summing them hid
    # which one was happening.
    refines_gated = 0                # never attempted: backlog or too stale
    refines_timeout = 0              # attempted, then killed at the ceiling
    last_stats = 0.0
    last_done = None                 # summary of the last finished utterance

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
                               spec_task=None, meta: dict | None = None,
                               spec_timing: dict | None = None):
        nonlocal busy, prev, in_flight, last_done
        busy = True
        in_flight += 1
        t0 = loop.time()
        meta = meta if meta is not None else {"uid": my_uid, "t_emit": t0}
        # Time spent queued behind other utterances before this one even
        # started — invisible in the per-card latency.
        meta["wait_ms"] = int((t0 - meta["t_emit"]) * 1000)
        meta["in_flight"] = in_flight
        meta["outcome"] = "error"
        try:
            if spec_task is not None:
                result = await spec_task  # transcription began during the pause
                if spec_timing:
                    meta.update(spec_timing)
            else:
                result = await submit_transcribe(loop, audio, lang_hint(mode),
                                                 whisper_prompt(), timing=meta)
            detected = result.get("language", "de")
            mode_for_utterance = mode
            pair = auto_pair()
            if pair and detected not in pair:
                # Whisper picked a language outside the active pair (e.g. Dutch
                # for German speech) — the decode itself is wrong. Redo it
                # pinned to the pair's primary language.
                meta["redo"] = True   # a second full decode on the one thread
                result = await submit_transcribe(loop, audio, pair[0],
                                                 whisper_prompt())
                detected = pair[0]
            t1 = loop.time()
            meta["transcribe_ms"] = int((t1 - t0) * 1000)
            text = clean_transcript(result)
            forced = forced_pair(mode)
            if (not text and forced
                    and len((result.get("text") or "").strip())
                    >= FORCED_REDO_MIN_CHARS):
                # A forced direction pinned the wrong language onto this
                # utterance and Whisper looped; cleaning threw all of it away.
                # Decode once more with the language free, so the other half of
                # a bilingual conversation is rescued instead of vanishing.
                meta["redo_forced"] = True
                result = await submit_transcribe(loop, audio, None,
                                                 whisper_prompt())
                text = clean_transcript(result)
                detected = result.get("language", detected)
                if text and detected != forced[0]:
                    # Translate it the way it was actually spoken, for this
                    # utterance only — the mode is left alone. Without this the
                    # rescued English would be handed to the translator labelled
                    # German, which is the same error one stage later.
                    meta["redo_direction"] = detected
                    mode_for_utterance = f"auto-{forced[0]}-{forced[1]}"
            if not text or HALLUCINATION_RE.match(text):
                meta["outcome"] = "discard_empty"
                await safe_send(ws, {"type": "discard", "id": my_uid})
                return
            norm = normalize_text(text)
            if len(norm) >= ECHO_MIN_CHARS and any(
                    r["speaker"] != speaker and t0 - r["t"] < ECHO_WINDOW_SEC
                    and is_echo(norm, r["norm"]) for r in recent_finals):
                # Same speech heard on the other channel moments ago: the mic
                # picked up the call audio (or vice versa). Drop the echo.
                meta["outcome"] = "discard_echo"
                await safe_send(ws, {"type": "discard", "id": my_uid})
                return
            source, targets = resolve_targets(mode_for_utterance, detected)

            # Merge with the previous utterance when it did not finish — no
            # terminal punctuation, a trailing-off ellipsis, or THIS one
            # opening mid-sentence — and this one resumed right after. Rejoins
            # German verb-final clauses split by a short pause, where the
            # fragment alone cannot be translated.
            #
            # The casing half was added after comparing a whole hour against a
            # batch transcription service: it produced 216 paragraph-sized
            # segments of ~39 words where this pipeline produced 554. Three
            # of five hand-aligned passages lost their meaning purely to a cut
            # — "…hier mächtig." without *windig*, "gibt es da auch keine
            # Abdeckung für." without the pool it refers to. Item 12 measured
            # the cap's boundaries as no worse than a real pause and left the
            # cutting alone, which was the right answer to the wrong question:
            # the cost is the chunking itself, and merging is the way to pay
            # less of it without spending latency.
            #
            # An interjection is absorbed rather than merged: it skips the
            # language gate (detection on a 3-word chunk is a coin toss, and
            # each mis-call both minted a card and broke the chain — 77 of
            # the 210 language-flip refusals over the hour were tiny), skips
            # the evidence test, and skips the question rule — "ne? Ja. Ja."
            # reads fine inline, which is how the batch reference renders it.
            # The chain's own split reason survives an absorption: a "Ja."
            # landing mid-flow says nothing about whether the main speaker
            # stopped.
            replaces = None
            gap = (t0 - len(audio) / SAMPLE_RATE - prev["t_end"]) if prev else 99
            absorbed = (prev is not None and prev["speaker"] == speaker
                        and gap < MERGE_GAP_SEC
                        and len(prev["text"]) < MERGE_MAX_CHARS
                        and len(text.split()) <= ABSORB_MAX_WORDS)
            if absorbed and prev["source"] != source:
                # The card keeps its direction: "Yeah." inside a German
                # paragraph is part of the German card, whatever the
                # detector called it.
                source, targets = resolve_targets(mode_for_utterance,
                                                  prev["source"])
            chain_split = prev["split"] if absorbed else meta.get("split")
            # On a merge, the incoming chunk (`merge_tail`) and the replaced
            # card's finished translation (`merge_base`) let the translation
            # stage do O(new chunk) work instead of O(whole card): the first
            # /takt run after absorption shipped retranslated every growing
            # card in full, one chain climbing 20 s → 39 s per translate,
            # Whisper starving behind the saturated GPU (queue_ms 5 s → 32 s)
            # and last-lag hitting 59 s on a slice that normally runs 5-9 s.
            merge_tail = merge_base = merge_fut = None
            if (absorbed
                    or (prev and prev["speaker"] == speaker
                        and prev["source"] == source
                        and gap < MERGE_GAP_SEC
                        and len(prev["text"]) < MERGE_MAX_CHARS
                        and (not looks_finished(prev["text"])
                             or continues_previous(text)
                             or flowed_on(prev.get("split")))
                        # A question hands the turn over — the next chunk is
                        # someone's answer, not a continuation — unless casing
                        # says otherwise. See yields_turn for the measurement.
                        and (continues_previous(text)
                             or not yields_turn(prev["text"])))):
                merge_tail = text
                text = prev["text"] + " " + text
                replaces = prev["uid"]
                # The predecessor's translation may be finished (in history),
                # or still in flight (its future) — either way the successor
                # stitches rather than retranslating, and the replaced card
                # must not leave its fragment behind as context or spend a
                # refine on text nobody can see any more.
                if history and history[-1].get("uid") == replaces:
                    merge_base = history.pop()
                merge_fut = prev.get("tdone")
                superseded.add(replaces)

            last_final[source] = text[-200:]
            tdone = loop.create_future()
            prev = {"uid": my_uid, "text": text, "source": source,
                    "speaker": speaker, "t_end": t0,
                    "split": chain_split, "tdone": tdone}
            recent_finals.append({"norm": normalize_text(text),
                                  "speaker": speaker, "t": t0})
            remember_for_gist(gist_pending, my_uid, source, text, replaces)
            final_msg = {"type": "final", "id": my_uid, "text": text,
                         "source": source, "target": targets[0],
                         "targets": targets, "speaker": speaker,
                         # Dialect words are marked in the heard text only —
                         # the translation is standard by design, so there is
                         # nothing there to point at.
                         "dialect": dialect_markers(text, source,
                                                    flavor_for(source)),
                         # Whisper picked this language from the audio; a tap
                         # on the chip re-runs the translation the other way.
                         # A rescued forced decode counts too: the direction
                         # came from the audio, not from the mode, so the user
                         # needs the same escape hatch.
                         "auto": bool(pair) or "redo_direction" in meta,
                         # A break, not a name — see VOICE_CHANGE_DIST.
                         "voice_change": bool(meta.get("voice_change"))}
            if replaces is not None:
                final_msg["replaces"] = replaces
                meta["merged"] = True
                # A merged fragment is the *same* person carrying on through a
                # micro-pause — that is the entire reason it merged. Drawing a
                # "new voice" break above the joined card would contradict the
                # merge that just happened.
                final_msg["voice_change"] = False
            await safe_send(ws, final_msg)
            meta["outcome"] = "final"
            meta["chars"] = len(text)
            await run_translations(my_uid, text, source, targets, t0, t1, meta,
                                   merge_base=merge_base,
                                   merge_tail=merge_tail,
                                   merge_fut=merge_fut, tdone=tdone)
        except Exception as exc:
            log.exception("transcription failed")
            await safe_send(ws, {"type": "error", "id": my_uid,
                                 "message": f"Transcription failed: {exc}"})
        finally:
            busy = False
            in_flight -= 1
            # Age of this chunk's *last* word when its translation landed...
            # `t_card` is stamped when the card actually reaches the screen;
            # falling back to now covers the paths that never got that far.
            # Measuring here instead would fold in the *background* refine pass,
            # which happens after the user can already read the card — that made
            # one run report a 125 s lag for text that appeared in about 13 s.
            meta["lag_ms"] = int(((meta.pop("t_card", None) or loop.time())
                                  - meta["t_emit"]) * 1000)
            # ...and of its first, which is the delay a listener actually
            # feels: everything above plus the time the chunk spent growing.
            meta["first_word_lag_ms"] = (meta["lag_ms"]
                                         + int(meta.get("chunk_sec", 0) * 1000))
            meta.pop("t_emit", None)
            last_done = {k: meta.get(k) for k in
                         ("chunk_sec", "split", "spec", "wait_ms", "lag_ms",
                          "first_word_lag_ms", "outcome")}
            trace(meta)

    async def run_translations(my_uid, text, source, targets, t0, t1,
                               meta: dict | None = None,
                               merge_base: dict | None = None,
                               merge_tail: str | None = None,
                               merge_fut=None, tdone=None):
        nonlocal refines_shed, refines_gated, refines_timeout
        meta = meta if meta is not None else {}
        try:
            await _run_translations(my_uid, text, source, targets, t0, t1,
                                    meta, merge_base, merge_tail, merge_fut,
                                    tdone)
        finally:
            # The successor of a merge chain waits on this future; it must
            # resolve on EVERY exit path or the chain stalls for
            # CHAIN_WAIT_SEC on any failure here. And a superseded uid whose
            # run failed must not linger in the set forever.
            if tdone is not None and not tdone.done():
                tdone.set_result(None)
            superseded.discard(my_uid)

    async def _run_translations(my_uid, text, source, targets, t0, t1,
                                meta, merge_base, merge_tail, merge_fut,
                                tdone):
        nonlocal refines_shed, refines_gated, refines_timeout
        if is_already_understood(text, source):
            # Not a failure and not a shed: the listener reads German, and
            # this sentence was inside what they read. The card still appears
            # with the heard text — only the translation is withheld, and the
            # ✨ tap fetches it if the guess was wrong. That escape hatch is
            # what makes skipping safe enough to do by default.
            meta["skipped_known"] = True
            meta["translate_ms"] = 0
            await safe_send(ws, {"type": "translation_done", "id": my_uid,
                                 "refining": False, "known": True,
                                 "transcribe_ms": int((t1 - t0) * 1000),
                                 "translate_ms": 0})
            return
        # Two-tier translation: stream the fast draft model first so text
        # appears immediately, then re-translate with the main model
        # behind the scenes and swap in its (better) answer. Either way,
        # the final text passes the declension guard (enforce_agreement)
        # and is corrected once when it trips.
        draft = draft_model if draft_model and draft_model != model else None
        context = list(history)
        # A merged card whose predecessor already finished translating only
        # pays for the NEW chunk: the old translation is replayed as an
        # instant delta and the tail streams after it. Without this, every
        # link in a merge chain retranslated the whole grown card — O(k²)
        # over a k-link chain — which saturated the GPU, starved Whisper,
        # and drove last-lag to 59 s on a slice that normally runs 5-9 s.
        # The stitched translation is exactly what two unmerged cards would
        # have shown; the refine pass (backlog-gated, sheddable) later
        # retranslates the whole card in one piece when there is capacity.
        # Single-target only: history stores one translation, and the
        # multi-target mode is rare enough that correctness beats savings.
        base = None
        if (merge_base is not None and merge_tail and len(targets) == 1
                and merge_base.get("target") == targets[0]
                and merge_base.get("translation")):
            base = merge_base["translation"]
        elif merge_fut is not None and merge_tail and len(targets) == 1:
            # The predecessor is still translating. Waiting for it is the
            # cheap path, not the slow one: the wait overlaps work already
            # in flight, while giving up means retranslating the whole grown
            # card — and under dense speech (a chunk every 2-4 s against a
            # 1-2.5 s draft translate) that fallback fed itself: each full
            # retranslate finished later, made the NEXT link miss its base
            # too, and within a minute nothing stitched at all (translate
            # p50 8.7 s in the first two minutes of the 62-min replay).
            try:
                base = await asyncio.wait_for(asyncio.shield(merge_fut),
                                              timeout=CHAIN_WAIT_SEC)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                base = None
        if base:
            await safe_send(ws, {"type": "translation_delta", "id": my_uid,
                                 "target": targets[0], "text": base + " "})
            tail_tr = await stream_translation(
                ws, my_uid, merge_tail, source, targets[0], draft or model,
                context, flavor_for(targets[0]), address, flavor_for(source),
                guard_language=bool(draft))
            translations = [base + " " + tail_tr
                            if tail_tr is not None else None]
        else:
            translations = await asyncio.gather(
                *(stream_translation(ws, my_uid, text, source, t,
                                     draft or model, context, flavor_for(t),
                                     address, flavor_for(source),
                                     guard_language=bool(draft))
                  for t in targets))
        if all(t is not None for t in translations):
            # Hand the finished translation to any successor already waiting
            # to stitch onto it — before the refine, deliberately: the
            # successor replaces this card either way, and what it stitches
            # onto is what was on screen when it did.
            if tdone is not None and not tdone.done():
                tdone.set_result(translations[0])
            if my_uid in superseded:
                # A merge replaced this card while it was translating. The
                # successor has its translation via the future; appending it
                # to history would plant the fragment as stale context, and
                # refining text nobody can see is pure GPU spend.
                superseded.discard(my_uid)
                meta["translate_ms"] = int((loop.time() - t1) * 1000)
                return
            history.append({"uid": my_uid, "source": source,
                            "target": targets[0], "text": text,
                            "translation": translations[0]})
            t2 = loop.time()
            meta["translate_ms"] = int((t2 - t1) * 1000)
            await safe_send(ws, {
                "type": "translation_done", "id": my_uid,
                "refining": bool(draft),
                "transcribe_ms": int((t1 - t0) * 1000),
                "translate_ms": int((t2 - t1) * 1000)})
            # The card is now on screen. Everything after this is improvement,
            # not latency, and must not be charged to the lag a listener feels.
            meta["t_card"] = t2
            texts = {}
            # What actually happened to the refine, recorded rather than
            # inferred. Two traps made the earlier measurements wrong and both
            # were accounting, not pipeline: `refines_shed` counts the gate and
            # the timeout in one number, so it cannot tell "never tried" from
            # "tried and gave up"; and `refine_ms` is stamped on every
            # utterance whether the pass ran or not, so a gate-skipped refine
            # reads as a fast successful one. Naming the outcome removes the
            # guesswork — and with it the need to exclude "~0 ms" rows by hand.
            outcomes: list[str] = []
            refine_wait = 0.0
            refine_changed = agreement_changed = False
            for t, streamed in zip(targets, translations):
                candidate = streamed
                stale = (loop.time() - meta.get("t_emit", t0)) > REFINE_MAX_AGE_SEC
                # Two backlogs can make a refine not worth running, and until
                # partials moved off the Whisper thread only one of them was
                # ever the binding constraint. Measured on the real recording
                # with a sane draft/main pairing: the Whisper queue sat at 1-3
                # while refine took a p50 of 8.8 s of Ollama time against an
                # utterance arriving every ~6 s. Gating on `whisper_pending`
                # alone therefore let refines pile onto the *translation*
                # backlog, which is what the next card actually waits for.
                if draft and (whisper_pending > REFINE_MAX_QUEUE
                              or in_flight > REFINE_MAX_IN_FLIGHT or stale):
                    # The refine pass runs after the card is already readable
                    # and competes with Ollama and Whisper for the same GPU.
                    # Behind a backlog, or on an utterance already this old,
                    # better wording is worth far less than catching up.
                    refines_shed += 1
                    refines_gated += 1
                    outcomes.append("gated")
                elif draft:
                    t_refine = loop.time()
                    timed_out = False
                    try:
                        refined = await asyncio.wait_for(
                            translate_once(text, source, t, model, context,
                                           flavor=flavor_for(t),
                                           address=address,
                                           heard_flavor=flavor_for(source)),
                            timeout=REFINE_TIMEOUT_SEC)
                    except (asyncio.TimeoutError, TimeoutError):
                        # Keep the draft. A refinement nobody waited for is
                        # not worth blocking the utterances behind it.
                        refined, timed_out = None, True
                        refines_shed += 1
                        refines_timeout += 1
                    refine_wait += loop.time() - t_refine
                    if refined:
                        candidate = refined
                        # "Landed" means the refine returned in time, not that
                        # it changed anything: the main model agreeing with the
                        # draft is a real and common outcome, and counting it
                        # as a failure would overstate the pass's cost.
                        outcomes.append("landed")
                        refine_changed = refine_changed or refined != streamed
                    else:
                        # A model error is not a timeout, and reading one as
                        # the other is how a broken Ollama looks like a slow
                        # one. Kept separate for exactly that reason.
                        outcomes.append("timeout" if timed_out else "error")
                else:
                    outcomes.append("off")
                if t == "de" and de_flavor:
                    # The declension guard assumes standard German; dialect
                    # forms ("dit Haus", "keene") would trip it and get
                    # "corrected" back to Hochdeutsch.
                    final = candidate
                else:
                    final, fixed = await enforce_agreement(text, source, t,
                                                           model, context,
                                                           candidate, address)
                    agreement_changed = agreement_changed or fixed
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
            # The refine pass runs after the card is already on screen, but it
            # still occupies Ollama while the next utterance waits its turn.
            # `refine_ms` is everything after the card — refine *and* the
            # declension guard — and is kept that way so numbers measured
            # before this instrumentation stay comparable. `refine_wait_ms` is
            # the refine alone, which is what the timeout applies to.
            meta["refine_ms"] = int((loop.time() - t2) * 1000)
            meta["refine_wait_ms"] = int(refine_wait * 1000)
            meta["refine"] = "+".join(outcomes)
            # Which pass changed the text. Without this the two are
            # indistinguishable downstream, and an agreement retry reads as a
            # landed refine — the one join `capture_refines.py` must get right.
            meta["refine_changed"] = refine_changed
            meta["agreement_changed"] = agreement_changed

    async def improve_card(card_uid, text: str, source: str, target: str):
        """Re-translate one card with the main model because the user asked.

        Deliberately not gated on the backlog. Every other Ollama job here
        yields to the pipeline, and that is right for work nobody requested —
        but a tap is a person waiting on purpose, and making them lose to a
        queue they cannot see is how the refine pass ended up invisible.

        The heard text arrives from the client, exactly as the language-chip
        flip does, so a card older than the context window still works.
        """
        if (not isinstance(card_uid, int) or target not in LANG_NAMES
                or source not in LANG_NAMES or source == target):
            return
        if card_uid in improving:
            return                       # already running; a second tap is a no-op
        if len(improving) >= IMPROVE_MAX_IN_FLIGHT:
            await safe_send(ws, {"type": "improved", "id": card_uid,
                                 "target": target, "error": "busy"})
            return
        improving.add(card_uid)
        try:
            # Context as it stood *before* this card. Anything later describes
            # a part of the conversation that had not happened yet, and for a
            # card older than the deque there is simply no context left —
            # which is honest, and better than borrowing someone else's.
            context = [h for h in history if h.get("uid", 0) < card_uid]
            try:
                improved = await asyncio.wait_for(
                    translate_once(text, source, target, model, context,
                                   flavor=flavor_for(target), address=address,
                                   heard_flavor=flavor_for(source)),
                    timeout=IMPROVE_TIMEOUT_SEC)
            except (asyncio.TimeoutError, TimeoutError):
                improved = None
            except Exception:
                log.exception("improve failed")
                improved = None
            if not improved:
                await safe_send(ws, {"type": "improved", "id": card_uid,
                                     "target": target, "error": "failed"})
                return
            if target == "de" and de_flavor:
                # Same exemption as the refine pass: the declension guard
                # assumes standard German and would "correct" dialect back to
                # Hochdeutsch.
                final = improved
            else:
                final, _ = await enforce_agreement(text, source, target, model,
                                                   context, improved, address)
            # An improved card that is still in context should steer what
            # follows — unless the user has already corrected it by hand,
            # which outranks anything a model produces.
            if card_uid not in corrected_uids:
                for h in history:
                    if h.get("uid") == card_uid and h.get("target") == target:
                        h["translation"] = final
            await safe_send(ws, {"type": "improved", "id": card_uid,
                                 "target": target, "text": final})
        finally:
            # Held for the whole job, guard included: releasing the slot early
            # would let a second tap start while this one is still on Ollama.
            improving.discard(card_uid)

    async def recap_window(text: str, source: str, target: str,
                           before_uid: int | None):
        """Re-translate a stretch of recent cards as one passage.

        The ✨ tap fixes a card the model got wrong. This fixes a card that was
        never translatable on its own: item 12 measured 41% of cards still
        broken after the item 8 merge, because merging reaches across a 2 s gap
        and a German clause routinely spans more. Joining the stretch is the
        same repair item 8 makes automatically, just wider and on request.

        Not gated on the backlog, for the reason the ✨ tap is not: somebody is
        waiting on purpose. One at a time, though — the window overlaps itself
        on a second tap, so a queue of them would be the same passage racing.
        """
        nonlocal recapping
        if (target not in LANG_NAMES or source not in LANG_NAMES
                or source == target):
            return
        if recapping:
            await safe_send(ws, {"type": "recap", "error": "busy"})
            return
        recapping = True
        try:
            # Context from before the window only. The window is the newest
            # thing that happened, so anything at or after its first card is
            # the passage itself — feeding it back as context would ask the
            # model to translate the text twice and agree with itself.
            context = ([h for h in history
                        if h.get("uid", 0) < before_uid]
                       if before_uid is not None else list(history))
            try:
                out = await asyncio.wait_for(
                    translate_once(text, source, target, model, context,
                                   flavor=flavor_for(target), address=address,
                                   heard_flavor=flavor_for(source)),
                    timeout=RECAP_TIMEOUT_SEC)
            except (asyncio.TimeoutError, TimeoutError):
                out = None
            except Exception:
                log.exception("recap failed")
                out = None
            if not out:
                await safe_send(ws, {"type": "recap", "error": "failed"})
                return
            # Deliberately NOT written into `history`, and it replaces no card.
            # This is a second reading of text the conversation already has;
            # letting it steer what follows would double-count the same words.
            await safe_send(ws, {"type": "recap", "text": out,
                                 "heard": text, "source": source,
                                 "target": target})
        finally:
            recapping = False

    async def refresh_gist():
        """Fold the utterances since the last refresh into the running gist.

        The batch is only dropped once the fold succeeds, so a failed or
        timed-out refresh costs nothing but a minute -- and because new
        utterances are only ever appended, taking a prefix is safe even though
        the conversation keeps moving while this runs.
        """
        nonlocal gist, gist_busy
        gist_busy = True
        try:
            batch = gist_pending[:GIST_MAX_LINES]
            lines = [f"[{b['source'].upper()}] {b['text']}" for b in batch]
            updated = await fold_gist(gist, lines, model)
            if updated:
                gist = updated
                # By uid, not by index: the list can be trimmed from the front
                # while this await is outstanding, and dropping the wrong
                # utterances would silently lose them from the conversation.
                done = {b["uid"] for b in batch}
                gist_pending[:] = [p for p in gist_pending
                                   if p["uid"] not in done]
                await safe_send(ws, {"type": "gist", "text": gist})
        finally:
            gist_busy = False

    def forget(old_uid):
        """Drop a superseded utterance from the conversation context, so a
        redirected card's mistranslation stops steering later turns."""
        for h in [h for h in history if h.get("uid") == old_uid]:
            history.remove(h)
        corrected_uids.discard(old_uid)

    async def handle_text(text, my_uid, pinned=None, replaces=None):
        """Typed input: same translation pipeline, no audio machinery —
        no Whisper, no merging, no echo dedupe, no busy flag.

        `pinned` is the user overriding detection (the type bar's language
        pin, or a tap on a card's chip); `replaces` is the card that tap
        was correcting, which this one takes over from."""
        t0 = loop.time()
        try:
            pair = auto_pair()
            conf = None
            chosen = bool(pair) and pinned in pair
            if chosen:
                detected = pinned
            elif pair:
                detected, conf = detect_language_scored(text, pair)
            else:
                detected = lang_hint(mode) or "de"
            source, targets = resolve_targets(mode, detected)
            msg = {"type": "final", "id": my_uid, "text": text,
                   "source": source, "target": targets[0],
                   "targets": targets, "speaker": "you",
                   # The path this actually fires on: typed dialect keeps its
                   # spelling, where speech does not.
                   "dialect": dialect_markers(text, source, flavor_for(source)),
                   # Only an auto mode has a direction worth flipping.
                   "auto": bool(pair),
                   # ...and the chip shouldn't claim to have detected a
                   # language the user picked by hand.
                   "chosen": chosen}
            if conf is not None:
                msg["conf"] = round(conf, 2)
            if replaces is not None:
                forget(replaces)
                msg["replaces"] = replaces
            await safe_send(ws, msg)
            await run_translations(my_uid, text, source, targets, t0, t0)
        except Exception as exc:
            log.exception("typed translation failed")
            await safe_send(ws, {"type": "error", "id": my_uid,
                                 "message": f"Translation failed: {exc}"})

    async def maybe_partial():
        nonlocal last_partial, busy, partial_busy, partials_skipped
        now = loop.time()
        if now - last_partial < PARTIAL_INTERVAL_SEC:
            return                       # the interval is by design
        fast = not _parakeet_unavailable
        if partial_busy:
            # One at a time, by design — the previous one is still decoding.
            # Deliberately NOT counted as skipped: `partials_skipped` measures
            # partials lost to an overloaded pipeline, and inflating it with
            # normal pacing would make the overload metric meaningless.
            return
        # On the fast path a partial has its own worker, so a busy Whisper
        # thread is no longer a reason to skip one — that starvation is
        # exactly the cost this change removes.
        if busy and not fast:
            partials_skipped += 1
            return
        if whisper_pending > PARTIAL_MAX_QUEUE:
            # Not contention any more: this says the finals are so far behind
            # that live text would be describing a different moment than the
            # cards underneath it.
            partials_skipped += 1
            return
        if not fast and any(c["vad"].speculating for c in channels.values()):
            # A speculation owns the Whisper thread and its result becomes a
            # final; only the slow path can collide with it.
            partials_skipped += 1
            return
        audio = next((c["vad"].current_audio() for c in channels.values()
                      if c["vad"].in_speech), None)
        if audio is None:
            return
        last_partial = now
        partial_busy = True
        if not fast:
            busy = True
        try:
            text = None
            if fast:
                text = await loop.run_in_executor(partial_executor,
                                                  transcribe_partial, audio)
                if text is not None:     # None still means "no fast model"
                    text = clean_partial(text)
            if text is None:             # no fast model — Whisper does it
                result = await submit_transcribe(loop, audio, lang_hint(mode),
                                                 whisper_prompt())
                text = clean_transcript(result)
            if text and not HALLUCINATION_RE.match(text):
                await safe_send(ws, {"type": "partial", "text": text})
        except Exception:
            log.exception("partial transcription failed")
        finally:
            partial_busy = False
            if not fast:
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
                    if "stats" in cfg:  # live pipeline overlay on/off
                        stats_on = bool(cfg["stats"])
                    if "gist" in cfg:   # running gist above the feed on/off
                        gist_on = bool(cfg["gist"])
                    if not gist and isinstance(cfg.get("gist_text"), str):
                        # Reconnect: the client is handing back the gist it is
                        # still displaying, so the next fold continues the
                        # conversation instead of starting a second one under
                        # text that describes the first. Only ever *seeds* an
                        # empty gist — a live session's own fold always wins,
                        # and a client cannot overwrite it mid-call.
                        gist = cfg["gist_text"].strip()[:GIST_SEED_MAX_CHARS]
                    if "draft_model" in cfg:  # "" means draft pass off
                        draft_model = cfg["draft_model"] or None
                    if draft_model and draft_model != model:
                        # Two models means Ollama has to hold both. A cold load
                        # of the main model takes far longer than
                        # REFINE_TIMEOUT_SEC, so without this every refine is
                        # aborted mid-load and reloads from scratch forever —
                        # measured as refine_ms pinned at exactly the timeout,
                        # i.e. the refinement never once landed. Pay the load
                        # now, before anyone is waiting on it.
                        asyncio.create_task(prewarm_model(model))
                    if "de_flavor" in cfg:  # "" means standard German
                        de_flavor = (cfg["de_flavor"]
                                     if cfg["de_flavor"] in FLAVOR_NOTES["de"]
                                     else "")
                    if "es_flavor" in cfg:  # "" means standard Spanish
                        es_flavor = (cfg["es_flavor"]
                                     if cfg["es_flavor"] in FLAVOR_NOTES["es"]
                                     else "")
                    if "address" in cfg:  # "" lets context decide the you-form
                        address = (cfg["address"]
                                   if cfg["address"] in VALID_ADDRESS else "")
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
                        # "source" is the type bar's language pin: set, it
                        # skips detection entirely for this message.
                        asyncio.create_task(
                            handle_text(typed.strip()[:2000], uid,
                                        pinned=cfg.get("source")))
                elif isinstance(cfg, dict) and cfg.get("type") == "retranslate":
                    # The user tapped a card's language chip: detection (or
                    # Whisper) put it in the wrong language. Redo it with the
                    # direction they picked, replacing the card in place. The
                    # client sends the text back, so this works for cards
                    # older than the context window.
                    original = cfg.get("text")
                    if isinstance(original, str) and original.strip():
                        uid += 1
                        asyncio.create_task(
                            handle_text(original.strip()[:2000], uid,
                                        pinned=cfg.get("source"),
                                        replaces=cfg.get("id")))
                elif isinstance(cfg, dict) and cfg.get("type") == "improve":
                    # The user tapped ✨ on a card: give it the main model with
                    # no deadline and no backlog gate. The refine pass already
                    # tried this under both, which is why it so often did not
                    # land at all.
                    original = cfg.get("text")
                    if isinstance(original, str) and original.strip():
                        asyncio.create_task(
                            improve_card(cfg.get("id"),
                                         original.strip()[:2000],
                                         cfg.get("source") or "de",
                                         cfg.get("target") or ""))
                elif isinstance(cfg, dict) and cfg.get("type") == "recap":
                    # "What did they just say?" — the client joins the cards
                    # it showed in the last RECAP_WINDOW_SEC and sends the
                    # heard text back, as every other on-demand path does, so
                    # this works past the server's context window too.
                    passage = cfg.get("text")
                    before = cfg.get("before_uid")
                    if isinstance(passage, str) and passage.strip():
                        asyncio.create_task(
                            recap_window(passage.strip()[:RECAP_MAX_CHARS],
                                         cfg.get("source") or "de",
                                         cfg.get("target") or "",
                                         before if isinstance(before, int)
                                         else None))
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
                    if whisper_pending > SPEC_MAX_QUEUE:
                        # Speculation only pays off if it finishes before the
                        # real chunk arrives. Behind a queue it cannot, and it
                        # doubles the audio the one thread has to decode.
                        specs_shed += 1
                        vad.speculating = False
                    else:
                        spec_timing = {}
                        ch["spec"] = {"len": len(early),
                                      "timing": spec_timing,
                                      "task": submit_transcribe(
                                          loop, early, lang_hint(mode),
                                          whisper_prompt(),
                                          timing=spec_timing)}
                if utterance is not None:
                    spec = ch.pop("spec", None)
                    expected = spec_expected_len(spec, vad) if spec else -1
                    spec_task = spec["task"] if len(utterance) == expected else None
                    spec_timing = spec["timing"] if spec_task else None
                    uid += 1
                    # A miss means that speculation is still occupying the one
                    # Whisper thread with a result nobody will read.
                    meta = {"t": round(time.time(), 3), "uid": uid,
                            "t_emit": loop.time(),
                            "speaker": SPEAKERS[tag],
                            "chunk_sec": round(len(utterance) / SAMPLE_RATE, 2),
                            "split": vad.split_reason,
                            "spec": ("hit" if spec_task else
                                     "miss" if spec else "none"),
                            "whisper_queue": whisper_pending,
                            "partials_skipped": partials_skipped,
                            "specs_shed": specs_shed,
                            "refines_shed": refines_shed,
                            "refines_gated": refines_gated,
                            "refines_timeout": refines_timeout}
                    partials_skipped = 0
                    specs_shed = refines_shed = 0
                    refines_gated = refines_timeout = 0
                    # Compared here rather than in `handle_utterance`, because
                    # this is the only place utterances are still in order.
                    # Handlers run concurrently and finish out of order, so a
                    # comparison made there would sometimes be against the
                    # wrong neighbour. It costs about a millisecond of numpy on
                    # audio already in memory — no model, no GPU.
                    sig = voiceprint.voice_signature(utterance)
                    dist = voiceprint.voice_distance(last_voice.get(tag), sig)
                    # Still measured with the marks off: the distance is what a
                    # future attempt gets re-tuned from, and it costs a
                    # millisecond. Only the on-screen claim is withheld.
                    meta["voice_dist"] = None if dist is None else round(dist, 3)
                    meta["voice_change"] = bool(VOICE_MARKS_ON
                                                and dist is not None
                                                and dist > VOICE_CHANGE_DIST)
                    if sig is not None:
                        # Only a describable utterance becomes the new
                        # reference: letting a 300 ms "mhm" overwrite it would
                        # make the next real utterance look like a new voice.
                        last_voice[tag] = sig
                    await safe_send(ws, {"type": "segment_start", "id": uid,
                                         "speaker": SPEAKERS[tag]})
                    asyncio.create_task(
                        handle_utterance(utterance, uid, SPEAKERS[tag],
                                         spec_task, meta, spec_timing))
            if any(c["vad"].in_speech for c in channels.values()):
                asyncio.create_task(maybe_partial())
            now = loop.time()
            # The running gist, on the same terms as the refine pass: it is a
            # background improvement to a screen the user can already read, so
            # it yields to anything anyone is waiting on.
            idle = (in_flight <= GIST_MAX_IN_FLIGHT
                    and whisper_pending <= GIST_MAX_QUEUE)
            if (gist_on and gist_pending and not gist_busy
                    and now - last_gist >= GIST_INTERVAL_SEC
                    and (idle or now - last_gist >= GIST_MAX_STALE_SEC)):
                last_gist = now
                asyncio.create_task(refresh_gist())
            if stats_on and now - last_stats >= STATS_INTERVAL_SEC:
                last_stats = now
                # Audio already spoken that is still growing into a chunk —
                # nothing about it can reach the screen until it is cut.
                speech_sec = max(
                    (len(c["vad"].speech) * FRAME_MS / 1000
                     for c in channels.values() if c["vad"].in_speech),
                    default=0.0)
                await safe_send(ws, {"type": "stats",
                                     "in_flight": in_flight,
                                     "whisper_queue": whisper_pending,
                                     "speech_sec": round(speech_sec, 1),
                                     "partials_skipped": partials_skipped,
                                     "last": last_done})
    except WebSocketDisconnect:
        pass
