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
import zlib
from difflib import SequenceMatcher
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
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
CORRECTIONS_PATH = Path(__file__).parent / "corrections.jsonl"
CORRECTION_EXAMPLES = 3            # retrieved corrections shown to the model

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


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
            names = [m["name"] for m in r.json().get("models", [])]
    except Exception as exc:
        log.warning("Ollama unreachable: %s", exc)
        return {"models": [], "default": DEFAULT_MODEL,
                "error": f"Cannot reach Ollama at {OLLAMA_URL} — is `ollama serve` running?"}
    # Skip embedding models; they can't translate.
    names = [n for n in names if "embed" not in n]
    return {"models": names, "default": DEFAULT_MODEL}


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


def translation_messages(text: str, source: str, target: str,
                         history: list[dict] | None = None) -> list[dict]:
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
    glossary = load_glossary()
    if glossary:
        system += ("\n\nGlossary — use these exact translations where "
                   "relevant:\n" + "\n".join(glossary))
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
                             history: list[dict] | None = None) -> str | None:
    """Streams deltas to the client; returns the full translation, or None on error.

    The caller sends the closing `translation_done` message (with metrics).
    """
    payload = {
        "model": model,
        "messages": translation_messages(text, source, target, history),
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


def normalize_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


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
        m = "auto-de-en" if mode == "auto" else mode
        parts = m.split("-")
        if parts[0] == "auto" and len(parts) == 3 and all(
                p in LANG_NAMES for p in parts[1:]):
            return parts[1], parts[2]
        return None

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
            translations = await asyncio.gather(
                *(stream_translation(ws, my_uid, text, source, t, model,
                                     list(history)) for t in targets))
            if all(t is not None for t in translations):
                history.append({"uid": my_uid, "source": source,
                                "target": targets[0], "text": text,
                                "translation": translations[0]})
                await safe_send(ws, {
                    "type": "translation_done", "id": my_uid,
                    "transcribe_ms": int((t1 - t0) * 1000),
                    "translate_ms": int((loop.time() - t1) * 1000)})
        except Exception as exc:
            log.exception("transcription failed")
            await safe_send(ws, {"type": "error", "id": my_uid,
                                 "message": f"Transcription failed: {exc}"})
        finally:
            busy = False

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
                    try:
                        pause_ms = int(cfg.get("pause_ms", pause_frames * FRAME_MS))
                        pause_frames = max(200, min(2000, pause_ms)) // FRAME_MS
                    except (TypeError, ValueError):
                        pass
                    for c in channels.values():
                        c["vad"].end_silence = pause_frames
                    log.info("config: mode=%s model=%s pause=%dms",
                             mode, model, pause_frames * FRAME_MS)
                elif isinstance(cfg, dict) and cfg.get("type") == "correction":
                    # An edited translation also fixes the live context, so
                    # follow-up utterances build on the corrected phrasing.
                    corrected = cfg.get("corrected")
                    if isinstance(corrected, str) and corrected.strip():
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
