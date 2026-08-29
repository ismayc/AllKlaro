#!/usr/bin/env python3
"""Replay speech into a running AllKlaro server at realtime pace.

The hard part of "it can't keep up with a fast talker" is reproducing it on
demand: you can't reliably talk fast on cue, and no two attempts are the same
recording. This synthesizes continuous speech with macOS `say`, streams it
into /ws exactly as the browser worklet would (16 kHz mono PCM, 128 ms
chunks, realtime pacing), and reports what came back and when.

    uv run python tools/replay.py                  # default: fast, few pauses
    uv run python tools/replay.py --rate 300       # faster still
    uv run python tools/replay.py --pace 1.5       # 1.5x realtime (overload)
    uv run python tools/replay.py --audio talk.m4a # a real recording instead
    uv run python tools/replay.py --draft-model ""  # single-pass, no draft

The server writes one trace record per utterance while this runs; summarize
them afterwards with `uv run python tools/trace_report.py`.
"""
import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import websockets

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 2048          # 128 ms — the browser worklet's buffer size

# Deliberately comma-driven German with almost no sentence-final punctuation:
# `say` pauses at full stops, and a pause is exactly what this must not give
# the VAD. Roughly a minute at the default rate.
DEFAULT_TEXT = """
Also pass mal auf, ich erzähl dir kurz, was gestern passiert ist, und zwar
war ich ja eigentlich verabredet, aber dann kam alles anders, weil die Bahn
ausgefallen ist, und du weißt ja, wie das hier läuft, wenn eine Bahn
ausfällt, dann fällt gleich die nächste auch aus, und am Ende stehen alle
zusammen am Gleis und keiner weiß was, also bin ich losgelaufen, quer durch
die Stadt, an dem Park vorbei, wo im Sommer immer diese Konzerte sind, und
unterwegs hab ich dann gemerkt, dass ich mein Ladekabel vergessen hatte, was
natürlich super ist, wenn man sowieso schon zu spät dran ist und niemandem
Bescheid geben kann, aber gut, irgendwann war ich da, komplett durchgeschwitzt,
und dann sagt sie mir, dass sie den Termin sowieso verschieben musste, und
da stand ich dann, und ehrlich gesagt war ich in dem Moment einfach nur froh,
dass ich mich überhaupt hingesetzt habe, weil meine Füße nach dem ganzen
Marsch wirklich fertig waren, und dann haben wir doch noch was getrunken und
uns eine gute Stunde unterhalten, über die Arbeit, über die Wohnung, über
alles Mögliche eigentlich, und plötzlich war es schon dunkel draußen
"""


def synth(text: str, voice: str, rate: int, out: Path) -> Path:
    """Render text to 16 kHz mono s16 WAV — the server's native format."""
    subprocess.run(
        ["say", "-v", voice, "-r", str(rate),
         "--data-format=LEI16@16000", "--file-format=WAVE",
         "-o", str(out), text],
        check=True)
    return out


def convert(src: Path, out: Path) -> Path:
    """Down-convert any real recording to the same format via afconvert."""
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    str(src), str(out)], check=True)
    return out


# Models under ~4 GB paraphrase too freely to trust even as drafts. Mirrors
# MIN_DRAFT_BYTES in static/app.js; tests/test_model_pairing.py runs the real
# JavaScript against this to catch the two drifting apart.
MIN_DRAFT_BYTES = 4e9


def resolve_draft(models: list[str], sizes: dict, model: str) -> str:
    """The draft the app itself would pick for `model`, given what's installed.

    A measurement arm that leaves the draft pass off is not measuring the app:
    an hour of the real recording ran single-pass on gemma3:12b at translate
    p50 111 s against 3.3 s paired, with 227 Ollama read timeouts. This is the
    no-saved-settings half of resolvePair() in static/app.js.
    """
    size = lambda m: sizes.get(m, 0)                              # noqa: E731
    smaller = sorted((m for m in models if m != model and size(m) < size(model)),
                     key=size)
    for m in smaller:
        if size(m) >= MIN_DRAFT_BYTES:
            return m
    return smaller[-1] if smaller else ""


def fetch_models(ws_url: str) -> tuple[list[str], dict, str]:
    """Ask the server what Ollama has, the same endpoint the UI reads."""
    base = ws_url.replace("wss://", "https://").replace("ws://", "http://")
    base = base.rsplit("/ws", 1)[0]
    with urllib.request.urlopen(base + "/api/models", timeout=10) as r:
        payload = json.load(r)
    return (payload.get("models", []), payload.get("sizes", {}),
            payload.get("default", ""))


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        if (w.getnchannels(), w.getframerate(), w.getsampwidth()) != (1, 16000, 2):
            sys.exit(f"{path}: need mono 16 kHz 16-bit, got "
                     f"{w.getnchannels()}ch {w.getframerate()}Hz "
                     f"{w.getsampwidth() * 8}-bit")
        return w.readframes(w.getnframes())


async def stream(ws, pcm: bytes, pace: float, t0: float) -> None:
    """Feed the socket at (pace × realtime), like a live microphone."""
    step = CHUNK_SAMPLES * 2                       # bytes per 128 ms chunk
    period = CHUNK_SAMPLES / SAMPLE_RATE / pace
    for i in range(0, len(pcm), step):
        await ws.send(pcm[i:i + step])
        # Sleep to the wall-clock deadline so send latency can't accumulate
        # into a slower-than-realtime feed.
        deadline = t0 + (i / step + 1) * period
        await asyncio.sleep(max(0, deadline - time.monotonic()))
    # A closing stretch of silence so the VAD ends the last utterance.
    for _ in range(16):
        await ws.send(b"\x00" * step)
        await asyncio.sleep(period)


async def collect(ws, events: list, t0: float, quiet: bool) -> None:
    async for raw in ws:
        msg = json.loads(raw)
        msg["_at"] = round(time.monotonic() - t0, 2)
        events.append(msg)
        if quiet:
            continue
        if msg["type"] == "final":
            print(f"  [{msg['_at']:6.2f}s] final   {msg['text'][:70]}")
        elif msg["type"] == "partial":
            print(f"  [{msg['_at']:6.2f}s] partial {msg['text'][-60:]}")
        elif msg["type"] == "translation_done":
            print(f"  [{msg['_at']:6.2f}s] done    #{msg['id']} "
                  f"transcribe {msg['transcribe_ms']}ms "
                  f"translate {msg['translate_ms']}ms")


async def run(args) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="allklaro-replay-"))
    if args.audio:
        wav = convert(Path(args.audio), tmp / "in.wav")
    else:
        text = Path(args.text).read_text() if args.text else DEFAULT_TEXT
        wav = synth(" ".join(text.split()), args.voice, args.rate, tmp / "in.wav")
    pcm = read_pcm(wav)
    secs = len(pcm) / 2 / SAMPLE_RATE
    print(f"audio: {secs:.1f}s  ->  {args.url}  (pace {args.pace}x)")

    cfg = {"type": "config", "mode": args.mode, "pause_ms": args.pause_ms,
           "stats": False}
    if args.model:
        cfg["model"] = args.model
    # An unset --draft-model used to send "", which the server reads as "draft
    # pass off": a silent single-pass arm that measures a configuration the
    # app never runs. Unset now means "whatever the app would pick"; passing
    # an empty string is how you ask for single-pass on purpose.
    draft = args.draft_model
    if draft is None:
        try:
            models, sizes, default = fetch_models(args.url)
            draft = resolve_draft(models, sizes, args.model or default)
        except Exception as exc:                       # server down, old build
            print(f"  could not resolve a draft model ({exc}); "
                  f"running single-pass")
            draft = ""
    cfg["draft_model"] = draft
    print(f"models: {args.model or 'server default'} + "
          f"draft {draft or '(off)'}")

    events: list = []
    t0 = time.monotonic()
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps(cfg))
        reader = asyncio.create_task(collect(ws, events, t0, args.quiet))
        await stream(ws, pcm, args.pace, t0)
        # Let the backlog drain: every started segment should finish.
        deadline = time.monotonic() + args.drain
        while time.monotonic() < deadline:
            started = sum(e["type"] == "segment_start" for e in events)
            ended = sum(e["type"] in ("translation_done", "discard", "error")
                        for e in events)
            if started and started == ended:
                break
            await asyncio.sleep(0.25)
        reader.cancel()

    wall = time.monotonic() - t0
    finals = [e for e in events if e["type"] == "final"]
    started = sum(e["type"] == "segment_start" for e in events)
    done = sum(e["type"] == "translation_done" for e in events)
    print(f"\n{secs:.1f}s of audio -> {started} utterances, {done} translated, "
          f"{sum(e['type'] == 'partial' for e in events)} partials, "
          f"{wall:.1f}s wall clock")
    if done < started:
        print(f"  ⚠️  {started - done} utterance(s) never finished within "
              f"{args.drain}s of drain time")
    if finals:
        tail = finals[-1]
        print(f"  last card landed {tail['_at']:.1f}s in, "
              f"{wall - tail['_at']:.1f}s of audio still unaccounted for"
              if tail['_at'] < secs else
              f"  last card landed {tail['_at']:.1f}s in")
    if args.out:
        # The trace file records timings and never the text, so the two risks
        # the hour-long #19 bracket exists to rule out (a mis-heard final
        # biasing the next decode, and cross-language pull in code-switching
        # stretches) cannot be read from it. Keep the raw event stream.
        Path(args.out).write_text(
            "".join(json.dumps(e) + "\n" for e in events))
        print(f"  events -> {args.out} ({len(events)} messages)")
    print("\nNow: uv run python tools/trace_report.py --last "
          f"{max(started, 1)}")
    return 0 if done == started and started else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:8710/ws")
    p.add_argument("--rate", type=int, default=240,
                   help="say words-per-minute (default 240; normal speech ~175)")
    p.add_argument("--voice", default="Anna", help="say voice (default Anna, de_DE)")
    p.add_argument("--text", help="file of text to speak instead of the default")
    p.add_argument("--audio", help="replay a real recording instead of synthesizing")
    p.add_argument("--pace", type=float, default=1.0,
                   help="feed speed vs realtime; >1 deliberately overloads")
    p.add_argument("--mode", default="auto-de-en")
    p.add_argument("--model", default="", help="translation model (server default)")
    p.add_argument("--draft-model", default=None,
                   help="fast first-pass model; unset picks the one the app "
                        "would pair with the main model, \"\" turns the "
                        "draft pass off")
    p.add_argument("--pause-ms", type=int, default=700)
    p.add_argument("--drain", type=float, default=60.0,
                   help="seconds to wait for in-flight work after the audio ends")
    p.add_argument("--quiet", action="store_true", help="summary only")
    p.add_argument("--out", help="write every WebSocket message to this JSONL "
                                 "file (the trace has timings but no text)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not args.audio and not shutil.which("say"):
        sys.exit("`say` not found — this synthesizer is macOS-only; "
                 "use --audio with a recording instead.")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
