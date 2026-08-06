"""A translation stage that costs the same every time.

Ollama's own timing is the loudest thing in a pipeline benchmark on this
machine: across replays of *identical* code the translate p50 ranged 2.9 s to
7.7 s, and two runs of the same arm differed by 3.6 s at first-word-lag p50 —
wider than any pipeline change worth making. Model residency moves with
whatever else the Mac is doing, so the noise is not going away by waiting.

This stands in for `ollama serve` at a latency you choose, so a replay
measures the pipeline (accumulation, chunking, queueing, shedding,
speculation) instead of measuring Ollama. It is emphatically NOT for judging
translation quality — the text it returns is filler of a realistic length.

    uv run python tools/fake_ollama.py --port 11435 --draft-ms 2600 \
        --main-ms 8800
    ALLKLARO_OLLAMA_URL=http://127.0.0.1:11435 uv run uvicorn server:app ...

Defaults come from the real recording so the shape of the run is familiar:
~2.6 s for the fast draft and ~8.8 s for the refine pass, the measured p50s.

Latency is per *request*, applied whole for a non-streaming call and spread
across the deltas of a streaming one, so a card fills in progressively the
way it does for real. Requests are served concurrently, exactly as Ollama
serves them, so queueing behaviour still emerges from the pipeline rather
than from this file.
"""
import argparse
import asyncio
import json
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

# Filler of a plausible length. Card width and delta count both matter to how
# the UI feels, and a one-word reply would make the pipeline look faster than
# it is.
FILLER = ("This is a fixed-cost stand-in for the translation of the sentence "
          "above, long enough to stream in several deltas.")


def latency_for(request: Request, body: dict) -> float:
    """Draft and refine cost different amounts; tell them apart by model name.

    The pipeline calls the draft model for the card the user reads and the
    main model for the background refine, so a single latency would hide the
    thing most worth measuring.
    """
    cfg = request.app.state.cfg
    model = (body.get("model") or "").strip()
    if body.get("options", {}).get("num_predict") == 1:
        return cfg.prewarm_ms / 1000        # the prewarm ping, not a translation
    if cfg.draft_model and model == cfg.draft_model:
        return cfg.draft_ms / 1000
    return cfg.main_ms / 1000


@app.get("/api/tags")
async def tags(request: Request):
    cfg = request.app.state.cfg
    return {"models": [{"name": n, "size": s} for n, s in cfg.models]}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    delay = latency_for(request, body)
    request.app.state.calls.append({"model": body.get("model"),
                                    "delay": delay, "t": time.time()})
    text = FILLER
    if not body.get("stream"):
        await asyncio.sleep(delay)
        return JSONResponse({"model": body.get("model"), "done": True,
                             "message": {"role": "assistant", "content": text}})

    async def stream():
        words = text.split(" ")
        # Spread the cost across the deltas so the card fills progressively.
        per = delay / max(1, len(words))
        for i, w in enumerate(words):
            await asyncio.sleep(per)
            chunk = {"model": body.get("model"), "done": False,
                     "message": {"role": "assistant",
                                 "content": w if i == 0 else " " + w}}
            yield json.dumps(chunk) + "\n"
        yield json.dumps({"model": body.get("model"), "done": True,
                          "message": {"role": "assistant", "content": ""}}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/_stats")
async def stats(request: Request):
    """What the pipeline actually asked for — lets a benchmark assert that an
    arm really did skip the refines it claimed to skip."""
    calls = request.app.state.calls
    return {"requests": len(calls),
            "by_model": {m: sum(1 for c in calls if c["model"] == m)
                         for m in {c["model"] for c in calls}}}


def build(cfg) -> FastAPI:
    app.state.cfg = cfg
    app.state.calls = []
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=11435)
    p.add_argument("--draft-ms", type=float, default=2600,
                   help="cost of a fast-draft translation (real p50)")
    p.add_argument("--main-ms", type=float, default=8800,
                   help="cost of a refine-pass translation (real p50)")
    p.add_argument("--prewarm-ms", type=float, default=50)
    p.add_argument("--draft-model", default="qwen2.5:7b-instruct",
                   help="which model name counts as the draft")
    cfg = p.parse_args()
    # Sizes matter: the UI picks a default draft by size, and an inverted
    # pairing is now rejected, so the stub has to look plausible.
    cfg.models = [("gemma3:12b", 8_100_000_000),
                  ("qwen2.5:7b-instruct", 4_700_000_000)]
    uvicorn.run(build(cfg), host="127.0.0.1", port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
