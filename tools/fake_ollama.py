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

## Why a fixed cost was not enough

A constant-latency server **cannot time out**, so it was blind to the one
knob the refine argument turns. That left the refine question stuck between a
rig precise enough to decide it and unable to see it, and live Ollama, which
can see it and cannot resolve anything below ~2.5 s.

`--dist lognormal` fixes that: latency is drawn from a distribution shaped by
two measured quantiles, so slow requests happen at a realistic rate and a
timeout ceiling actually bites.

**Repeatability survives, because it never came from the latency being
constant — it comes from the draws being reproducible.** Each request's cost
is derived from the seed and the request itself (model + the sentence being
translated), so a given utterance costs the same in every arm regardless of
what else ran. That matters more than it sounds: an arm that *skips* work
would otherwise shift every later request onto a different draw, and the arms
would then differ for a reason that is not the pipeline. Measured that way
once, at 652 ms of pure artefact. Change `--seed` to resample the world; keep
it to compare two pipelines inside the same one.
"""
import argparse
import asyncio
import json
import math
import random
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()


def lognormal_params(p50: float, p90: float) -> tuple[float, float]:
    """(mu, sigma) of the lognormal with these two quantiles.

    Quantiles rather than mean/stdev because quantiles are what the pipeline
    traces report, so the stub can be pointed straight at a measurement
    without anyone converting anything by hand.
    """
    if p90 <= p50:
        return math.log(max(p50, 1e-9)), 0.0
    return math.log(p50), (math.log(p90) - math.log(p50)) / 1.2815515655446004


class Latency:
    """Reproducible per-request latencies, in seconds, keyed by CONTENT.

    Sequential draws are the obvious design and they are wrong for any arm
    that changes *which* requests happen. Measured the hard way: an arm that
    skipped 5 translations shifted every later request onto a different draw,
    so the two arms disagreed by 652 ms for a reason that had nothing to do
    with the pipeline — the exact class of error this stub exists to remove.

    Keying on the request instead means a given sentence costs the same in
    every arm no matter what else ran, so arms that add, remove or reorder
    work stay comparable. That is strictly stronger than sequence-keying, and
    it is what makes an A/B over *load* possible at all.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fixed(self, ms: float, key: str = "") -> float:
        return ms / 1000

    def lognormal(self, p50: float, p90: float, key: str = "") -> float:
        mu, sigma = lognormal_params(p50, p90)
        if sigma == 0:
            return p50 / 1000
        rng = random.Random(f"{self.seed}\x00{key}")
        return math.exp(rng.gauss(mu, sigma)) / 1000


def request_key(body: dict) -> str:
    """What makes this request *this* request: the model and the text asked
    about. Deliberately not the whole body — the prompt carries a rolling
    conversation history, so including it would make the same sentence cost
    differently depending on what preceded it."""
    msgs = body.get("messages") or []
    last = msgs[-1].get("content", "") if msgs else ""
    return f"{body.get('model')}\x00{last}"

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
    lat = request.app.state.latency
    model = (body.get("model") or "").strip()
    if body.get("options", {}).get("num_predict") == 1:
        # The prewarm ping is not a translation, and must not consume a draw:
        # it fires once per session and would otherwise shift every latency
        # after it, making two arms differ for a reason that is not the
        # pipeline.
        return cfg.prewarm_ms / 1000
    draft = cfg.draft_model and model == cfg.draft_model
    p50 = cfg.draft_ms if draft else cfg.main_ms
    if getattr(cfg, "dist", "fixed") != "lognormal":
        return lat.fixed(p50)
    p90 = cfg.draft_p90 if draft else cfg.main_p90
    return lat.lognormal(p50, p90, request_key(body))


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
    app.state.latency = Latency(getattr(cfg, "seed", 0))
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
    p.add_argument("--dist", choices=("fixed", "lognormal"), default="fixed",
                   help="fixed is the old constant cost; lognormal spreads it "
                        "so a timeout ceiling can actually bite")
    # Tail quantiles, from the real recording: the refines the 10 s ceiling
    # was killing needed 14-17 s against a ~8 s median, and one attempt in
    # twelve exceeded 20 s in the capture run.
    p.add_argument("--main-p90", type=float, default=14000,
                   help="90th percentile refine cost (--dist lognormal)")
    p.add_argument("--draft-p90", type=float, default=4000,
                   help="90th percentile draft cost (--dist lognormal)")
    p.add_argument("--seed", type=int, default=0,
                   help="latency draws are reproducible; hold this fixed "
                        "across the arms of an A/B, change it to resample")
    cfg = p.parse_args()
    # Sizes matter: the UI picks a default draft by size, and an inverted
    # pairing is now rejected, so the stub has to look plausible.
    cfg.models = [("gemma3:12b", 8_100_000_000),
                  ("qwen2.5:7b-instruct", 4_700_000_000)]
    uvicorn.run(build(cfg), host="127.0.0.1", port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
