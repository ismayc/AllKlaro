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
constant — it comes from the draws being reproducible.** The sequence is
seeded and consumed in request order, so two arms of an A/B run see the
identical latency sequence and any difference between them is the pipeline.
Change `--seed` to resample the world; keep it to compare two pipelines
inside the same one.
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
    """A reproducible stream of per-request latencies, in seconds.

    Draws are taken in request order from one seeded generator, so the Nth
    translation of a replay always costs the same — which is what lets two
    arms be compared at all. A fresh `Latency` per run, never a global.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def fixed(self, ms: float) -> float:
        return ms / 1000

    def lognormal(self, p50: float, p90: float) -> float:
        mu, sigma = lognormal_params(p50, p90)
        if sigma == 0:
            return p50 / 1000
        return math.exp(self.rng.gauss(mu, sigma)) / 1000

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
    return lat.lognormal(p50, p90)


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
