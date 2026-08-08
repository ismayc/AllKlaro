"""The fixed-cost translation stub.

A benchmark harness that is subtly wrong is worse than none — it produces
confident numbers about the wrong thing — so the stub is held to the two
properties the measurements depend on: it costs what it says, and it speaks
the protocol the real client parses.
"""
import argparse
import json
import time

import pytest
from fastapi.testclient import TestClient

from tools import fake_ollama


def make(**kw):
    cfg = argparse.Namespace(draft_ms=40, main_ms=120, prewarm_ms=0,
                             draft_model="qwen2.5:7b-instruct",
                             models=[("gemma3:12b", 8_100_000_000),
                                     ("qwen2.5:7b-instruct", 4_700_000_000)])
    for k, v in kw.items():
        setattr(cfg, k, v)
    return TestClient(fake_ollama.build(cfg))


def test_draft_and_refine_cost_different_amounts():
    """One latency for both would hide the thing most worth measuring: the
    refine pass is the expensive one, and whether it is worth its cost is an
    open question this harness exists to answer."""
    c = make()
    t = time.perf_counter()
    c.post("/api/chat", json={"model": "qwen2.5:7b-instruct", "stream": False,
                              "messages": []})
    draft = time.perf_counter() - t
    t = time.perf_counter()
    c.post("/api/chat", json={"model": "gemma3:12b", "stream": False,
                              "messages": []})
    main = time.perf_counter() - t
    assert draft == pytest.approx(0.04, abs=0.06)
    assert main > draft


def test_repeated_identical_requests_cost_the_same():
    """The entire point. The live server's spread on this call was 2.9-7.7 s."""
    c = make(main_ms=80)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        c.post("/api/chat", json={"model": "gemma3:12b", "stream": False,
                                  "messages": []})
        times.append(time.perf_counter() - t)
    assert max(times) - min(times) < 0.05, times


def test_prewarm_ping_is_not_billed_as_a_translation():
    """`num_predict: 1` is the model-load ping. Charging it a full
    translation would make prewarming look like a pipeline stall."""
    c = make(main_ms=500, prewarm_ms=0)
    t = time.perf_counter()
    c.post("/api/chat", json={"model": "gemma3:12b", "stream": False,
                              "messages": [], "options": {"num_predict": 1}})
    assert time.perf_counter() - t < 0.2


def test_streaming_matches_what_the_client_parses():
    """server.stream_translation reads NDJSON, takes
    `message.content` as the delta and stops on `done`."""
    c = make(draft_ms=20)
    with c.stream("POST", "/api/chat",
                  json={"model": "qwen2.5:7b-instruct", "stream": True,
                        "messages": []}) as r:
        assert r.status_code == 200
        chunks = [json.loads(l) for l in r.iter_lines() if l.strip()]
    assert len(chunks) > 3, "a single-delta stream would not exercise the UI"
    assert all("message" in ch for ch in chunks)
    assert chunks[-1]["done"] is True
    assert not any(ch["done"] for ch in chunks[:-1])
    text = "".join(ch["message"]["content"] for ch in chunks)
    assert text.strip()


def test_tags_lists_models_with_sizes():
    """/api/models feeds the UI's model picker, which needs sizes — and the
    pairing guard rejects a draft that is not smaller than the main model."""
    body = make().get("/api/tags").json()
    sizes = {m["name"]: m["size"] for m in body["models"]}
    assert sizes["qwen2.5:7b-instruct"] < sizes["gemma3:12b"]


def test_stats_report_what_the_pipeline_asked_for():
    """Lets a benchmark assert an arm really skipped the refines it claims."""
    c = make()
    for _ in range(3):
        c.post("/api/chat", json={"model": "qwen2.5:7b-instruct",
                                  "stream": False, "messages": []})
    c.post("/api/chat", json={"model": "gemma3:12b", "stream": False,
                              "messages": []})
    stats = c.get("/_stats").json()
    assert stats["requests"] == 4
    assert stats["by_model"]["qwen2.5:7b-instruct"] == 3
    assert stats["by_model"]["gemma3:12b"] == 1


# ------------------------------------------------- latency distribution

def draws(n, seed=0, p50=8000, p90=14000):
    """n distinct requests. Latency is keyed on the request, so the key has to
    vary — passing none would model the same sentence asked n times, which
    correctly returns the same answer every time."""
    lat = fake_ollama.Latency(seed)
    return [lat.lognormal(p50, p90, f"m\x00utterance {i}") * 1000
            for i in range(n)]


def test_the_quantiles_are_the_ones_asked_for():
    """The stub is pointed straight at a measurement, so it has to honour the
    quantiles a trace reports rather than a mean nobody measured."""
    xs = sorted(draws(4000))
    p50, p90 = xs[len(xs) // 2], xs[int(len(xs) * 0.9)]
    assert 7500 < p50 < 8500, f"p50 {p50:.0f} is not near 8000"
    assert 13000 < p90 < 15200, f"p90 {p90:.0f} is not near 14000"


def test_the_same_seed_gives_the_identical_sequence():
    """This is what makes an A/B possible at all. Repeatability never came
    from the latency being constant — it comes from the draws being
    reproducible, which survives making them variable."""
    assert draws(50, seed=7) == draws(50, seed=7)
    assert draws(50, seed=7) != draws(50, seed=8)


def test_a_timeout_ceiling_can_actually_bite():
    """The whole point. A fixed-cost server cannot time out, so it was blind
    to the one knob the refine argument turns."""
    over = [d for d in draws(2000) if d > 20_000]
    assert over, "nothing ever exceeded a 20 s ceiling — still un-timeout-able"
    assert len(over) < 400, "more than a fifth over 20 s is not this recording"


def test_a_degenerate_spread_falls_back_to_the_median():
    """p90 <= p50 is a misconfiguration, not a request for a weird
    distribution: serve the median rather than something unusable."""
    lat = fake_ollama.Latency(0)
    assert lat.lognormal(5000, 5000) == pytest.approx(5.0)
    assert lat.lognormal(5000, 1000) == pytest.approx(5.0)


def test_fixed_remains_the_default_and_is_unchanged():
    """Every number recorded before this change was measured at fixed cost;
    the default must not silently move under them."""
    c = make(dist="fixed", seed=0, main_p90=99999, draft_p90=99999)
    t0 = time.monotonic()
    c.post("/api/chat", json={"model": "gemma3:12b", "stream": False,
                              "messages": []})
    assert 0.10 < time.monotonic() - t0 < 0.30       # main_ms=120, as before


def test_the_prewarm_ping_does_not_consume_a_draw():
    """It fires once per session and is not a translation. Letting it take a
    draw would shift every latency after it, so two arms would differ for a
    reason that is not the pipeline."""
    c = make(dist="lognormal", seed=3, main_p90=400, draft_p90=200)
    body = {"model": "gemma3:12b", "stream": False, "messages": []}
    first = c.post("/api/chat", json=body) and c.app.state.calls[-1]["delay"]
    c2 = make(dist="lognormal", seed=3, main_p90=400, draft_p90=200)
    c2.post("/api/chat", json={**body, "options": {"num_predict": 1}})
    c2.post("/api/chat", json=body)
    assert c2.app.state.calls[-1]["delay"] == pytest.approx(first)


def test_latency_is_keyed_on_the_request_not_its_position():
    """The property that makes an A/B over *load* possible.

    Sequential draws look reproducible and are not: an arm that skips five
    translations shifts every later request onto a different draw, and the
    two arms then differ for a reason that is not the pipeline. Measured that
    way once, at 652 ms.
    """
    lat = fake_ollama.Latency(1)
    a = lat.lognormal(8000, 14000, "gemma3:12b\x00Wie geht es dir?")
    # Any amount of other traffic in between must not move it.
    for i in range(20):
        lat.lognormal(8000, 14000, f"gemma3:12b\x00filler {i}")
    b = lat.lognormal(8000, 14000, "gemma3:12b\x00Wie geht es dir?")
    assert a == b
    assert a != lat.lognormal(8000, 14000, "gemma3:12b\x00Etwas anderes")


def test_the_key_ignores_the_rolling_history():
    """The prompt carries recent conversation, so keying on the whole body
    would make the same sentence cost differently depending on what preceded
    it — reintroducing the very coupling this removes."""
    base = {"model": "gemma3:12b"}
    k1 = fake_ollama.request_key({**base, "messages": [
        {"content": "earlier turn"}, {"content": "Das Wasser verdunstet."}]})
    k2 = fake_ollama.request_key({**base, "messages": [
        {"content": "a completely different earlier turn"},
        {"content": "Das Wasser verdunstet."}]})
    assert k1 == k2
