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
