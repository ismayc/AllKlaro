"""Repeatability check: two IDENTICAL arms, so every delta is noise.

Unlike compare.py this makes no baseline/fix claim. It prints the absolute
run-to-run delta for each metric, because that delta IS the measurement
floor -- the smallest effect the rig can honestly resolve.

Acceptance for PROGRESS.md #3: first_word_lag_ms p50 delta well under 1000 ms
(two identical live-Ollama arms previously differed by 3600 ms).
"""
import collections
import json
import sys


def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in recs if r.get("outcome") == "final"]


def pct(vals, p):
    vals = sorted(vals)
    k = min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1))))
    return vals[k]


a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"finals: run1={len(a)} run2={len(b)}  delta={abs(len(a) - len(b))}")
print()
print(f"{'metric':20} {'run1 p50':>9} {'run2 p50':>9} {'delta':>8} "
      f"{'run1 p90':>9} {'run2 p90':>9} {'delta':>8}")
for f in ("first_word_lag_ms", "lag_ms", "translate_ms", "decode_ms",
          "queue_ms", "transcribe_ms"):
    va = [r[f] for r in a if f in r]
    vb = [r[f] for r in b if f in r]
    if not va or not vb:
        continue
    p50a, p50b = pct(va, 50), pct(vb, 50)
    p90a, p90b = pct(va, 90), pct(vb, 90)
    print(f"{f:20} {p50a:9.0f} {p50b:9.0f} {abs(p50a - p50b):8.0f} "
          f"{p90a:9.0f} {p90b:9.0f} {abs(p90a - p90b):8.0f}")

print()
# Deterministic counters: these should match EXACTLY on identical arms.
# Any drift here is a correctness signal, not timing noise.
for name, recs in (("run1", a), ("run2", b)):
    spec = collections.Counter(r.get("spec") for r in recs)
    split = collections.Counter(r.get("split") for r in recs)
    print(f"{name}: spec hit={spec['hit']} miss={spec['miss']} "
          f"none={spec['none']}  splits={dict(split)}")

sa = collections.Counter(r.get("spec") for r in a)
sb = collections.Counter(r.get("spec") for r in b)
same = all(sa[k] == sb[k] for k in set(sa) | set(sb))
print(f"deterministic counters identical: {same}")

fa = pct([r["first_word_lag_ms"] for r in a if "first_word_lag_ms" in r], 50)
fb = pct([r["first_word_lag_ms"] for r in b if "first_word_lag_ms" in r], 50)
d = abs(fa - fb)
print()
print(f"VERDICT first_word_lag p50 delta = {d:.0f} ms "
      f"(live-Ollama reference was 3600 ms)")
print("  harness resolves sub-second effects" if d < 1000
      else "  STILL NOISY -- Ollama was not the only source")
