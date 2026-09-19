#!/usr/bin/env python3
"""A blind, order-randomized human read of what the refine pass is worth.

The open question the counters cannot answer is *value*: when a landed refine
changes the draft, is the change better, worse, or a wash under live pacing?
An LLM judge was tried and failed: scored twice with the two candidates
swapped, 67% of its verdicts flipped with position (see PROGRESS.md #2 and
`allklaro-refine-pass-decision`). That is a position bias, and the fix is to
take the judgment away from a rater who can see which candidate is which.

The protocol
------------
1. Capture real draft/refined pairs under live pacing:

       # server on 8710 with the real pair, then replay the hour:
       uv run python tools/capture_refines.py \\
         --audio ~/.cache/allklaro/hour.wav --json ~/.cache/allklaro/refine-rows.json

   Each row already carries `draft`, `final`, `substantive`, and `changed_by`.
   Only `substantive` rows (a landed refine that changed more than the two
   models' punctuation) are worth a human's time.

2. Build the blind rating sheet. Each pair is shown as two panels, A and B,
   with the refined and draft text assigned to the sides at random per item, no
   labels saying which is which:

       uv run python tools/refine_ab.py prepare \\
         ~/.cache/allklaro/refine-rows.json --out ~/.cache/allklaro/refine-ab

   That writes `refine-ab.html` (open it locally, since the pairs are private
   conversation text and never leave the machine) and `refine-ab.key.json`
   (which side was the refine; do not open it until after rating).

3. Rate every pair as "A better", "about the same", or "B better", then use
   the sheet's Download button to save `refine-ab.verdicts.json`.

4. Tally, unblinding against the key:

       uv run python tools/refine_ab.py tally \\
         ~/.cache/allklaro/refine-ab.key.json ~/.cache/allklaro/refine-ab.verdicts.json

   Ties (about the same) are dropped, and the remaining decided votes go
   through a two-sided exact sign test. The refine earns its cost only if it
   wins clearly more often than it loses; a null or negative result is the
   evidence that would reopen how aggressively it runs.

Sample size: read every substantive pair from the hour. If the decided count
is small (the hour yields on the order of ten), the sign test will not resolve
a modest effect, so accumulate pairs across several recordings into one sheet
rather than calling an underpowered result.
"""
import argparse
import html
import json
import random
import sys
from math import comb
from pathlib import Path


def substantive_pairs(rows: list[dict]) -> list[dict]:
    """The rows a human should read: a landed refine that changed the text
    beyond punctuation. Everything else is noise for this question."""
    return [r for r in rows if r.get("substantive")]


def build_sheet(rows: list[dict], seed: int = 0) -> tuple[list[dict], dict]:
    """Blinded items and the answer key.

    For each substantive pair, the refined and draft texts are assigned to
    sides A and B at random (seeded, so a run is reproducible). The returned
    items carry no hint of which side is which; the key maps each id to the
    side that holds the refine.
    """
    rng = random.Random(seed)
    items, key = [], {}
    for r in substantive_pairs(rows):
        uid = r["uid"]
        refine_is_a = rng.random() < 0.5
        a, b = (r["final"], r["draft"]) if refine_is_a else (r["draft"], r["final"])
        items.append({"id": uid, "heard": r["heard"],
                      "source": r.get("source", ""), "target": r.get("target", ""),
                      "A": a, "B": b})
        key[str(uid)] = "A" if refine_is_a else "B"
    return items, key


def tally(key: dict, verdicts: dict) -> dict:
    """Unblind the verdicts and run a two-sided exact sign test.

    `verdicts` maps id -> "A" | "B" | "same". A verdict matching the key's
    refine side is a refine win; the opposite side is a draft win; "same" (or a
    pair left unrated) is a tie and is dropped from the test.
    """
    refine_wins = draft_wins = ties = 0
    for uid, refine_side in key.items():
        v = verdicts.get(uid) or verdicts.get(str(uid))
        if v not in ("A", "B"):
            ties += 1
        elif v == refine_side:
            refine_wins += 1
        else:
            draft_wins += 1
    decided = refine_wins + draft_wins
    return {
        "refine_wins": refine_wins,
        "draft_wins": draft_wins,
        "ties": ties,
        "decided": decided,
        "net": refine_wins - draft_wins,
        "p_value": sign_test_p(refine_wins, decided),
    }


def sign_test_p(wins: int, n: int) -> float | None:
    """Two-sided exact sign test: the probability, if the refine and the draft
    were equally likely to win each decided pair, of a split at least this
    lopsided. None when nothing was decided."""
    if n == 0:
        return None
    total = 2 ** n
    tail = sum(comb(n, i) for i in range(n + 1)
               if abs(i - n / 2) >= abs(wins - n / 2))
    return min(1.0, tail / total)


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Refine A/B</title>
<style>
:root {{ color-scheme: light dark; --bg:#fff; --fg:#111; --panel:#f2f2f2;
  --line:#ccc; --pick:#2a7; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161616; --fg:#eee;
  --panel:#242424; --line:#444; }} }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); font:16px/1.5 system-ui,sans-serif;
  margin:0; padding:16px; }}
h1 {{ font-size:20px; }}
.pair {{ border:1px solid var(--line); border-radius:10px; padding:14px;
  margin:14px 0; }}
.heard {{ opacity:.7; font-size:14px; margin-bottom:10px; }}
.panel {{ background:var(--panel); border-radius:8px; padding:12px; margin:8px 0;
  font-size:17px; }}
.tag {{ font-weight:700; margin-right:8px; }}
.btns {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
button {{ font-size:16px; padding:12px 14px; border:1px solid var(--line);
  border-radius:8px; background:var(--bg); color:var(--fg); flex:1 1 30%;
  min-width:110px; }}
button.on {{ background:var(--pick); color:#fff; border-color:var(--pick); }}
#save {{ position:sticky; bottom:0; width:100%; margin-top:16px; flex:1 1 100%; }}
#count {{ font-weight:700; }}
</style></head><body>
<h1>Refine A/B: a blind read</h1>
<p>For each pair, pick the better translation of the heard line, or
&ldquo;about the same&rdquo;. You cannot tell which is the refine; that is the
point. Rated <span id="count">0</span> of {n}.</p>
<div id="pairs"></div>
<button id="save">Download verdicts</button>
<script>
const ITEMS = {items};
const KEY = "refine-ab";
const verdicts = JSON.parse(localStorage.getItem(KEY) || "{{}}");
const pairs = document.getElementById("pairs");
const count = document.getElementById("count");
function draw() {{
  pairs.innerHTML = "";
  for (const it of ITEMS) {{
    const box = document.createElement("div"); box.className = "pair";
    box.innerHTML = `<div class="heard">heard: ${{esc(it.heard)}}</div>
      <div class="panel"><span class="tag">A</span>${{esc(it.A)}}</div>
      <div class="panel"><span class="tag">B</span>${{esc(it.B)}}</div>`;
    const btns = document.createElement("div"); btns.className = "btns";
    for (const [val, label] of [["A","A is better"],["same","About the same"],["B","B is better"]]) {{
      const b = document.createElement("button");
      b.textContent = label;
      if (verdicts[it.id] === val) b.className = "on";
      b.onclick = () => {{ verdicts[it.id] = val;
        localStorage.setItem(KEY, JSON.stringify(verdicts)); draw(); }};
      btns.appendChild(b);
    }}
    box.appendChild(btns); pairs.appendChild(box);
  }}
  count.textContent = ITEMS.filter(it => verdicts[it.id]).length;
}}
function esc(s) {{ const d = document.createElement("div");
  d.textContent = s || ""; return d.innerHTML; }}
document.getElementById("save").onclick = () => {{
  const blob = new Blob([JSON.stringify(verdicts, null, 2)],
    {{ type: "application/json" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "refine-ab.verdicts.json"; a.click();
  URL.revokeObjectURL(a.href);
}};
draw();
</script></body></html>
"""


def render_html(items: list[dict]) -> str:
    """A self-contained local rating page embedding the blinded pairs."""
    return PAGE.format(n=len(items),
                       items=json.dumps(items, ensure_ascii=False))


def cmd_prepare(args) -> int:
    rows = json.loads(Path(args.rows).read_text())
    items, key = build_sheet(rows, args.seed)
    if not items:
        print("No substantive refines to rate in that capture.")
        return 1
    out = Path(args.out)
    html_path = out.with_suffix(".html")
    key_path = out.with_suffix(".key.json")
    html_path.write_text(render_html(items), encoding="utf-8")
    key_path.write_text(json.dumps(key, indent=2))
    print(f"{len(items)} pairs -> {html_path}")
    print(f"answer key      -> {key_path}  (do not open until after rating)")
    return 0


def cmd_tally(args) -> int:
    key = json.loads(Path(args.key).read_text())
    verdicts = json.loads(Path(args.verdicts).read_text())
    s = tally(key, verdicts)
    print(f"refine better : {s['refine_wins']}")
    print(f"draft better  : {s['draft_wins']}")
    print(f"about the same: {s['ties']}  (dropped)")
    print(f"decided       : {s['decided']}  (net {s['net']:+d} for the refine)")
    if s["p_value"] is None:
        print("no decided pairs, nothing to test")
    else:
        verdict = ("refine wins" if s["net"] > 0 else
                   "draft wins" if s["net"] < 0 else "dead even")
        print(f"sign test p   : {s['p_value']:.3f}  ({verdict})")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare", help="build the blind rating sheet")
    prep.add_argument("rows", help="capture_refines.py --json output")
    prep.add_argument("--out", default="refine-ab",
                      help="output stem (.html and .key.json are written)")
    prep.add_argument("--seed", type=int, default=0)
    prep.set_defaults(fn=cmd_prepare)
    tal = sub.add_parser("tally", help="unblind the verdicts and test them")
    tal.add_argument("key")
    tal.add_argument("verdicts")
    tal.set_defaults(fn=cmd_tally)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
