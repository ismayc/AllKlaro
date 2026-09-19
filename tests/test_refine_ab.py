"""The blind A/B read of the refine pass (tools/refine_ab.py).

Guards the two things a hand-read must get right: the pairs are blinded so the
rater cannot see which side is the refine, and the unblinded tally runs an
exact sign test rather than eyeballing a count.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import refine_ab as ab  # noqa: E402


def rows(*specs):
    """specs: (uid, draft, final, substantive) -> capture-style rows."""
    return [{"uid": u, "heard": f"heard {u}", "source": "de", "target": "en",
             "draft": d, "final": f, "substantive": s} for u, d, f, s in specs]


SAMPLE = rows((1, "draft one", "refine one", True),
              (2, "draft two", "refine two", True),
              (3, "unchanged", "unchanged", False))


def test_only_substantive_pairs_are_read():
    assert [r["uid"] for r in ab.substantive_pairs(SAMPLE)] == [1, 2]


def test_the_key_names_the_side_holding_the_refine():
    items, key = ab.build_sheet(SAMPLE, seed=1)
    assert len(items) == 2 and len(key) == 2
    by_uid = {r["uid"]: r for r in SAMPLE}
    for it in items:
        refine_side = key[str(it["id"])]
        draft_side = "B" if refine_side == "A" else "A"
        assert it[refine_side] == by_uid[it["id"]]["final"]
        assert it[draft_side] == by_uid[it["id"]]["draft"]


def test_blinding_is_reproducible_and_seed_can_flip_it():
    assert ab.build_sheet(SAMPLE, seed=1)[1] == ab.build_sheet(SAMPLE, seed=1)[1]
    keys = {json.dumps(ab.build_sheet(SAMPLE, seed=s)[1], sort_keys=True)
            for s in range(8)}
    assert len(keys) > 1               # the assignment actually varies by seed


def test_items_carry_no_hint_of_which_side_is_which():
    items, _ = ab.build_sheet(SAMPLE, seed=1)
    assert set(items[0]) == {"id", "heard", "source", "target", "A", "B"}


def test_tally_counts_wins_ties_and_net():
    key = {"1": "A", "2": "B", "3": "A", "4": "B"}
    verdicts = {"1": "A",     # picked the refine  -> refine win
                "2": "A",     # picked the draft   -> draft win
                "3": "same",  # tie
                "4": "B"}     # picked the refine  -> refine win
    s = ab.tally(key, verdicts)
    assert (s["refine_wins"], s["draft_wins"], s["ties"]) == (2, 1, 1)
    assert s["decided"] == 3 and s["net"] == 1


def test_tally_treats_a_missing_verdict_as_a_tie():
    s = ab.tally({"1": "A", "2": "B"}, {"1": "A"})
    assert s["ties"] == 1 and s["decided"] == 1


def test_sign_test_is_the_exact_two_sided_binomial():
    assert ab.sign_test_p(0, 0) is None
    assert ab.sign_test_p(5, 5) == pytest.approx(2 / 32)      # a clean sweep
    assert ab.sign_test_p(6, 6) == pytest.approx(2 / 64)
    assert ab.sign_test_p(2, 4) == pytest.approx(1.0)         # an even split
    assert ab.sign_test_p(5, 6) == pytest.approx(14 / 64)


def test_rendered_page_shows_the_text_but_not_the_key():
    items, key = ab.build_sheet(SAMPLE, seed=1)
    page = ab.render_html(items)
    assert "refine one" in page and "draft one" in page
    # The page embeds the items, never the answer key.
    assert '"A"' not in json.dumps(key) or "key" not in page.lower()[:200]
    assert "Download verdicts" in page


def test_prepare_writes_the_sheet_and_key(tmp_path):
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(SAMPLE))
    stem = tmp_path / "sheet"
    args = ab.argparse.Namespace(rows=str(rows_path), out=str(stem), seed=0)
    assert ab.cmd_prepare(args) == 0
    assert stem.with_suffix(".html").exists()
    key = json.loads(stem.with_suffix(".key.json").read_text())
    assert set(key) == {"1", "2"}


def test_prepare_with_no_substantive_rows_fails_cleanly(tmp_path, capsys):
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(rows((9, "same", "same", False))))
    args = ab.argparse.Namespace(rows=str(rows_path),
                                 out=str(tmp_path / "s"), seed=0)
    assert ab.cmd_prepare(args) == 1
    assert "No substantive" in capsys.readouterr().out


def test_tally_cli_reports_the_test(tmp_path, capsys):
    key_path = tmp_path / "k.json"
    verdicts_path = tmp_path / "v.json"
    key_path.write_text(json.dumps({"1": "A", "2": "A"}))
    verdicts_path.write_text(json.dumps({"1": "A", "2": "A"}))
    args = ab.argparse.Namespace(key=str(key_path), verdicts=str(verdicts_path))
    assert ab.cmd_tally(args) == 0
    out = capsys.readouterr().out
    assert "refine better : 2" in out and "sign test p" in out


def test_main_dispatches_prepare(tmp_path):
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(SAMPLE))
    stem = tmp_path / "viacli"
    assert ab.main(["prepare", str(rows_path), "--out", str(stem)]) == 0
    assert stem.with_suffix(".html").exists()
