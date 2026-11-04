"""Offline tests for the self-consistency measures (evals/consistency.py)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import consistency as co  # noqa: E402

DELTAS = {"driver_a": "RUS", "driver_b": "ANT", "lap_gap_s": 0.3, "sections": [
    {"section": "T4", "gain_s": -0.101, "start_m": 1517.0}, {"section": "T12", "gain_s": 0.125, "start_m": 3599.0}]}
SOURCES = [{"section": "T4", "drivers": ["RUS", "ANT"], "text": "Time in section: RUS 12.9s | ANT 12.8s"}]


def run(answer):
    return {"answer": answer, "sources": SOURCES, "deltas": DELTAS}


def test_jaccard_basics():
    assert co.jaccard({1, 2}, {1, 2}) == 1.0 and co.jaccard({1}, {2}) == 0.0 and co.jaccard(set(), set()) == 1.0
    assert co.mean_pairwise([{1}]) is None


def test_identical_answers_agree_completely():
    a = "### T4\nANT faster by 0.101s, exit 149.0 km/h.\n### T12\nRUS faster by 0.125s."
    r = co.analyse("RUS vs ANT", "barcelona", [run(a), run(a), run(a)])
    assert r["numbers"] == 1.0 and r["sections"] == 1.0 and r["winners"] == 1.0 and r["length_cv"] == 0.0 and r["flaky_checks"] == []


def test_different_numbers_and_sections_lower_agreement():
    a = run("### T4\nANT faster by 0.101s with a 149.0 km/h exit.")
    b = run("### T12\nRUS faster by 0.125s with a 225.5 km/h exit.")
    r = co.analyse("RUS vs ANT", "barcelona", [a, b])
    assert r["numbers"] < 0.5 and r["sections"] == 0.0


def test_a_flipped_winner_is_counted_as_disagreement():
    good = run("### T4\nANT faster by 0.101s.")
    flipped = run("### T4\nRUS faster by 0.101s.")
    assert co.analyse("q", "barcelona", [good, flipped])["winners"] == 0.0
    assert co.analyse("q", "barcelona", [good, good])["winners"] == 1.0


def test_winners_are_only_compared_where_two_runs_make_the_claim():
    a = run("### T4\nANT faster by 0.101s.")
    b = run("### T12\nRUS faster by 0.125s.")
    value, n = co.winner_agreement([co.winners_named(a["answer"], DELTAS), co.winners_named(b["answer"], DELTAS)])
    assert value is None and n == 0


def test_a_check_that_changes_verdict_is_flaky():
    ok = run("### T4\nANT faster by 0.101s.")
    bad = run("### T4\nRUS faster by 0.101s.")                     # reversed_winner flips
    assert "reversed_winner" in co.analyse("q", "barcelona", [ok, bad])["flaky_checks"]


def test_summary_flags_low_agreement_questions():
    rows = [{"question": "stable", "numbers": 0.9, "sections": 1.0, "winners": 1.0, "flaky_checks": []},
            {"question": "wobbly", "numbers": 0.3, "sections": 0.5, "winners": None, "flaky_checks": ["number_grounding"]}]
    s = co.summarise(rows)
    assert s["low_agreement"] == ["wobbly"] and s["questions_with_flaky_checks"] == 1
