"""Offline tests for the deterministic answer checks (evals/code_checks.py)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import code_checks as cc  # noqa: E402

GAP_CHUNK = {"section": "T4", "text": (
    "Braking (corner entry):\n  RUS: first brakes between 1551m and 1616m (gap in the car data) | braking distance to first apex 81–147m\n"
    "  HAM: first brakes at 1593m from 260.0 km/h\n  → Brake-on points overlap within a gap in the car data: cannot tell who brakes later\n"
    "Time in section: RUS 20.100s | HAM 19.978s | HAM faster by 0.122s\nApex speeds: RUS 147.0 km/h | HAM 152.0 km/h")}
CLEAN_CHUNK = {"section": "T12", "text": (
    "Time in section: RUS 10.000s | HAM 10.125s | RUS faster by 0.125s\n"
    "  RUS: first brakes at 3629m from 273.0 km/h\n  HAM: first brakes at 3651m from 264.0 km/h\n"
    "  → HAM brakes 22m later than RUS")}


# ── uncertainty compliance ──

def test_firm_braking_claim_in_a_gap_section_fails():
    ans = "### T4\nHamilton was faster here. HAM braked 27m later than Russell into T4, which explains the gap."
    r = cc.uncertainty_compliance(ans, [GAP_CHUNK])
    assert r.label == "fail" and r.details[0]["section"] == "T4"


def test_hedged_claim_in_a_gap_section_passes():
    ans = "### T4\nHamilton was 0.122s faster, with +5 km/h at the apex. The braking comparison is unreliable due to a gap in Russell's car data, so who braked later into T4 cannot be determined."
    assert cc.uncertainty_compliance(ans, [GAP_CHUNK]).label == "pass"


def test_braking_claim_about_a_different_section_is_not_flagged():
    ans = "### T12\nHamilton braked 22m later than Russell into T12.\n\n### T4\nHamilton was faster by 0.122s; braking cannot be compared."
    assert cc.uncertainty_compliance(ans, [GAP_CHUNK, CLEAN_CHUNK]).label == "pass"


def test_no_gap_chunk_means_not_applicable():
    assert cc.uncertainty_compliance("HAM braked 22m later.", [CLEAN_CHUNK]).label == "n/a"


def test_section_names_are_matched_whole_not_as_prefixes():
    ans = "### T3-T4\nRussell braked 12m later than Hamilton."          # not T4 alone
    assert cc.uncertainty_compliance(ans, [GAP_CHUNK]).label == "pass"


# ── number grounding ──

def test_numbers_from_the_chunks_are_grounded():
    ans = "Russell was 0.125s faster in T12, braking at 3629m from 273.0 km/h; Hamilton braked at 3651m."
    assert cc.number_grounding(ans, [CLEAN_CHUNK]).label == "pass"


def test_a_difference_of_two_stated_numbers_is_grounded():
    assert cc.number_grounding("Hamilton braked 22m later (3629m vs 3651m).", [CLEAN_CHUNK]).label == "pass"


def test_an_invented_number_fails():
    r = cc.number_grounding("Russell was 0.201s faster and topped out at 331 km/h.", [CLEAN_CHUNK])
    assert r.label == "fail" and "0.201s" in r.explanation and "331km/h" in r.explanation


def test_a_wrong_sum_is_caught_and_the_right_one_passes():
    chunks = [{"section": "a", "text": "faster by 0.125s"}, {"section": "b", "text": "faster by 0.113s"},
              {"section": "c", "text": "HAM faster by 0.092s"}, {"section": "d", "text": "HAM faster by 0.122s"}]
    wrong = "Net advantage across these sections: +0.122s"                   # truly 0.125+0.113-0.092-... is not this
    # 0.122s is stated in a chunk, so the model quoting it passes; a total that exists nowhere must fail
    assert cc.number_grounding("In total Russell gained 0.999s.", chunks).label == "fail"
    assert cc.number_grounding("Russell's three gains total 0.330s.", chunks).label == "pass"      # 0.125+0.113+0.092
    assert cc.number_grounding(wrong, chunks).label == "pass"


def test_numbers_without_units_and_prose_are_ignored():
    assert cc.number_grounding("T4 at Apex 2 in 2026, P1, 3 sections, 1s earlier.", [CLEAN_CHUNK]).label == "n/a"


def test_lap_comparison_numbers_count_as_evidence():
    deltas = {"lap_gap_s": 0.063, "sections": [{"section": "T12", "gain_s": 0.125, "retrieved": True}]}
    assert cc.number_grounding("The lap gap was 0.063s.", [], deltas).label == "pass"


# ── totals ──

SIX = {"lap_gap_s": 0.063, "sections": [
    {"section": "T12", "gain_s": 0.125, "retrieved": True},
    {"section": "Main straight (finish, after final corner)", "gain_s": 0.113, "retrieved": True},
    {"section": "T3", "gain_s": 0.078, "retrieved": True},
    {"section": "T1-T2", "gain_s": -0.092, "retrieved": True},
    {"section": "T4", "gain_s": -0.122, "retrieved": True},
    {"section": "T10-T11", "gain_s": -0.080, "retrieved": True}]}
SIX_ANSWER = ("T12 +0.125s, the main straight at the finish +0.113s, T3 +0.078s; Hamilton took T1-T2 0.092s, "
              "T4 0.122s and T10-T11 0.080s. ")


def test_a_wrong_total_is_caught_even_though_its_digits_exist_elsewhere():
    ans = SIX_ANSWER + "Russell's net advantage across these six sections: +0.122s."
    assert cc.number_grounding(ans, [], SIX).label == "pass"            # 0.122s is T4's gain: grounded as a number
    r = cc.total_consistency(ans, [], SIX)
    assert r.label == "fail" and r.details[0]["claimed_s"] == 0.122


def test_the_right_total_passes():
    ans = SIX_ANSWER + "Russell's net advantage across these six sections is +0.022s."
    assert cc.total_consistency(ans, [], SIX).label == "pass"


def test_a_subset_total_passes():
    ans = SIX_ANSWER + "Hamilton's combined advantage in the three corners was 0.294s."
    assert cc.total_consistency(ans, [], SIX).label == "pass"


def test_no_total_means_not_applicable():
    assert cc.total_consistency(SIX_ANSWER, [], SIX).label == "n/a"


# ── coverage ──

DELTAS = {"lap_gap_s": 0.06, "sections": [
    {"section": "T12", "gain_s": 0.125, "retrieved": True},
    {"section": "Main straight (finish, after final corner)", "gain_s": 0.113, "retrieved": True},
    {"section": "T4", "gain_s": -0.122, "retrieved": False},
    {"section": "T9", "gain_s": 0.01, "retrieved": False}]}


def test_coverage_counts_retrieved_and_discussed_top_sections():
    ans = "T12 was decisive. On the main straight at the finish Russell gained too."
    ret, cov = cc.coverage(ans, [], DELTAS)
    assert ret.label == "pass" and ret.score == round(2 / 3, 3)         # T4 was not retrieved
    assert cov.score == round(2 / 3, 3)


def test_coverage_fails_when_the_big_sections_are_missing():
    deltas = {"sections": [{"section": s, "gain_s": g, "retrieved": r} for s, g, r in
                           [("T1", 0.3, False), ("T2", 0.2, False), ("T3", 0.1, True), ("T4", 0.01, True)]]}
    ret, cov = cc.coverage("Only T4 matters.", [], deltas)
    assert ret.label == "fail" and cov.label == "fail"


def test_coverage_not_applicable_without_a_pair():
    assert all(r.label == "n/a" for r in cc.coverage("x", [], None))


def test_straight_aliases():
    assert cc.section_mentioned("Gains came on the main straight before T1.", "Main straight (start, before T1)")
    assert not cc.section_mentioned("Gains came at the finish straight.", "Main straight (start, before T1)")
    assert cc.section_mentioned("On the T3-T4 straight Russell was quicker.", "Between T3-T4")


def test_the_saved_examples_run_cleanly():
    import json
    for ex in json.loads((ROOT / "web" / "examples.json").read_text()):
        results = cc.run_all(ex["answer"], ex["sources"], ex.get("deltas"))
        assert {r.name for r in results} == {"uncertainty_compliance", "number_grounding", "total_consistency", "retrieval_coverage", "answer_coverage"}
        assert all(r.label in {"pass", "fail", "n/a"} for r in results)
