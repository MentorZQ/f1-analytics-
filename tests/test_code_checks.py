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


def test_a_braking_claim_about_another_section_in_the_same_sentence_is_not_blamed_on_the_gap_section():
    ans = ("### Hamilton's gains\nHamilton recovered time at T1-T2 (+0.092s) and T4 (+0.122s), driven by higher apex "
           "speeds and more aggressive braking at T1-T2, and a 5.0 km/h apex speed advantage at T4.")
    assert cc.uncertainty_compliance(ans, [GAP_CHUNK]).label == "pass"


def test_the_same_sentence_fails_when_the_braking_claim_is_about_the_gap_section():
    ans = ("### Hamilton's gains\nHamilton recovered time at T1-T2 (+0.092s) and at T4 (+0.122s), "
           "braking 12m later at T4.")
    assert cc.uncertainty_compliance(ans, [GAP_CHUNK]).label == "fail"


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
        assert {r.name for r in results} == {"uncertainty_compliance", "number_grounding", "total_consistency", "inference_rate",
                                             "retrieval_coverage", "answer_coverage", "reversed_winner", "self_correction",
                                             "wrong_driver_or_session", "numbers_on_right_driver", "team_attribution", "team_ranking",
                                             "tyre_claims", "grid_start_claim", "completeness", "false_premise", "overclaiming"}
        assert all(r.label in {"pass", "fail", "info", "n/a"} for r in results)


# ── inference rate ──

def test_inference_rate_counts_causal_sentences():
    ans = ("Russell exited at 225.5 km/h and Hamilton at 220.0 km/h in T12. "
           "Hamilton braked 22m later, but his later braking did not translate into a time advantage. "
           "The deficit fed into the next straight because the exit speed was lower.")
    r = cc.inference_rate(ans)
    assert r.label == "info" and r.score == round(2 / 3, 3)
    assert len(r.details) == 2


def test_inference_rate_zero_for_plain_reporting():
    ans = "Russell was 0.125s faster in T12 with an exit of 225.5 km/h. Hamilton exited at 220.0 km/h and lost the section."
    assert cc.inference_rate(ans).score == 0.0


def test_inference_rate_ignores_tables_and_fragments():
    assert cc.inference_rate("| Section | Delta |\n| T12 | +0.125s |\nShort.").label == "n/a"


# ── regressions found by running the checks on a 40-question experiment ──

def test_a_claim_under_another_sections_heading_is_not_blamed_on_the_gap_section():
    gap = {"section": "T10-T11", "text": "LAW: first brakes between 3269m and 3306m (gap in the car data)\nTime in section: LAW 7.593s | HUL 7.432s | HUL faster by 0.161s"}
    ans = ("### T10-T11 (Corner Sequence)\nHUL faster by 0.161s. Lawson's exact point has a data gap, so Hulkenberg braked at least 1m later.\n\n"
           "### T12 (Corner) | 3599m-3896m\nLawson arrived with a +16 km/h entry advantage, a benefit of his stronger T10-T11 exit. "
           "Although Hulkenberg braked 15m later (3626m vs 3612m), Lawson held the apex.")
    assert cc.uncertainty_compliance(ans, [gap]).label == "pass"


def test_a_firm_claim_under_the_gap_sections_own_heading_still_fails():
    gap = {"section": "T10-T11", "text": "LAW: first brakes between 3269m and 3306m (gap in the car data)"}
    ans = "### T10-T11\nHulkenberg braked 15m later than Lawson into the corner."
    assert cc.uncertainty_compliance(ans, [gap]).label == "fail"


def test_thousands_separators_are_one_number():
    assert cc.numbers_with_units("Based on T12 (distance window 3,599m-3,896m)") == [(3599.0, "m"), (3896.0, "m")]
    chunk = [{"section": "x", "text": "Distance window: 3599m - 3896m"}]
    assert cc.number_grounding("Window 3,599m-3,896m", chunk).label == "pass"


def test_overall_for_one_section_is_not_a_lap_total():
    ans = SIX_ANSWER + "Russell was still 0.125s faster through T12 overall."
    assert cc.total_consistency(ans, [], SIX).label == "pass"


def test_net_across_sections_is_still_checked_even_when_the_digits_match_one_section():
    ans = SIX_ANSWER + "Russell's net advantage across these six sections: +0.122s."
    assert cc.total_consistency(ans, [], SIX).label == "fail"


# ── more regressions from the experiment ──

def test_a_bold_line_under_a_heading_does_not_reset_the_section():
    gap = {"section": "T10-T11", "drivers": ["LAW", "HUL"], "text": "LAW: first brakes between 3269m and 3306m (gap in the car data)"}
    ans = ("### T10-T11 (Corner Sequence)\n**HUL faster by 0.161s**\nBraking has a data gap.\n\n"
           "### T12 (Corner)\n**LAW faster by 0.139s**\nAlthough Hulkenberg braked 15m later (3626m vs 3612m), Lawson held the apex.")
    assert cc.uncertainty_compliance(ans, [gap]).label == "pass"


def test_a_gap_chunk_about_other_drivers_does_not_blame_a_claim_about_different_drivers():
    gap = {"section": "T4", "drivers": ["RUS", "HAM"], "text": "RUS: first brakes between 1551m and 1616m (gap in the car data)"}
    ans = "### T4\nHadjar braked 29m later than Verstappen into the corner."
    assert cc.uncertainty_compliance(ans, [gap]).label == "pass"
    ans2 = "### T4\nHamilton braked 29m later than Russell into the corner."
    assert cc.uncertainty_compliance(ans2, [gap]).label == "fail"


def test_caution_counts_as_a_hedge():
    gap = {"section": "T4", "drivers": ["VER", "HAD"], "text": "gap in the car data"}
    ans = "### T4\nHadjar braked 29m later than Verstappen, though the data warrants some caution here."
    assert cc.uncertainty_compliance(ans, [gap]).label == "pass"


def test_a_lap_level_gap_is_not_a_section_total():
    lap = {"section": None, "chunk_type": "driver_lap_summary", "text": "Gap to pole: 0.147s"}
    ans = SIX_ANSWER + "Norris finished P4, 0.147s behind Russell across the measured sections."
    assert cc.total_consistency(ans, [lap], SIX).label == "pass"
    assert cc.total_consistency(ans, [], SIX).label == "fail"


def test_a_headingless_claim_belongs_to_the_section_named_just_before_it():
    gaps = [{"section": "T4", "drivers": ["RUS", "ANT"], "text": "gap in the car data"},
            {"section": "T7-T8", "drivers": ["RUS", "ANT"], "text": "gap in the car data"},
            {"section": "T12", "drivers": ["RUS", "ANT"], "text": "first brakes at 3629m"}]
    ans = ("**T12** was the single biggest loss (0.084s). Despite ANT braking 12m later, Russell had the better exit.\n\n"
           "**T7-T8 (0.066s)** shows Russell carrying higher apex speeds.")
    assert cc.uncertainty_compliance(ans, gaps).label == "pass"
    bad = ("**T7-T8 (0.066s)** shows Russell carrying higher apex speeds. Antonelli braked 12m later than Russell here.")
    assert cc.uncertainty_compliance(bad, gaps).label == "fail"


def test_a_claim_after_a_list_of_sections_covers_every_section_in_the_list():
    gap = {"section": "T4", "drivers": ["RUS", "HAM"], "text": "RUS: first brakes between 1551m and 1616m (gap in the car data)"}
    ans = ("Hamilton clawed back 0.294s through T1-T2, T4, and T10-T11, where his later braking and higher apex speeds "
           "were genuine advantages over Russell.")
    r = cc.uncertainty_compliance(ans, [gap])
    assert r.label == "fail" and r.details[0]["section"] == "T4"


# ── regressions from the retrieval-fix experiment ──

def test_lap_times_written_as_minutes_and_seconds_are_one_number():
    assert cc.numbers_with_units("best lap 1:20.421s") == [(80.421, "s")]
    chunk = [{"section": None, "text": "Lap time: 80.421s"}]
    assert cc.number_grounding("Alonso's best lap was 1:20.421s", chunk).label == "pass"


def test_a_section_can_be_cited_by_its_distance_window():
    assert cc.section_mentioned("### Main Straight (4393m–4608m)", "Main straight (finish, after final corner)", 4393.0)
    assert not cc.section_mentioned("### Main Straight (1000m–1100m)", "Main straight (finish, after final corner)", 4393.0)


def test_combined_gains_cited_by_window_are_checked_against_all_discussed_sections():
    deltas = {"lap_gap_s": 0.063, "sections": [
        {"section": "T3", "gain_s": 0.078, "start_m": 993.0}, {"section": "T5-T6", "gain_s": 0.070, "start_m": 2141.0},
        {"section": "T12", "gain_s": 0.125, "start_m": 3599.0},
        {"section": "Main straight (finish, after final corner)", "gain_s": 0.113, "start_m": 4393.0}]}
    ans = ("### T3\nx\n### T5-T6\nx\n### T12\nx\n### Main Straight (4393m-4608m)\nx\n"
           "**Combined gains: ~0.386s across these sections**")
    assert cc.total_consistency(ans, [], deltas).label == "pass"


def test_section_times_and_per_section_gains_are_not_totals():
    ans = (SIX_ANSWER + "Norris completed the section in 6.442s, a cumulative speed advantage. "
           "For completeness: Hamilton gained 0.092s at T1-T2 and 0.122s at T4, so the net result was Russell on pole.")
    assert cc.total_consistency(ans, [], SIX).label in {"pass", "n/a"}      # examined, nothing wrong to flag


def test_a_rounded_time_is_grounded_only_when_the_exact_figure_is_shown_with_it():
    chunk = [{"section": None, "text": "Gap to pole: +0.186s"}]
    r = cc.number_grounding("Roughly 0.2s off the front.", chunk)
    assert r.label == "fail" and "exact figure not shown" in r.explanation
    assert cc.number_grounding("Roughly 0.2s (0.186s) off the front.", chunk).label == "pass"
    assert cc.number_grounding("Pole was 0.186s ahead. Roughly 0.2s.", chunk).label == "fail"   # exact figure is in another sentence


def test_a_made_up_or_two_decimal_figure_is_not_a_rounding():
    chunk = [{"section": None, "text": "Gap to pole: +0.186s"}]
    assert cc.number_grounding("Roughly 0.5s (0.186s) off the front.", chunk).label == "fail"
    assert cc.number_grounding("A gap of 0.21s.", chunk).label == "fail"


def test_the_largest_margin_is_a_single_section_gain_not_a_total():
    ans = SIX_ANSWER + "The speed advantage translated into a 0.125s time gain, the largest margin across the sections analysed."
    assert cc.total_consistency(ans, [], SIX).label == "pass"
    wrong = SIX_ANSWER + "Russell's net advantage across the sections analysed was 0.122s."
    assert cc.total_consistency(wrong, [], SIX).label == "fail"                    # still caught


def test_bare_lap_times_count_as_seconds_so_lap_to_lap_differences_are_grounded():
    chunk = [{"section": None, "text": "Lap 1: 1:19.911 <- personal best\n  Lap 3: 1:19.727 <- personal best\n  Lap 5: 1:19.401"}]
    assert cc.numbers_with_units("Lap 1: 1:19.911") == [(79.911, "s")]
    assert cc.numbers_with_units("best lap 1:20.421s") == [(80.421, "s")]                  # with an "s": still one number
    ans = "He went from 1:19.911 to 1:19.727 (-0.184s) and on to 1:19.401 (-0.326s from Lap 3)."
    assert cc.number_grounding(ans, chunk).label == "pass"
    assert cc.number_grounding("He improved by 0.9s between runs.", chunk).label == "fail"


# ── regressions from the Haiku run on the slim store ──

def test_more_ways_of_saying_the_data_does_not_settle_it_count_as_hedges():
    gap = {"section": "T4", "drivers": ["RUS", "HAM"], "text": "RUS: first brakes between 1551m and 1616m (gap in the car data)"}
    for sentence in ("Hamilton braked later into T4, but the context does not directly support that answer in full.",
                     "ANT's measured braking distance was shorter, but that is not sufficient to say which driver braked later in T4.",
                     "The context does not give a clean answer to who braked later into T4, because the car data has a gap at that point."):
        assert cc.uncertainty_compliance("### T4\n" + sentence, [gap]).label == "pass", sentence


def test_a_plain_claim_is_still_caught_after_widening_the_hedges():
    gap = {"section": "T4", "drivers": ["RUS", "HAM"], "text": "RUS: first brakes between 1551m and 1616m (gap in the car data)"}
    assert cc.uncertainty_compliance("### T4\nHamilton braked later than Russell into T4.", [gap]).label == "fail"


def test_a_written_out_sum_is_checked_as_arithmetic_not_as_a_total():
    ok = SIX_ANSWER + "Net picture: gains of 0.086s + 0.103s + 0.142s = 0.331s were outweighed by the straight."
    bad = SIX_ANSWER + "Net picture: gains of 0.086s + 0.103s + 0.142s = 0.431s were outweighed by the straight."
    assert cc.total_consistency(ok, [], SIX).label in {"pass", "n/a"}
    r = cc.total_consistency(bad, [], SIX)
    assert r.label == "fail" and "do not add up" in r.explanation


def test_totals_over_sectors_are_not_judged_against_section_gains():
    ans = SIX_ANSWER + "VER was quicker in sectors 1 and 2, but those gains (0.055s combined) were far smaller than the 0.204s HAM gained in the final sector."
    assert cc.total_consistency(ans, [], SIX).label in {"pass", "n/a"}


def test_a_gain_written_beside_a_named_straight_or_corner_is_not_a_total():
    deltas = {"lap_gap_s": 0.109, "sections": [{"section": "Main straight (finish, after final corner)", "gain_s": -0.101, "start_m": 5245.0},
                                               {"section": "T8-T9-T10", "gain_s": -0.077, "start_m": 3752.0}]}
    ans = "PIA lost time on the main straight (0.103s), the T8-T10 complex (0.077s) and the finish straight (0.101s), which together exceed the gap when combined."
    assert cc.total_consistency(ans, [], deltas).label in {"pass", "n/a"}


def test_a_sentence_that_says_its_figures_are_not_a_total_is_not_judged_as_one():
    ans = SIX_ANSWER + "The sectional deltas (0.333s + 0.346s + 0.320s + 0.248s) are not a total for the lap, and cannot be reconciled with the 0.519s gap."
    assert cc.total_consistency(ans, [], SIX).label in {"pass", "n/a"}
    assert cc.total_consistency(SIX_ANSWER + "Net advantage across these six sections: +0.122s.", [], SIX).label == "fail"      # still caught


def test_combined_with_and_a_window_total_are_not_section_list_totals():
    assert cc.total_consistency(SIX_ANSWER + "LIN's later braking, combined with an earlier throttle pickup, gave him the small 0.125s advantage.", [], SIX).label in {"pass", "n/a"}
    assert cc.total_consistency(SIX_ANSWER + "The 0.125s gap is the total for the 300m window.", [], SIX).label in {"pass", "n/a"}
    assert cc.total_consistency(SIX_ANSWER + "Their combined advantage was 0.122s.", [], SIX).label == "fail"                  # a real combined claim is still checked


def test_a_single_section_gain_after_across_the_section_is_not_a_total():
    deltas = {"lap_gap_s": 0.317, "sections": [{"section": "T3", "start_m": 1383.0, "gain_s": -0.020}, {"section": "T1-T2", "start_m": 715.0, "gain_s": 0.087}]}
    answer = "GAS was quicker through T3. The 0.020s gain comes from the distribution of speed across the section."
    assert cc.total_consistency(answer, [], deltas).label != "fail"


def test_figures_in_a_net_sentence_other_than_the_net_are_its_parts():
    deltas = {"lap_gap_s": 0.205, "sections": [{"section": "Main straight", "start_m": 0.0, "gain_s": -0.120},
                                               {"section": "T1-T2", "start_m": 715.0, "gain_s": -0.122},
                                               {"section": "T3", "start_m": 1024.0, "gain_s": 0.131},
                                               {"section": "T4", "start_m": 1500.0, "gain_s": 0.091}]}
    answer = "The sections sum to the Q1 gap as stated (COL ahead 0.120s and 0.122s, MAG ahead 0.131s and 0.091s, net 0.205s for MAG)."
    assert cc.total_consistency(answer, [], deltas).label != "fail"
    wrong = "The sections sum to a net 0.300s for MAG."
    assert cc.total_consistency(wrong, [], deltas).label == "fail"
