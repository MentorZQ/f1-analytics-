"""Offline tests for evals/claim_checks.py. Several cases are real sentences from experiment answers."""

import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import claim_checks as cl  # noqa: E402

DELTAS = {"driver_a": "RUS", "driver_b": "ANT", "lap_gap_s": 0.317, "sections": [
    {"section": "T1-T2", "gain_s": -0.060, "start_m": 634.0, "end_m": 993.0},         # ANT faster
    {"section": "T4", "gain_s": -0.101, "start_m": 1517.0, "end_m": 2141.0},          # ANT faster
    {"section": "T7-T8", "gain_s": 0.066, "start_m": 2347.0, "end_m": 2656.0},        # RUS faster
    {"section": "T13", "gain_s": 0.065, "start_m": 3896.0, "end_m": 4074.0},
    {"section": "T14", "gain_s": 0.005, "start_m": 4162.0, "end_m": 4393.0},
    {"section": "Main straight (finish, after final corner)", "gain_s": 0.113, "start_m": 4393.0, "end_m": 4608.0}]}


def rw(answer, deltas=DELTAS):
    return cl.reversed_winner(answer, [], deltas)


# ── reversed winner ──

def test_a_table_row_naming_the_slower_driver_fails():
    r = rw("| **T1–T2** | RUS faster by **0.060s** | RUS exit speed +5.0 km/h |")        # the real Antonelli answer
    assert r.label == "fail" and r.details[0]["said"] == "RUS" and r.details[0]["stored"] == "ANT"


def test_the_stored_winner_passes_in_every_phrasing():
    for text in ("### T1–T2\n**ANT faster by 0.060s**",
                 "Antonelli was 0.060s quicker through T1–T2.",
                 "T1–T2: +0.060s to ANT",
                 "ANT gained 0.060s at T1-T2.",
                 "RUS was 0.060s slower at T1–T2.",
                 "| T1–T2 | ANT | 0.060s |"):
        assert rw(text).label == "pass", text


def test_slower_and_lost_wordings_are_read_correctly():
    assert rw("Russell lost 0.101s at T4.").label == "pass"           # RUS lost -> ANT won: matches the store
    assert rw("Antonelli was 0.101s slower at T4.").label == "fail"


def test_a_sentence_about_several_sections_is_not_judged():
    s = "Verstappen was faster through T1-T2 (by 0.060s) and T4 (by 0.101s) but slower through T7-T8 (0.066s)."
    assert rw(s).label == "n/a"


def test_pronouns_and_possessives_are_not_guessed():
    assert rw("Antonelli's one advantage over Russell came at T4, where he was 0.101s faster.").label == "n/a"
    assert rw("Russell's 154 km/h exit, and was 0.066s faster through T7-T8.").label == "n/a"


def test_a_heading_gives_the_section_for_the_sentence_beneath():
    ok = "### T4 (1517m–2141m)\nAntonelli was 0.101s quicker here."
    bad = "### T4 (1517m–2141m)\nRussell was 0.101s quicker here."
    assert rw(ok).label == "pass" and rw(bad).label == "fail"


def test_no_comparison_means_not_applicable():
    assert cl.reversed_winner("RUS faster by 0.1s", [], None).label == "n/a"


# ── self-correction ──

def test_leaked_self_correction_is_caught():
    r = cl.self_correction("- **T1–T2**: ANT faster by **0.060s** — wait, corrected: **RUS** faster by 0.060s here")
    assert r.label == "fail"
    assert cl.self_correction("Exits at 123 km/h vs 143 km/h — wait, notably Hamilton's exit speed is higher").label == "fail"
    assert cl.self_correction("Russell waited for the tow before the final sector.").label == "pass"


# ── wrong driver or session ──

EVIDENCE = [{"section": "T4", "drivers": ["RUS", "HAM"], "text": "Head-to-Head at T4: RUS vs HAM\nEvent: Barcelona 2026 Qualifying"}]


def test_a_stray_driver_fails():
    r = cl.wrong_driver_or_session("Russell beat Hamilton, and Verstappen was a distant fifth.", EVIDENCE,
                                   "Why did Russell beat Hamilton?", "barcelona")
    assert r.label == "fail" and "VER" in r.explanation


def test_drivers_from_the_question_or_the_evidence_are_fine():
    ans = "Russell beat Hamilton; the question about Norris cannot be answered from this data."
    assert cl.wrong_driver_or_session(ans, EVIDENCE, "Why did Norris lose?", "barcelona").label == "pass"


def test_another_circuit_or_year_fails():
    assert cl.wrong_driver_or_session("At Monza the same pattern held.", EVIDENCE, "Russell vs Hamilton", "barcelona").label == "fail"
    assert cl.wrong_driver_or_session("In the 2025 session Russell led.", EVIDENCE, "Russell vs Hamilton", "barcelona").label == "fail"
    assert cl.wrong_driver_or_session("At Barcelona 2026 Russell led.", EVIDENCE, "Russell vs Hamilton", "barcelona").label == "pass"


# ── numbers on the right driver ──

SPEEDS = [{"section": "T4", "drivers": ["RUS", "HAM"], "text":
           "Apex speeds:\n  Apex 1 at ~1706m: RUS 147.0 km/h | HAM 152.0 km/h | advantage: HAM +5.0\n  Entry speed: RUS 284.0km/h | HAM 276.8km/h"}]


def test_a_speed_attached_to_the_wrong_driver_fails():
    r = cl.numbers_on_right_driver("Hamilton carried 147 km/h through the apex.", SPEEDS)
    assert r.label == "fail" and r.details[0]["driver"] == "HAM"


def test_the_right_speed_and_ambiguous_forms_pass():
    assert cl.numbers_on_right_driver("Hamilton carried 152 km/h through the apex.", SPEEDS).label == "pass"
    # first member of 'x vs y' and differences are not attached to one driver
    assert cl.numbers_on_right_driver("Russell edged Hamilton: 284.0 km/h vs 276.8 km/h, +7.2 km/h.", SPEEDS).label in {"n/a", "pass"}
    assert cl.numbers_on_right_driver("Hamilton was 5 km/h faster than Russell.", SPEEDS).label in {"n/a", "pass"}


# ── completeness ──

def test_a_missing_named_driver_or_corner_fails():
    r = cl.completeness("Russell was quicker through T4.", SPEEDS, DELTAS, "Russell vs Hamilton at T4")
    assert r.label == "fail" and r.details["drivers"] == ["HAM"]
    r = cl.completeness("Russell and Hamilton were close.", SPEEDS, DELTAS, "Russell vs Hamilton at T12")
    assert r.label == "fail" and r.details["corners"] == [12]


def test_first_and_last_corner_are_resolved():
    q = "Who was faster through the first corner, Russell or Hamilton?"
    assert cl.completeness("Russell and Hamilton through T1.", SPEEDS, DELTAS, q).label == "pass"
    assert cl.completeness("Russell and Hamilton through T2.", SPEEDS, DELTAS, q).label == "fail"
    q = "Who had the better exit from the final corner, Russell or Antonelli?"
    assert cl.completeness("Russell and Antonelli at T14.", SPEEDS, DELTAS, q).label == "pass"
    assert cl.completeness("The final corner is T13; Russell and Antonelli.", SPEEDS, DELTAS, q).label == "fail"


def test_completeness_not_applicable_without_names():
    assert cl.completeness("anything", [], None, "What made the pole lap quick?").label == "n/a"


# ── false premise ──

CLEAN = [{"section": "T4", "drivers": ["VER"], "text": "VER P5 Red Bull. Head-to-head at T4."}]


def test_a_premise_the_evidence_lacks_must_be_challenged():
    q = "Why did Verstappen crash in Q3 at Barcelona?"
    explained = "Verstappen crashed because he carried too much speed into T4 and lost the rear."
    challenged = "The retrieved data does not show any crash for Verstappen in Q3; he completed a normal lap and was classified P5."
    assert cl.false_premise(explained, CLEAN, q).label == "fail"
    assert cl.false_premise(challenged, CLEAN, q).label == "pass"
    assert cl.false_premise("There is no record of a crash involving Verstappen.", CLEAN, q).label == "pass"


def test_a_premise_the_evidence_supports_is_not_flagged():
    evidence = [{"section": None, "text": "Race control: car 1 crashed at turn 4, red flag"}]
    assert cl.false_premise("He crashed at turn 4.", evidence, "Why did Verstappen crash?").label == "n/a"
    assert cl.false_premise("anything", CLEAN, "Who was fastest?").label == "n/a"


def test_penalty_and_failure_premises():
    q = "Why was Hamilton penalised in Monza qualifying?"
    assert cl.false_premise("He was penalised for impeding Norris.", CLEAN, q).label == "fail"
    assert cl.false_premise("The context does not mention any penalty for Hamilton.", CLEAN, q).label == "pass"
    assert cl.false_premise("The engine failure cost him.", CLEAN, "Why did Norris's engine failure happen?").label == "fail"


# ── overclaiming ──

GAP = [{"section": "T4", "drivers": ["RUS", "HAM"], "text": "RUS: first brakes between 1551m and 1616m (gap in the car data)"}]


def test_absolute_braking_claims_in_a_gap_section_fail():
    assert cl.overclaiming("### T4\nHamilton clearly braked later than Russell here.", GAP).label == "fail"
    assert cl.overclaiming("### T4\nThis proves Russell was braking too early.", GAP).label == "fail"


def test_hedged_other_section_or_non_braking_absolutes_pass():
    assert cl.overclaiming("### T4\nWho braked later cannot be determined.", GAP).label == "pass"
    assert cl.overclaiming("### T12\nHamilton clearly braked later than Russell.", GAP).label == "pass"
    assert cl.overclaiming("### T4\nRussell clearly had the higher exit speed.", GAP).label == "pass"
    assert cl.overclaiming("anything", [{"section": "T4", "text": "first brakes at 1593m"}]).label == "n/a"


def test_a_comparison_without_driver_names_is_not_applicable_rather_than_an_error():
    assert cl.reversed_winner("RUS faster by 0.1s", [], {"sections": [{"section": "T1", "gain_s": 0.1}]}).label == "n/a"


# ── team attribution ──

GRID = [{"section": None, "chunk_type": "grid", "text": "  P3 HAD (Red Bull Racing) 1:19.303  +0.785s to pole\n  P4 LEC (Ferrari) 1:19.327\n  P1 RUS (Mercedes) 1:18.518"}]


def test_a_driver_attached_to_the_wrong_team_fails():
    r = cl.team_attribution("Leclerc was just 0.024s behind Haas's Hadjar in P3.", GRID)
    assert r.label == "fail" and r.details[0]["driver"] == "HAD" and r.details[0]["said"] == "haas"


def test_other_ways_of_naming_a_team_are_read():
    assert cl.team_attribution("Hadjar (Haas) took P3.", GRID).label == "fail"
    assert cl.team_attribution("Hadjar of McLaren took P3.", GRID).label == "fail"
    assert cl.team_attribution("The Ferrari driver Russell was quick.", GRID).label == "fail"


def test_the_right_team_passes_and_unrelated_text_is_not_applicable():
    assert cl.team_attribution("Red Bull's Hadjar and Ferrari's Leclerc were close.", GRID).label == "pass"
    assert cl.team_attribution("Hadjar (Red Bull) took P3.", GRID).label == "pass"
    assert cl.team_attribution("Hadjar took P3 and Leclerc P4.", GRID).label == "n/a"


def test_racing_bulls_and_red_bull_are_not_confused():
    ev = [{"section": None, "text": "  P8 LAW (Racing Bulls) 1:19.994\n  P3 HAD (Red Bull Racing) 1:19.303"}]
    assert cl.team_attribution("Racing Bulls' Lawson and Red Bull's Hadjar", ev).label == "pass"
    assert cl.team_attribution("Red Bull's Lawson", ev).label == "fail"


# ── regressions from the data-generated held-out run ──

def test_took_longer_means_slower_not_winner():
    deltas = {"driver_a": "RUS", "driver_b": "ALB", "lap_gap_s": 2.4, "sections": [{"section": "T4-T5", "gain_s": 0.461, "start_m": 1138.0}]}
    ok = "- **T4-T5 section:** Albon took 0.461s longer (6.957s vs 6.496s)."
    bad = "- **T4-T5 section:** Russell took 0.461s longer (6.957s vs 6.496s)."
    assert cl.reversed_winner(ok, [], deltas).label == "pass"
    assert cl.reversed_winner(bad, [], deltas).label == "fail"


def test_a_speed_in_a_one_driver_chunk_belongs_to_that_driver():
    pair = {"section": "T2-T3", "drivers": ["VER", "PER"], "text": "Top speed: VER 346.0 km/h | PER 340.0 km/h"}
    summary = {"section": None, "drivers": ["PER"], "text": "Speed trap (speed trap): 346.0 km/h\nSpeed trap 1: 324.0 km/h"}
    ans = "The speed traps show VER ahead (326/338/347 km/h against PER's 324/337/346 km/h)."
    assert cl.numbers_on_right_driver(ans, [pair, summary]).label in {"pass", "n/a"}
    swapped = "PER reached 346 km/h in the trap."
    assert cl.numbers_on_right_driver(swapped, [pair]).label == "fail"                      # without the summary it is still caught


def test_a_braking_speed_on_a_line_led_by_the_drivers_code_belongs_to_that_driver():
    corner = {"text": "Head-to-Head at T11\n  ALO: first brakes at 5018m from 308.0 km/h | braking distance 128m\n  STR: first brakes at 5031m from 290.0 km/h",
              "drivers": ["ALO", "STR"], "chunk_type": "head_to_head_event"}
    straight = {"text": "Top speed: ALO 311.0km/h | STR 308.0km/h", "drivers": ["ALO", "STR"], "chunk_type": "head_to_head_event"}
    values = cl.driver_speed_values([corner, straight])
    assert 308.0 in values["ALO"] and 290.0 in values["STR"]
    answer = "ALO first braked at 5018m from 308.0 km/h, STR from 290.0 km/h."
    assert cl.numbers_on_right_driver(answer, [corner, straight]).label == "pass"


@pytest.mark.parametrize("text", ["The last driver through was Lance Stroll... no, the context does not name him.",
                                  "It was Perez… actually Hulkenberg."])
def test_an_ellipsis_followed_by_a_correction_is_a_self_correction(text):
    assert cl.self_correction(text).label == "fail"


def test_a_speed_after_a_sentence_break_is_not_attached_to_the_driver_named_before_it():
    src = [{"text": "ALB 147.0 km/h exit | HUL 130.0 km/h exit", "drivers": ["ALB", "HUL"], "chunk_type": "head_to_head_event"}]
    answer = "Hülkenberg picked up the throttle earlier, at 53m versus Albon's 63m. His exit speed was 130.0 km/h against Albon's 147.0 km/h."
    assert cl.numbers_on_right_driver(answer, src).label != "fail"


# ── team ranking and tyre claims ──

TEAM_GRID = [{"chunk_type": "grid", "drivers": [], "text": (
    "Qualifying grid - Melbourne (best lap of each driver)\n"
    "  P1 RUS (Mercedes) 1:18.518  +0.000s to pole\n  P2 ANT (Mercedes) 1:18.811  +0.293s to pole\n"
    "  P3 HAD (Red Bull Racing) 1:19.303  +0.785s to pole\n  P4 LEC (Ferrari) 1:19.327  +0.809s to pole\n"
    "  P5 PIA (McLaren) 1:19.380  +0.862s to pole\n  P6 NOR (McLaren) 1:19.475  +0.957s to pole\n"
    "  P7 HAM (Ferrari) 1:19.478  +0.960s to pole\n")}]


def test_ferrari_called_fourth_quickest_team_is_checked_against_the_grid():
    # by best car: Mercedes 1, Red Bull 2, Ferrari 3, McLaren 4. By average Ferrari (79.40) edges McLaren (79.43): also 3.
    r = cl.team_ranking("Ferrari were the fourth quickest team in qualifying.", TEAM_GRID)
    assert r.label == "fail" and "Ferrari" in r.explanation or "ferrari" in r.explanation
    assert cl.team_ranking("Ferrari were the third quickest team.", TEAM_GRID).label == "pass"
    assert cl.team_ranking("McLaren were the fourth-fastest team on the grid.", TEAM_GRID).label == "pass"


def test_team_ranking_is_not_checked_without_a_grid_or_a_ranking():
    assert cl.team_ranking("Ferrari were the fourth quickest team.", []).label == "n/a"
    assert cl.team_ranking("Ferrari were quick.", TEAM_GRID).label == "n/a"
    assert cl.team_ranking("Ferrari and McLaren were the third quickest teams.", TEAM_GRID).label == "n/a"      # two teams: ambiguous


SAME_TYRES = [{"chunk_type": "driver_lap_summary", "drivers": ["RUS"], "text": "Qualifying Lap Summary: RUS\nTyre: SOFT (1 laps old) | Fresh set"},
              {"chunk_type": "driver_lap_summary", "drivers": ["HAM"], "text": "Qualifying Lap Summary: HAM\nTyre: SOFT (1 laps old) | Fresh set"}]
DIFFERENT_TYRES = [SAME_TYRES[0], {"chunk_type": "driver_lap_summary", "drivers": ["HAM"], "text": "Tyre: MEDIUM (4 laps old) | Used set"}]


def test_blaming_the_tyres_when_both_ran_the_same_fresh_softs_fails():
    answer = "Hamilton lost time because of tyre wear compared with Russell, which explains the gap."
    r = cl.tyre_claims(answer, SAME_TYRES)
    assert r.label == "fail" and "same" in r.explanation
    assert cl.tyre_claims(answer, DIFFERENT_TYRES).label == "pass"
    assert cl.tyre_claims(answer, []).label == "fail"                                   # no tyre information at all


def test_saying_the_tyres_were_the_same_is_not_a_tyre_claim():
    assert cl.tyre_claims("Both ran the same fresh softs, so the gap is not due to the tyres.", SAME_TYRES).label == "n/a"
    assert cl.tyre_claims("Hamilton was 0.2s slower through T3.", SAME_TYRES).label == "n/a"
