"""Offline tests for evals/claim_checks.py. Several cases are real sentences from experiment answers."""

import sys
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
