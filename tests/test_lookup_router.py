"""
Tests for the lookup router: one-fact questions answered from the session data with no model call.

No store, key or model is needed: the session context is built by hand. The router has to be right or silent, so most
of these tests check that it stays silent, including the two wrong answers found when it was first run on the question sets.

Run from the project root:
    pytest tests/test_lookup_router.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from events_and_chunking.lookup_router import matched_intents, route  # noqa: E402


def drv(team, pos, stage, times, speed=None):
    best = min(times.values())
    return {"team": team, "qualifying_position": pos, "stage": stage, "lap_time_seconds": best,
            "session_times": dict(times), "speed_trap_kmh": speed}


CTX = {
    "drivers": {
        "NOR": drv("McLaren", 1, "Q3", {1: 79.911, 2: 79.727, 3: 79.327}, 330.0),
        "PIA": drv("McLaren", 2, "Q3", {1: 80.076, 2: 79.808, 3: 79.436}, 331.0),
        "HUL": drv("Haas", 3, "Q3", {1: 80.781, 2: 80.411, 3: 80.339}, 329.0),
        "GAS": drv("Alpine", 4, "Q2", {1: 80.500, 2: 80.600}, 331.0),
        "STR": drv("Aston Martin", 5, "Q1", {1: 81.9}, 327.0),
        "LEC": {**drv("Ferrari", 6, "Q3", {1: 80.0, 2: 80.1}), "speed_trap_kmh": 326.0},   # reached Q3, set no Q3 time
    },
    "no_time": {"ZHO": {"team": "Sauber", "reason": "stopped on track"}},
    "entrants": {"NOR", "PIA", "HUL", "GAS", "STR", "LEC", "ZHO"},
    "standings": ["NOR", "PIA", "HUL", "GAS", "STR", "LEC"],
    "pole": "NOR",
    "fastest_sector": {1: (26.223, "NOR"), 2: (30.1, "PIA")},
}


def ask(question, *named):
    return route(question, CTX, set(named))


# ── what it answers ──

def test_best_lap_names_the_driver_team_and_session():
    r = ask("What was Hülkenberg's best qualifying lap?", "HUL")
    assert r.answer == "HUL (Haas) set a best lap of 1:20.339 in Q3."
    assert r.route == "lap_time" and r.drivers == ["HUL"]


def test_a_named_session_gives_that_session_lap():
    assert ask("what was norris's best lap in Q1", "NOR").answer == "NOR (McLaren) set a Q1 lap of 1:19.911."
    assert ask("Lindblad Q2 time", "GAS").answer == "GAS (Alpine) set a Q2 lap of 1:20.600."


def test_a_session_the_driver_has_no_time_in_goes_to_the_model():
    assert ask("what was Gasly's best lap in Q3", "GAS") is None          # the model explains why there is no lap


def test_grid_position_says_before_penalties():
    assert ask("Where did Gasly qualify?", "GAS").answer == "GAS (Alpine) qualified P4, before any grid penalties."
    assert "before any grid penalties" in ask("What was Strolls's grid position?", "STR").answer


def test_gap_to_pole_uses_the_best_laps_and_names_the_sessions():
    r = ask("How far off pole was Gasly?", "GAS")
    assert r.answer == "GAS (Alpine) was 1.173s off pole (GAS's Q1 lap against NOR's Q3 lap): 1:20.500 against NOR's 1:19.327."
    assert r.drivers == ["GAS", "NOR"]


def test_pole_sitter_has_no_gap_to_pole():
    assert "took pole" in ask("What was Norris's gap to pole?", "NOR").answer


def test_who_got_pole_and_who_was_in_a_position():
    assert ask("who got pole?").answer == "NOR (McLaren) took pole with a 1:19.327 in Q3."
    assert ask("who set the fastest lap in qualifying").route == "pole"
    assert ask("who was P3?").answer == "HUL (Haas) qualified P3, before any grid penalties."
    assert ask("who finished 2nd in qualifying").answer.startswith("PIA (McLaren) qualified P2")


def test_stage_questions():
    assert ask("Did Gasly make it into Q3?", "GAS").answer == "GAS (Alpine) did not make Q3; he went out in Q2, qualifying P4."
    assert ask("Did Hulkenberg make Q3", "HUL").answer == "HUL (Haas) made Q3, qualifying P3."
    assert ask("How far did Strolls get?", "STR").answer == "STR (Aston Martin) went out in Q1, qualifying P5."
    assert ask("did Zhou make Q2", "ZHO").answer == "ZHO (Sauber) did not make Q2; he went out in Q1, having set no timed lap."


def test_a_driver_who_reached_q3_but_set_no_q3_time_is_not_called_eliminated():
    assert ask("Did Leclerc make it to Q3?", "LEC").answer == "LEC (Ferrari) made Q3 but set no time in Q3, qualifying P6."


def test_last_into_q3_uses_q2_times():
    r = ask("Who was the last driver to get through to Q3?")
    assert r.route == "last_into_q3"
    assert r.answer == "HUL (Haas) was last into Q3, with a Q2 time of 1:20.411."


def test_quickest_in_a_session():
    assert ask("who topped Q1?").answer == "NOR (McLaren) was quickest in Q1, with 1:19.911."


def test_speed_trap_for_the_field_and_for_a_driver():
    ctx = {**CTX, "drivers": {**CTX["drivers"], "GAS": {**CTX["drivers"]["GAS"], "speed_trap_kmh": 328.0}}}
    r = route("Who recorded the highest speed trap reading?", ctx, set())
    assert r.answer == "PIA (McLaren) had the highest speed trap reading, 331.0 km/h."
    assert ask("what was Norris's speed trap?", "NOR").answer == "NOR (McLaren)'s speed trap reading was 330.0 km/h."


def test_a_tied_speed_trap_names_everyone_tied():
    r = ask("fastest through the speed trap?")                      # PIA and GAS both 331.0 in the test data
    assert r.answer == "PIA (McLaren) and GAS (Alpine) tied for the highest speed trap reading, 331.0 km/h."
    assert r.drivers == ["PIA", "GAS"]


def test_fastest_sector():
    assert ask("Who set the quickest sector 1 time?").answer == "NOR (McLaren) set the quickest sector 1 time, 26.223s."
    assert ask("who was fastest in sector 2").drivers == ["PIA"]


# ── when it must stay silent ──

@pytest.mark.parametrize("question,named", [
    # analysis questions that happen to contain a lookup phrase (the first wrong answer found on the question sets)
    ("where was bearman strongest on his best lap at barcelona", ("HUL",)),
    ("why was Norris's best lap so much quicker", ("NOR",)),
    ("how did Norris set his best lap", ("NOR",)),
    # topics the router has no figures for (the second wrong answer found: a tyre question read as a Q3 lap lookup)
    ("was leclerc on a new or used set for his final Q3 lap", ("LEC",)),
    ("what compound was Norris on for his best lap", ("NOR",)),
    ("what was Norris's best lap before the red flag", ("NOR",)),
    ("did Gasly get a penalty for his best lap", ("GAS",)),
    # comparisons
    ("what was Norris's best lap compared to Piastri", ("NOR", "PIA")),
    ("who was faster, Norris or Piastri", ("NOR", "PIA")),
    ("what is the gap between Norris and Piastri", ("NOR", "PIA")),
    # two questions in one
    ("what was Norris's best lap and what tyre was he on", ("NOR",)),
    ("what was Norris's grid position and his gap to pole", ("NOR",)),
    ("what was Norris's best lap? And Piastri's?", ("NOR",)),
    ("what was Gasly's best lap; where did he qualify", ("GAS",)),
    # no intent the router knows
    ("tell me about Norris", ("NOR",)),
    ("top speed for Norris", ("NOR",)),
    ("Sector 3 time for Gasly", ("GAS",)),
    ("", ()),
])
def test_it_stays_silent(question, named):
    assert ask(question, *named) is None


def test_one_driver_needed_but_none_named_goes_to_the_model():
    assert ask("What was the best lap?") is None
    assert ask("where did he qualify?") is None


def test_a_driver_not_in_the_session_goes_to_the_model():
    assert route("What was Sainz's best lap?", CTX, {"SAI"}) is None


def test_a_driver_with_no_timed_lap_is_not_given_a_lap_time():
    assert ask("What was Zhou's best lap?", "ZHO") is None
    assert ask("Where did Zhou qualify?", "ZHO") is None


def test_last_into_q3_needs_everyones_q2_time_or_it_is_silent():
    ctx = {**CTX, "drivers": {**CTX["drivers"], "HUL": {**CTX["drivers"]["HUL"], "session_times": {3: 80.339}}}}
    assert route("who was last into Q3", ctx, set()) is None


def test_a_tie_for_last_into_q3_is_not_guessed():
    drivers = {"A": drv("X", 1, "Q3", {2: 80.0, 3: 79.0}), "B": drv("Y", 2, "Q3", {2: 80.0, 3: 79.5})}
    ctx = {"drivers": drivers, "entrants": {"A", "B"}, "standings": ["A", "B"], "pole": "A"}
    assert route("who was last into Q3", ctx, set()) is None


def test_position_data_that_does_not_match_is_not_used():
    ctx = {**CTX, "standings": ["PIA", "NOR", "HUL"]}                     # order disagrees with qualifying_position
    assert route("who was P1?", ctx, set()) is None


def test_intent_detection_is_exposed_for_the_evaluation():
    assert matched_intents("where did Gasly qualify") == ["position"]
    assert matched_intents("what was his gap to pole and grid position") == ["position", "gap_to_pole"]
