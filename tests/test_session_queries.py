"""Whole-session views: grid, team, section ranking, team advantage, and the planner that picks them."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RAG_data_layers/F1 Project Thoughts"))
sys.path.insert(0, str(ROOT / "tests"))

import query  # noqa: E402
import rag_retrieval as rr  # noqa: E402
import test_session_snapshot as tss  # noqa: E402
from events_and_chunking import session_queries as sq  # noqa: E402
from events_and_chunking import session_snapshot as snap  # noqa: E402
from events_and_chunking.chunk_telemetry import build_telemetry_chunks  # noqa: E402
from events_and_chunking.detect_events_official import detect_official_events  # noqa: E402

CTX = {"drivers": {
    "AAA": {"lap_time_seconds": 80.000, "qualifying_position": 1, "team": "Alpha"},
    "BBB": {"lap_time_seconds": 80.050, "qualifying_position": 2, "team": "Alpha"},
    "CCC": {"lap_time_seconds": 80.053, "qualifying_position": 3, "team": "Beta"},
    "DDD": {"lap_time_seconds": 80.900, "qualifying_position": 4, "team": "Beta"},
    "EEE": {"lap_time_seconds": 80.700, "qualifying_position": 5, "team": "Gamma"},          # classified behind DDD, quicker
}}


# ── grid ──

def test_grid_lists_every_driver_with_gaps_and_flags_the_tight_ones():
    text = sq.grid_text(CTX, "Testland 2026")
    assert "Qualifying grid - Testland 2026" in text
    assert "P1 AAA (Alpha) 1:20.000  +0.000s to pole" in text
    assert "P3 CCC (Beta) 1:20.053  +0.053s to pole, +0.003s to the car ahead" in text
    assert "AAA/BBB 0.050s" in text and "BBB/CCC 0.003s" in text and "less than 0.100s" in text
    assert "Spread of the quickest 5 laps: 0.900s" in text


def test_grid_notes_when_classification_differs_from_best_lap_order():
    assert "Classification order differs from best-lap order around: DDD" in sq.grid_text(CTX)
    assert "differs" not in sq.grid_text({"drivers": {k: v for k, v in CTX["drivers"].items() if k in {"AAA", "BBB", "CCC"}}})


def test_grid_ignores_drivers_without_a_lap_and_empty_sessions():
    ctx = {"drivers": {**CTX["drivers"], "ZZZ": {"team": "Zed"}}}
    assert "ZZZ" not in sq.grid_text(ctx)
    assert sq.grid_text({"drivers": {}}) is None


# ── teams ──

def test_team_summary_has_both_cars_the_gap_and_the_best_car():
    text = sq.team_text(CTX, "Alpha")
    assert "AAA: P1, best lap 1:20.000, +0.000s to pole" in text
    assert "BBB: P2, best lap 1:20.050, +0.050s to pole" in text
    assert "Between teammates: AAA was 0.050s quicker." in text and "Best car: AAA (P1)." in text
    assert sq.team_text(CTX, "Nobody") is None


# ── sections across the field ──

@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    tel = {d: tss.telemetry(i, 1.0 - 0.01 * i) for i, (d, _, _) in enumerate(tss.DRIVERS)}
    for d, t in tel.items():
        t["Driver"] = d
    fastest = pd.DataFrame({"Driver": [d for d, _, _ in tss.DRIVERS], "Team": [t for _, t, _ in tss.DRIVERS],
                            "QualifyingPosition": [p for _, _, p in tss.DRIVERS]})
    events = detect_official_events(tel, [{"number": 1, "label": "1", "dist": 1000.0}, {"number": 2, "label": "2", "dist": 1850.0}], tss.LAP)
    path = tmp_path_factory.mktemp("sq") / "s.npz"
    build_telemetry_chunks({"driver_fastest_laps": fastest, "telemetry": tel}, events,
                           {"year": 2026, "circuit": "Testland", "session": "Qualifying"}, head_to_head_top_n=2, snapshot_path=str(path))
    return snap.load_snapshot(path)


TEAM_CTX = {"drivers": {d: {"team": t, "lap_time_seconds": 80 + i * 0.1, "qualifying_position": p}
                        for i, (d, t, p) in enumerate(tss.DRIVERS)}}


def test_section_times_cover_every_section_and_driver(snapshot):
    sections = sq.section_times(snapshot)
    assert len(sections) == len(snapshot.events)
    assert all(set(s["times"]) == set(snapshot.telemetry) for s in sections)
    assert all(t > 0 for s in sections for t in s["times"].values())


def test_section_ranking_is_ordered_quickest_first_with_gaps(snapshot):
    sections = sq.section_times(snapshot)
    corner = sq.sections_for_corner(sections, 1)[0]
    text = sq.section_ranking_text(corner, TEAM_CTX)
    lines = [l for l in text.split("\n") if l.strip().startswith(tuple("12345"))]
    times = [float(l.split()[-1].rstrip("s")) if "+" not in l else float(l.split()[-2].rstrip("s")) for l in lines]
    assert len(lines) == 5 and times == sorted(times)
    assert lines[0].strip().startswith("1.") and "+" not in lines[0]


def test_team_advantage_names_a_leader_per_section_and_totals_them(snapshot):
    text = sq.team_advantage_text(snapshot, TEAM_CTX)
    assert text.startswith("Team advantage by section")
    assert sum(1 for l in text.split("\n") if "ahead of" in l) >= len(snapshot.events) - 1
    assert "Sections led by each team:" in text


# ── the planner ──

@pytest.mark.parametrize("question,expect", [
    ("where did each team have an advantage in melbourne", {"team_advantage": True, "grid": False}),
    ("why were many drivers close together in barca", {"team_advantage": False, "grid": True}),
    ("what was the gap between P4 and P5", {"team_advantage": False, "grid": True}),
    ("Where did Hamilton lose time to Russell?", {"team_advantage": False, "grid": False}),
])
def test_planner_picks_whole_session_views(question, expect):
    p = sq.plan(question, [], no_driver_named="hamilton" not in question.lower())
    assert {k: p[k] for k in expect} == expect


def test_a_named_team_adds_its_summary_and_the_grid_but_not_team_advantage():
    p = sq.plan("each team vs the ferraris", ["Ferrari"], no_driver_named=False)
    assert p["teams"] == ["Ferrari"] and p["grid"] and not p["team_advantage"]


def test_section_best_only_when_no_driver_is_named():
    assert sq.plan("who was fastest through T9", [], no_driver_named=True)["section_best"]
    assert not sq.plan("who was fastest through T9, Norris or Piastri", [], no_driver_named=False)["section_best"]


def test_corner_numbers_asked(snapshot):
    sections = sq.section_times(snapshot)
    assert sq.corner_numbers_asked("fastest through T2", sections) == [2]
    assert sq.corner_numbers_asked("best exit from turn 1", sections) == [1]
    assert sq.corner_numbers_asked("who was quickest through the first corner", sections) == [1]
    assert sq.corner_numbers_asked("who was quickest through the final corner", sections) == [2]
    assert sq.corner_numbers_asked("who was quickest", sections) == []


# ── the context budget ──

@pytest.mark.parametrize("n,extras,pair", [(6, 0, False), (6, 0, True), (6, 3, True), (6, 3, False), (6, 1, True), (3, 3, True), (2, 3, True), (8, 2, True)])
def test_the_budget_always_adds_up_and_keeps_room_for_ordinary_retrieval(n, extras, pair):
    e, p, r = query.split_budget(n, extras, pair)
    assert e + p + r == n and min(e, p, r) >= 0
    assert r >= min(query.MIN_RETRIEVED, n)
    assert e <= extras and (p == 0 or pair)


def test_whole_session_blocks_come_before_the_pair_sections():
    assert query.split_budget(6, 2, True) == (2, 2, 2)       # team summary + grid, two pair sections, two retrieved
    assert query.split_budget(6, 0, True) == (0, 4, 2)       # a plain pair question: four sections, two retrieved
    assert query.split_budget(6, 3, False) == (3, 0, 3)      # no pair: the blocks plus retrieval


# ── named corners: resolve, pin, and state the facts ──

def test_corner_facts_name_the_first_and_last_corner():
    sections = [{"id": "T1-T2", "corner_numbers": [1, 2]}, {"id": "T3", "corner_numbers": [3]}, {"id": "T13", "corner_numbers": [13]},
                {"id": "T14", "corner_numbers": [14]}, {"id": "Main straight", "corner_numbers": []}]
    text = sq.corner_facts_text(sections)
    assert "T1 to T14" in text and "The first corner is T1 (section T1-T2)" in text and "The final corner is T14 (section T14)" in text
    assert sq.corner_facts_text([{"id": "Main straight", "corner_numbers": []}]) is None


def test_the_planner_asks_for_corner_facts_only_when_a_corner_is_named():
    assert sq.plan("who had the better exit from the final corner", [], False)["corner_facts"]
    assert sq.plan("compare them through turn 9", [], False)["corner_facts"]
    assert sq.plan("how was the first corner", [], False)["corner_facts"]
    assert not sq.plan("where did Hamilton lose time to Russell", [], False)["corner_facts"]


def test_the_final_corner_is_pinned_into_the_pair_sections_on_the_real_melbourne_lap():
    pinned = query.pinned_sections("Who had the better exit from the final corner in Melbourne, Russell or Antonelli?", "melbourne")
    assert pinned == ("T14",)
    top = query.top_pair_sections("melbourne", {"RUS", "ANT"}, k=4, pin=pinned)
    assert top[0]["metadata"]["event_id"] == "T14" and len(top) == 4
    by_gap = query.top_pair_sections("melbourne", {"RUS", "ANT"}, k=4)
    assert "T14" not in [c["metadata"]["event_id"] for c in by_gap]                 # the old behaviour: T14 was left out


def test_a_turn_inside_a_merged_section_pins_that_section():
    assert query.pinned_sections("who was quicker through turn 9 at Monza", "monza") == ("T8-T9-T10",)
    assert query.pinned_sections("how was the first corner", "barcelona") == ("T1-T2",)
    assert query.pinned_sections("where did Hamilton lose time", "barcelona") == ()


def test_pinned_sections_come_first_and_the_rest_follow_by_time_gap():
    top = query.top_pair_sections("barcelona", {"RUS", "HAM"}, k=4, pin=("T9",))
    ids = [c["metadata"]["event_id"] for c in top]
    assert ids[0] == "T9"
    rest = [abs(c["metadata"]["time_delta_s"]) for c in top[1:]]
    assert rest == sorted(rest, reverse=True)


# ── speed traps, qualifying stage, names ──

STAGED = {"drivers": {
    "AAA": {"lap_time_seconds": 80.0, "qualifying_position": 1, "team": "Alpha", "speed_trap_kmh": 340.0, "stage": "Q3"},
    "BBB": {"lap_time_seconds": 80.1, "qualifying_position": 2, "team": "Alpha", "speed_trap_kmh": 353.0, "stage": "Q3"},
    "CCC": {"lap_time_seconds": 80.5, "qualifying_position": 3, "team": "Beta", "speed_trap_kmh": 350.0, "stage": "Q3"},
    "DDD": {"lap_time_seconds": 81.0, "qualifying_position": 4, "team": "Beta", "speed_trap_kmh": 345.0, "stage": "Q2"},
    "EEE": {"lap_time_seconds": 82.0, "qualifying_position": 5, "team": "Gamma", "speed_trap_kmh": 330.0, "stage": "Q1"}}}


def test_the_grid_shows_each_cars_speed_trap_and_stage():
    text = sq.grid_text(STAGED)
    assert "P2 BBB (Alpha) 1:20.100  +0.100s to pole, +0.100s to the car ahead; speed trap 353 km/h; reached Q3" in text
    assert "P4 DDD (Beta)" in text and "; out in Q2" in text and "; out in Q1" in text


def test_the_grid_states_the_highest_speed_traps_and_the_qualifying_progression():
    text = sq.grid_text(STAGED)
    assert "Highest speed-trap readings: BBB 353 km/h, CCC 350 km/h, DDD 345 km/h (of 5 drivers)" in text
    assert "3 reached Q3" in text and "1 went out in Q2" in text and "1 went out in Q1" in text


def test_a_grid_without_those_fields_is_unchanged():
    text = sq.grid_text(CTX)
    assert "speed trap" not in text and "Qualifying progression" not in text


@pytest.mark.parametrize("question", ["Who recorded the highest speed trap reading?", "which driver was last into Q3", "who was eliminated in Q1",
                                      "what was the top speed of the field", "who was knocked out in Q2"])
def test_field_wide_lookups_ask_for_the_grid(question):
    assert sq.plan(question, [], True)["grid"]


def test_the_stored_progression_and_speed_trap_are_parsed_from_the_lap_summary():
    doc = ("Qualifying position: P10\nQualifying progression: Qualified for Q3\nSpeed trap (speed trap): 346.0 km/h\n")
    assert query.parse_stage(doc) == "Q3" and query.parse_speed_trap(doc) == 346.0
    assert query.parse_stage("Qualifying progression: Eliminated at end of Q2 — did not qualify for Q3") == "Q2"
    assert query.parse_stage("Qualifying progression: Eliminated at end of Q1 — did not reach Q2 or Q3") == "Q1"
    assert query.parse_stage("no progression here") is None and query.parse_speed_trap("nothing") is None


def test_the_prompt_tells_the_model_which_code_a_first_name_means():
    note = rr.driver_key_note("What was Sergio's grid position at Albert Park?")
    assert note == "Names in the question: 'Sergio' means PER (Perez)."
    assert rr.driver_key_note("Where did Perez qualify?") is None                  # a surname needs no key
    assert "'Max' means VER" in rr.driver_key_note("Max vs Lewis") and "'Lewis' means HAM" in rr.driver_key_note("Max vs Lewis")
    assert rr.driver_key_note("Who was fastest?") is None
    prompt = rr.format_prompt("What was Sergio's grid position?", [], session_ctx=None)
    assert "[NOTE] Names in the question: 'Sergio' means PER (Perez)." in prompt


# ── who reached Q3, last into Q3, and like-for-like comparisons ──

def _timed(**kw):
    base = {"AAA": (80.0, 1, "Q3", {1: 81.0, 2: 80.5, 3: 80.0}), "BBB": (80.6, 2, "Q3", {1: 81.2, 2: 80.9, 3: 80.6}),
            "CCC": (80.4, 10, "Q3", {1: 81.5, 2: 80.4})}                                  # CCC: P10, crashed in Q3, no Q3 time
    base.update(kw)
    return {"drivers": {c: {"lap_time_seconds": t, "qualifying_position": p, "team": c, "stage": st, "session_times": ts}
                        for c, (t, p, st, ts) in base.items()}}


def test_a_top_ten_car_with_no_q3_time_is_shown_as_having_reached_q3():
    text = sq.grid_text(_timed())
    assert "CCC" in text and "reached Q3 but set no Q3 time" in text


def test_last_into_q3_comes_from_the_q2_times_not_the_classification():
    # BBB is P2 on the Q3 order but CCC (P10) has the quicker Q2 time; the slowest Q2 time among the advancers is BBB's 1:20.900
    text = sq.grid_text(_timed())
    assert "the slowest Q2 time among them: BBB 1:20.900" in text
    assert "decided by the Q2 times" in text


def test_the_grid_warns_that_gaps_to_pole_mix_sessions_when_cars_went_out_early():
    text = sq.grid_text(_timed(DDD=(82.0, 20, "Q1", {1: 82.0})))
    assert "mixes sessions" in text and "went out in Q1" in text


def test_session_times_are_parsed_from_the_lap_summary():
    doc = "Qualifying progression: Qualified for Q3 (set no time in Q3; recorded as a DNF)\nBest time in each session: Q1 1:15.964; Q2 1:15.281; Q3 no time set\n"
    assert query.parse_stage(doc) == "Q3"
    assert query.parse_session_times(doc) == {1: pytest.approx(75.964), 2: pytest.approx(75.281)}
    assert query.parse_session_times("nothing") == {}


def test_a_comparison_across_sessions_also_gives_the_like_for_like_gap():
    ctx = {"drivers": {"LEC": {"lap_time_seconds": 75.281, "qualifying_position": 10, "stage": "Q3", "session_times": {1: 75.964, 2: 75.281}},
                       "ALO": {"lap_time_seconds": 78.815, "qualifying_position": 22, "stage": "Q1", "session_times": {1: 78.815}}}}
    text = query.lap_comparison_chunk({"driver_a": "LEC", "driver_b": "ALO"}, ctx)["text"]
    assert "not like-for-like" in text
    assert "Compare them in Q1, the last session in which both set a time: LEC 1:15.964, ALO 1:18.815; LEC was 2.851s quicker." in text
    assert "ALO went out in Q1" in text


def test_two_q3_drivers_are_compared_in_q3_and_a_slower_q3_lap_is_called_out():
    ctx = {"drivers": {"LAW": {"lap_time_seconds": 79.994, "qualifying_position": 8, "stage": "Q3", "session_times": {1: 80.491, 2: 80.144, 3: 79.994}},
                       "LIN": {"lap_time_seconds": 79.971, "qualifying_position": 9, "stage": "Q3", "session_times": {1: 80.409, 2: 79.971, 3: 81.247}}}}
    text = query.lap_comparison_chunk({"driver_a": "LAW", "driver_b": "LIN"}, ctx)["text"]
    assert "Compare them in Q3, the last session in which both set a time: LAW 1:19.994, LIN 1:21.247; LAW was 1.253s quicker." in text
    assert "LIN's Q3 lap was 1.276s slower than his Q2 lap (1:19.971)." in text
    assert "went out" not in text


def test_a_comparison_within_one_session_is_unchanged():
    ctx = {"drivers": {"AAA": {"lap_time_seconds": 80.0, "qualifying_position": 1, "session_times": {1: 81.0, 3: 80.0}},
                       "BBB": {"lap_time_seconds": 80.6, "qualifying_position": 2, "session_times": {1: 81.2, 3: 80.6}}}}
    text = query.lap_comparison_chunk({"driver_a": "AAA", "driver_b": "BBB"}, ctx)["text"]
    assert "like-for-like" not in text


def test_the_lap_summary_states_entry_to_q3_and_the_time_set_in_each_session():
    from events_and_chunking.chunk_driver import progression_lines
    crashed = pd.Series({"QSessionReached": "Q3", "Q1Time": 75.964, "Q2Time": 75.281, "Q3Time": float("nan"), "QCrashed": True})
    assert progression_lines(crashed) == ["Qualifying progression: Qualified for Q3 (set no time in Q3; recorded as a DNF)",
                                          "Best time in each session: Q1 1:15.964; Q2 1:15.281; Q3 no time set"]
    out_q1 = pd.Series({"QSessionReached": "Q1", "Q1Time": 78.815, "Q2Time": float("nan"), "Q3Time": float("nan"), "QCrashed": False})
    assert progression_lines(out_q1)[0].startswith("Qualifying progression: Eliminated at end of Q1")
    assert progression_lines(out_q1)[1] == "Best time in each session: Q1 1:18.815"
    # a source without per-session times (the FastF1 path) keeps the plain status line
    assert progression_lines(pd.Series({"QSessionReached": "Q3"})) == ["Qualifying progression: Qualified for Q3"]


def test_the_gap_to_the_car_ahead_uses_one_session_and_is_labelled():
    ctx = _timed(DDD=(80.7, 3, "Q3", {1: 81.3, 2: 80.8, 3: 80.7}))
    ctx["drivers"]["DDD"]["qualifying_position"] = 3
    ctx["drivers"]["BBB"]["qualifying_position"] = 2
    text = sq.grid_text(ctx)
    ddd = next(l for l in text.splitlines() if " DDD " in l)
    assert "+0.100s to the car ahead (both Q3 times)" in ddd              # 80.7 - 80.6, both Q3 laps
    ccc = next(l for l in text.splitlines() if " CCC " in l)
    assert "no like-for-like gap to the car ahead" in ccc                  # CCC set no Q3 time: nothing to subtract


def test_cars_from_different_sessions_are_not_listed_as_separated_by_a_tiny_gap():
    ctx = {"drivers": {"AAA": {"lap_time_seconds": 79.994, "qualifying_position": 8, "stage": "Q3", "session_times": {1: 80.5, 2: 80.1, 3: 79.994}},
                       "BBB": {"lap_time_seconds": 79.971, "qualifying_position": 9, "stage": "Q3", "session_times": {1: 80.4, 2: 79.971, 3: 81.2}}}}
    assert "separated by less than" not in sq.grid_text(ctx)


# ── gap to pole is the best lap against the pole lap ──

@pytest.mark.parametrize("question,expected", [("What was Nico's gap to pole in qualifying at Melbourne?", True),
                                               ("How far off pole was Oscar at Monza?", True),
                                               ("How far behind pole was Lewis?", True),
                                               ("Why didn't Lewis get pole at the Spanish Grand Prix?", False),
                                               ("How far apart were Bortoleto and Lindblad?", False)])
def test_pole_gap_questions_are_recognised(question, expected):
    assert query.is_pole_gap_question(question) is expected


def test_a_pole_gap_question_gets_the_best_lap_gap_not_the_like_for_like_one():
    ctx = {"pole": "RUS", "drivers": {
        "RUS": {"lap_time_seconds": 78.518, "qualifying_position": 1, "stage": "Q3", "session_times": {1: 79.507, 2: 78.934, 3: 78.518}},
        "HUL": {"lap_time_seconds": 80.303, "qualifying_position": 11, "stage": "Q2", "session_times": {1: 81.024, 2: 80.303}}}}
    deltas = {"driver_a": "RUS", "driver_b": "HUL", "lap_session": None}
    text = query.lap_comparison_chunk(deltas, ctx, "What was Nico's gap to pole?")["text"]
    assert "Gap to pole: HUL's best lap 1:20.303 (Q2) against RUS's pole lap 1:18.518 (Q3) is 1.785s. Give this as the gap to pole." in text
    assert "like-for-like" not in text and "compare the two best laps" in text
    pair = query.lap_comparison_chunk(deltas, ctx, "How far apart were Russell and Hulkenberg?")["text"]
    assert "Compare them in Q2" in pair


def test_the_field_wide_fastest_sectors_are_in_the_grid():
    ctx = dict(STAGED, fastest_sector={1: (26.5, "AAA"), 2: (17.284, "BBB")})
    text = sq.grid_text(ctx)
    assert "Fastest sector times in the session, over all 5 drivers: sector 1 26.500s (AAA), sector 2 17.284s (BBB)" in text


@pytest.mark.parametrize("question", ["Who set the quickest sector 2 time in qualifying at Albert Park?", "Who had the fastest sector 3?"])
def test_fastest_sector_questions_ask_for_the_grid(question):
    assert sq.plan(question, [], True)["grid"]


# ── one-fact questions get short answers ──

@pytest.mark.parametrize("question", ["What was Bearman's best qualifying lap at Melbourne?", "Which driver only just made it into Q3 at Melbourne?",
                                      "Who recorded the highest speed trap reading at Monza?", "How far off pole was Oscar at Monza?",
                                      "How far did Hülkenberg get in qualifying at Melbourne?", "How far apart were Bortoleto and Lindblad?",
                                      "Where did Ocon qualify?", "What was Lawson's grid position at Melbourne?"])
def test_one_fact_questions_get_the_short_style(question):
    assert query.is_lookup_question(question)


@pytest.mark.parametrize("question", ["Why was Verstappen so bad in Monza?", "Compare Isack and Max during Aussie quali",
                                      "Who was faster through the first corner, Russell or Hamilton?", "Which driver carried more speed through the slowest corner?",
                                      "Who was quicker through T11, Stroll or Fernando?", "How did Stroll stack up against Sainz?",
                                      "Where did each team have an advantage in Melbourne?"])
def test_analysis_questions_keep_the_full_answer(question):
    assert not query.is_lookup_question(question)


def test_a_one_fact_question_asks_for_a_short_answer_and_a_small_token_budget(monkeypatch):
    seen = {}

    def fake(prompt, **kw):
        seen.update(kw, prompt=prompt)
        return "Hulkenberg."
    monkeypatch.setattr(query, "call_llm", fake)
    query.answer_with_sources("What was Bearman's best qualifying lap at Melbourne?", model="m")
    assert seen["max_tokens"] == query.CONCISE_MAX_TOKENS and "Answer style: this question asks for one fact" in seen["prompt"]
    seen.clear()
    query.answer_with_sources("Compare Russell and Hamilton at Barcelona.", model="m")
    assert "max_tokens" not in seen and "Answer style" not in seen["prompt"]


def test_a_one_fact_answer_is_cut_to_its_first_sentences():
    answer = ("BOR (Audi), with a Q2 time of 1:20.221. The context lists him as the slowest Q2 time among the 10 drivers who reached Q3. "
              "He set no Q3 time. The next-quickest was HUL at 1:20.303.")
    assert query.limit_sentences(answer, 2) == ("BOR (Audi), with a Q2 time of 1:20.221. The context lists him as the slowest Q2 time "
                                                "among the 10 drivers who reached Q3.")
    assert query.limit_sentences("Piastri was 0.109s off pole: 1:19.436 vs 1:19.327.", 3) == "Piastri was 0.109s off pole: 1:19.436 vs 1:19.327."
    assert query.lookup_sentence_limit("How far off pole was Oscar?") == 3 and query.lookup_sentence_limit("What was his best lap?") == 2


def test_the_cut_is_applied_to_one_fact_questions_only(monkeypatch):
    long = "First fact. Second fact. Third fact. Fourth fact."
    monkeypatch.setattr(query, "call_llm", lambda prompt, **k: long)
    assert query.answer_with_sources("What was Bearman's best qualifying lap at Melbourne?", model="m")["answer"] == "First fact. Second fact."
    assert query.answer_with_sources("Compare Russell and Hamilton at Barcelona.", model="m")["answer"] == long
