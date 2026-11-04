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
