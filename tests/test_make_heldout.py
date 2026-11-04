"""The data-generated held-out set: helpers, determinism, and the committed file."""

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import query  # noqa: E402
import test_session_snapshot as tss  # noqa: E402
from evals import make_heldout as mh  # noqa: E402
from events_and_chunking import session_snapshot as snap  # noqa: E402
from events_and_chunking.chunk_telemetry import build_telemetry_chunks  # noqa: E402
from events_and_chunking.detect_events_official import detect_official_events  # noqa: E402

FILE = ROOT / "evals" / "questions_heldout_data.json"


def matches(must: str, text: str) -> bool:
    """The same matching the evaluator uses."""
    text = text.lower().replace("*", "").replace(",", "")
    return any((re.search(a.strip()[3:], text, re.I) if a.strip().startswith("re:") else a.strip().lower() in text) for a in must.split("|"))


# ── helpers ──

def test_lap_times_are_offered_in_both_formats():
    assert mh.lap_alternatives(79.327) == "1:19.327|79.327"
    assert mh.lap_alternatives(61.0) == "1:01.000|61.000"


@pytest.mark.parametrize("n,expected", [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"), (13, "13th"),
                                        (21, "21st"), (22, "22nd")])
def test_ordinals(n, expected):
    assert mh.ordinal(n) == expected


def test_a_position_matches_its_own_forms_and_not_a_neighbouring_number():
    must = mh.position_alternatives(4)
    for ok in ("He qualified P4.", "He was 4th.", "He was fourth on the grid."):
        assert matches(must, ok), ok
    for bad in ("He qualified P40.", "He was 14th.", "He was fifth."):
        assert not matches(must, bad), bad


def test_driver_names_cover_surname_first_name_and_code():
    must = mh.name_alternatives("HUL")
    for ok in ("Hülkenberg was quickest.", "Hulkenberg was quickest.", "Nico was quickest.", "HUL was quickest."):
        assert matches(must, ok), ok
    assert not matches(must, "Hamilton was quickest.")


def test_display_uses_a_surname_or_a_first_name_never_an_empty_string():
    rng = random.Random(1)
    shown = {mh.display("SAI", rng) for _ in range(60)}
    assert shown <= {"Sainz", "Carlos"} and "Sainz" in shown


# ── determinism, on a synthetic session ──

@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    tel = {d: tss.telemetry(i, 1.0 - 0.01 * i) for i, (d, _, _) in enumerate(tss.DRIVERS)}
    for d, t in tel.items():
        t["Driver"] = d
    fastest = pd.DataFrame({"Driver": [d for d, _, _ in tss.DRIVERS], "Team": [t for _, t, _ in tss.DRIVERS],
                            "QualifyingPosition": [p for _, _, p in tss.DRIVERS]})
    events = detect_official_events(tel, [{"number": 1, "label": "1", "dist": 1000.0}, {"number": 2, "label": "2", "dist": 1850.0}], tss.LAP)
    path = tmp_path_factory.mktemp("mh") / "s.npz"
    build_telemetry_chunks({"driver_fastest_laps": fastest, "telemetry": tel}, events,
                           {"year": 2026, "circuit": "Testland", "session": "Qualifying"}, head_to_head_top_n=2, snapshot_path=str(path))
    stages = ["Q3", "Q3", "Q2", "Q2", "Q1"]
    drivers = {d: {"time": 80.0 + i * 0.137, "position": p, "team": t, "trap": 300.0 + i, "stage": stages[i]} for i, (d, t, p) in enumerate(tss.DRIVERS)}
    return {"drivers": drivers, "fastest_sector": {1: (26.1, "AAA")}, "snapshot": snap.load_snapshot(path)}


def make(synthetic, seed):
    counter = iter(range(1, 999))
    return mh.build("monza", synthetic, random.Random(seed), lambda: f"d{next(counter):02d}")


def test_the_same_seed_gives_the_same_questions_and_a_different_seed_does_not(synthetic):
    assert make(synthetic, 7) == make(synthetic, 7)
    assert make(synthetic, 7) != make(synthetic, 8)


def test_every_generated_question_has_facts_a_correct_answer_must_contain(synthetic):
    qs = make(synthetic, 7)
    assert qs and all(q["must_include"] and q["expected_drivers"] and q["answerable"] for q in qs)
    kinds = {q["kind"] for q in qs}
    assert {"data_lap_time", "data_position", "data_gap_to_pole", "data_speed_trap", "data_pair_gap", "data_teammate_gap", "data_stage"} <= kinds
    assert "data_q3_cutoff" not in kinds                                  # that family assumed P10 = last into Q3, which the data contradicts


def test_generated_gaps_are_computed_from_the_lap_times(synthetic):
    qs = make(synthetic, 7)
    pair = next(q for q in qs if q["kind"] == "data_pair_gap")
    a, b = pair["expected_drivers"]
    expected = abs(synthetic["drivers"][a]["time"] - synthetic["drivers"][b]["time"])
    assert pair["must_include"] == [f"{expected:.3f}"]


# ── the committed file ──

def test_the_committed_file_is_well_formed():
    qs = json.loads(FILE.read_text())
    assert len(qs) >= 50 and len({q["id"] for q in qs}) == len(qs)
    assert all(q["session"] is None and q["author"] == "generated" and q["must_include"] for q in qs)
    assert {q["expected_session"] for q in qs} == {"monza", "barcelona", "melbourne"}
    assert len({q["question"] for q in qs}) == len(qs)                    # no duplicate wording


def test_every_committed_question_is_routed_to_its_own_session_without_a_hint():
    for q in json.loads(FILE.read_text()):
        assert query.detect_session(q["question"]) == q["expected_session"], q["question"]


def test_the_committed_corner_questions_are_checkable_from_the_snapshots_alone():
    qs = [q for q in json.loads(FILE.read_text()) if q["kind"] == "data_corner_winner"]
    assert qs
    for q in qs:
        key = q["expected_session"]
        snapshot = snap.load_snapshot(ROOT / "RAG_data_layers" / "session_snapshots" / f"{query.SESSIONS[key]}.npz")
        a, b = q["expected_drivers"]
        section = re.search(r"through (\S+(?: \S+)*?) \(turn", q["note"]).group(1)
        winner, gap = mh.section_gap(snapshot, a, b, section)
        assert f"{gap:.3f}" in q["must_include"] and q["note"].endswith(f"{winner} quicker by {gap:.3f}s")


# ── the elimination-stage family ──

def test_stage_answers_are_recognised_and_the_wrong_stage_is_not():
    q3, q2 = mh.STAGE_ALTERNATIVES["Q3"], mh.STAGE_ALTERNATIVES["Q2"]
    assert matches(q3, "He qualified for Q3 and finished P8.") and matches(q3, "Bottas reached Q3.")
    assert not matches(q3, "He did not qualify for Q3; he was knocked out in Q2.")
    assert not matches(q3, "He was eliminated in Q1.")
    assert matches(q2, "He was eliminated at the end of Q2.") and matches(q2, "Knocked out in Q2.")
    assert not matches(q2, "He qualified for Q3.")


def test_the_stage_family_never_uses_a_p10_assumption():
    qs = json.loads(FILE.read_text())
    assert not any(q["kind"] == "data_q3_cutoff" for q in qs)
    stage = [q for q in qs if q["kind"] == "data_stage"]
    assert len(stage) == 6 and all("progression" in q["note"] for q in stage)
