"""The saved session snapshot must rebuild exactly the head-to-head chunks the store was built with."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from events_and_chunking import session_snapshot as snap  # noqa: E402
from events_and_chunking.chunk_telemetry import build_telemetry_chunks  # noqa: E402
from events_and_chunking.detect_events_official import detect_official_events  # noqa: E402

LAP = 3000.0
DRIVERS = [("AAA", "Alpha", 1), ("BBB", "Alpha", 2), ("CCC", "Beta", 3), ("DDD", "Beta", 4), ("EEE", "Gamma", 5)]


def telemetry(seed: int, pace: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dist = np.arange(0, LAP, 8.0)
    brake = np.zeros(len(dist), bool)
    speed = np.full(len(dist), 300.0 * pace)
    throttle = np.full(len(dist), 100.0)
    for on, off in ((800, 960), (1700, 1800)):
        shift = rng.uniform(-12, 12)
        m = (dist >= on + shift) & (dist <= off + shift)
        brake |= m
        throttle[m] = 0.0
        speed[(dist >= on + shift) & (dist <= off + 40 + shift)] = 120.0 * pace
    t = np.cumsum(8.0 / (speed / 3.6))
    return pd.DataFrame({
        "Time": pd.to_timedelta(t, unit="s"), "Distance": dist, "Speed": speed, "Throttle": throttle, "Brake": brake,
        "nGear": np.where(speed > 200, 8.0, 3.0), "DRS": np.zeros(len(dist)), "RPM": np.full(len(dist), 11000.0),
        "X": dist, "Y": np.zeros(len(dist)), "Driver": "x", "Interpolated": np.zeros(len(dist), bool)})


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tel = {d: telemetry(i, 1.0 - 0.01 * i) for i, (d, _, _) in enumerate(DRIVERS)}
    for d, t in tel.items():
        t["Driver"] = d
    fastest = pd.DataFrame({"Driver": [d for d, _, _ in DRIVERS], "Team": [t for _, t, _ in DRIVERS],
                            "QualifyingPosition": [p for _, _, p in DRIVERS],
                            "LapTime": pd.to_timedelta([80.0 + i * 0.1 for i in range(len(DRIVERS))], unit="s")})
    apexes = [{"number": 1, "label": "1", "dist": 1000.0}, {"number": 2, "label": "2", "dist": 1850.0}]
    events = detect_official_events(tel, apexes, LAP)
    info = {"year": 2026, "circuit": "Testland", "session": "Qualifying"}
    path = tmp_path_factory.mktemp("snap") / "s.npz"
    chunks = build_telemetry_chunks({"driver_fastest_laps": fastest, "telemetry": tel}, events, info,
                                    head_to_head_top_n=3, snapshot_path=str(path))
    return chunks, snap.load_snapshot(path, align=False), fastest, tel


def test_the_snapshot_round_trips_the_data_exactly(built):
    _, loaded, fastest, tel = built
    pd.testing.assert_frame_equal(loaded.fastest.reset_index(drop=True), fastest.reset_index(drop=True), check_dtype=True)
    for d, t in tel.items():
        pd.testing.assert_frame_equal(loaded.telemetry[d].reset_index(drop=True), t.reset_index(drop=True), check_dtype=True)


def test_every_stored_head_to_head_chunk_is_rebuilt_with_identical_text(built):
    chunks, loaded, _, _ = built
    stored = {c.chunk_id: c for c in chunks if c.chunk_type == "head_to_head_event"}
    assert stored, "the synthetic session produced no head-to-head chunks"
    rebuilt = {}
    pairs = {(c.metadata["driver_a"], c.metadata["driver_b"]) for c in stored.values()}
    for a, b in pairs:
        for c in snap.compare_drivers(loaded, a, b):
            rebuilt[c.chunk_id] = c
    assert set(rebuilt) >= set(stored)
    for cid, c in stored.items():
        assert rebuilt[cid].text == c.text, cid
        assert rebuilt[cid].metadata == c.metadata, cid


def test_any_pair_can_be_compared_even_if_no_chunk_was_stored_for_it(built):
    chunks, loaded, _, _ = built
    stored_pairs = {frozenset((c.metadata["driver_a"], c.metadata["driver_b"])) for c in chunks if c.chunk_type == "head_to_head_event"}
    never_stored = [frozenset(p) for p in (("AAA", "EEE"), ("BBB", "EEE"), ("AAA", "DDD"))
                    if frozenset(p) not in stored_pairs]
    assert never_stored, "expected at least one pair outside the stored set with top_n=3"
    pair = sorted(never_stored[0])
    out = snap.compare_drivers(loaded, pair[0], pair[1])
    assert out and all(c.chunk_type == "head_to_head_event" for c in out)
    assert {out[0].metadata["driver_a"], out[0].metadata["driver_b"]} == set(pair)


def test_pair_order_follows_the_driver_table_whichever_way_it_is_asked(built):
    _, loaded, _, _ = built
    one = snap.compare_drivers(loaded, "AAA", "CCC")
    two = snap.compare_drivers(loaded, "CCC", "AAA")
    assert [c.text for c in one] == [c.text for c in two]
    assert one[0].metadata["driver_a"] == "AAA"


def test_unknown_drivers_give_nothing(built):
    _, loaded, _, _ = built
    assert snap.compare_drivers(loaded, "AAA", "ZZZ") == []


def test_a_snapshot_with_the_wrong_version_is_refused(tmp_path):
    import json
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, meta=np.array(json.dumps({"version": 99, "layout": {}, "drivers": [], "events": [], "session_info": {}})))
    with pytest.raises(ValueError):
        snap.load_snapshot(path)


# ── laps from the other qualifying sessions ──

def test_focus_session_is_the_last_shared_session_when_best_laps_come_from_different_sessions():
    assert snap.focus_session({1: 81.0, 2: 80.5}, {1: 82.0}) == 1                       # one driver out in Q1
    assert snap.focus_session({1: 80.4, 2: 79.97, 3: 81.2}, {1: 80.5, 2: 80.1, 3: 79.99}) == 3     # both in Q3, one slower there
    assert snap.focus_session({1: 81.0, 2: 80.5, 3: 80.0}, {1: 81.2, 2: 80.9, 3: 80.6}) is None   # same best session
    assert snap.focus_session({}, {1: 80.0}) is None


def test_other_session_laps_round_trip_and_the_pair_is_compared_on_those_laps(tmp_path):
    tel = {d: telemetry(i, 1.0 - 0.01 * i) for i, (d, _, _) in enumerate(DRIVERS)}
    for d, t in tel.items():
        t["Driver"] = d
    q1_lap = telemetry(9, 0.93)                                   # BBB's slower Q1 lap
    q1_lap["Driver"] = "BBB"
    fastest = pd.DataFrame({"Driver": [d for d, _, _ in DRIVERS], "Team": [t for _, t, _ in DRIVERS],
                            "QualifyingPosition": [p for _, _, p in DRIVERS],
                            "LapTime": pd.to_timedelta([80.0 + i * 0.1 for i in range(len(DRIVERS))], unit="s"),
                            "Q1Time": [81.0] * 5, "Q2Time": [80.5, 80.4, np.nan, np.nan, np.nan], "Q3Time": [80.0, np.nan, np.nan, np.nan, np.nan]})
    events = detect_official_events(tel, [{"number": 1, "label": "1", "dist": 1000.0}, {"number": 2, "label": "2", "dist": 1850.0}], LAP)
    path = tmp_path / "s.npz"
    build_telemetry_chunks({"driver_fastest_laps": fastest, "telemetry": tel, "session_telemetry": {"BBB@Q1": q1_lap}}, events,
                           {"year": 2026, "circuit": "Testland", "session": "Qualifying"}, head_to_head_top_n=3, snapshot_path=str(path))
    loaded = snap.load_snapshot(path, align=False)
    pd.testing.assert_frame_equal(loaded.session_telemetry["BBB@Q1"].reset_index(drop=True), q1_lap.reset_index(drop=True), check_dtype=True)
    best = snap.compare_drivers(loaded, "AAA", "BBB")
    like = snap.compare_drivers(loaded, "AAA", "BBB", session=1)
    # AAA's best is Q3 (not stored as a Q1 lap): no Q1 lap for him, so the like-for-like comparison is unavailable
    assert best and like == []
    assert snap.best_session(snap.driver_row(loaded, "AAA")) == 3 and snap.best_session(snap.driver_row(loaded, "BBB")) == 2
    # a driver whose best lap IS the session asked for uses his main telemetry
    assert snap.compare_drivers(loaded, "AAA", "BBB", session=3) == []                  # BBB has no Q3 lap
    only_best_q1 = snap.lap_telemetry(loaded, "BBB", 1)
    assert only_best_q1 is not None and only_best_q1.equals(loaded.session_telemetry["BBB@Q1"])


def test_a_like_for_like_comparison_uses_and_labels_the_other_session_laps(tmp_path):
    tel = {d: telemetry(i, 1.0 - 0.01 * i) for i, (d, _, _) in enumerate(DRIVERS)}
    for d, t in tel.items():
        t["Driver"] = d
    q1 = {f"{d}@Q1": telemetry(20 + i, 0.9 - 0.02 * i) for i, d in enumerate(("AAA", "BBB"))}
    fastest = pd.DataFrame({"Driver": [d for d, _, _ in DRIVERS], "Team": [t for _, t, _ in DRIVERS],
                            "QualifyingPosition": [p for _, _, p in DRIVERS],
                            "LapTime": pd.to_timedelta([80.0 + i * 0.1 for i in range(len(DRIVERS))], unit="s"),
                            "Q1Time": [81.0] * 5, "Q2Time": [80.5, 80.4, np.nan, np.nan, np.nan], "Q3Time": [80.0, np.nan, np.nan, np.nan, np.nan]})
    events = detect_official_events(tel, [{"number": 1, "label": "1", "dist": 1000.0}, {"number": 2, "label": "2", "dist": 1850.0}], LAP)
    path = tmp_path / "s.npz"
    build_telemetry_chunks({"driver_fastest_laps": fastest, "telemetry": tel, "session_telemetry": q1}, events,
                           {"year": 2026, "circuit": "Testland", "session": "Qualifying"}, head_to_head_top_n=3, snapshot_path=str(path))
    loaded = snap.load_snapshot(path)
    best, like = snap.compare_drivers(loaded, "AAA", "BBB"), snap.compare_drivers(loaded, "AAA", "BBB", session=1)
    assert like and len(like) == len(best)
    assert all("AAA and BBB Q1 laps" in c.text and c.metadata["lap_session"] == "Q1" for c in like)
    assert [c.text for c in like] != [c.text for c in best]


# ── distance alignment ──

def test_the_alignment_correction_is_zero_at_both_ends_and_largest_mid_lap():
    assert abs(snap._bulge(0.0, 3000.0, 12.0)) < 1e-12 and abs(snap._bulge(3000.0, 3000.0, 12.0)) < 1e-12
    assert snap._bulge(1500.0, 3000.0, 12.0) == pytest.approx(12.0)


def test_a_lap_whose_distance_axis_is_offset_is_pulled_back_into_line(built):
    import copy
    _, loaded, _, _ = built
    shifted = copy.deepcopy(loaded)
    t = shifted.telemetry["AAA"].copy()
    length = float(t["Distance"].iloc[-1])
    t["Distance"] = t["Distance"] + snap._bulge(t["Distance"].to_numpy(), length, 14.0)         # an axis that runs 14 m late mid-lap
    shifted.telemetry["AAA"] = t
    aligned = snap.align_distance(shifted)
    assert aligned.alignment_m["AAA"] > 5.0                                                       # the late axis is found (a weakly varying synthetic speed trace)
    assert aligned.telemetry["AAA"]["Distance"].iloc[0] == pytest.approx(0.0) and aligned.telemetry["AAA"]["Distance"].iloc[-1] == pytest.approx(length)
    assert "DistanceRaw" in aligned.telemetry["AAA"].columns


def test_alignment_leaves_the_exact_lap_time_and_the_section_total_unchanged(built):
    import copy
    _, loaded, _, _ = built
    aligned = snap.align_distance(copy.deepcopy(loaded))
    a, b = "AAA", "CCC"
    raw = snap.compare_drivers(loaded, a, b)
    al = snap.compare_drivers(aligned, a, b)
    total = lambda cs: sum(c.metadata["time_delta_s"] for c in cs if c.metadata.get("time_delta_s") is not None)   # noqa: E731
    assert raw and al and total(raw) == pytest.approx(total(al), abs=0.0015)
