"""
Offline tests for the official-corner section detector, using synthetic laps with
known braking and throttle points. No network or API key needed.

Run from the project root:
    pytest tests/test_detect_events_official.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from events_and_chunking import detect_events_official as de  # noqa: E402

LAP = 3000.0


def lap(brakes=(), throttle_off=(), seed=0, jitter=15.0, n_drivers=12):
    """
    Synthetic laps. `brakes` = [(brake_on, brake_off)] and `throttle_off` =
    [(lift, back_on)], each shifted per driver by up to `jitter` metres so
    drivers brake at different points.
    """
    rng = np.random.default_rng(seed)
    dist = np.arange(0, LAP, 5.0)
    out = {}
    for i in range(n_drivers):
        shift = rng.uniform(-jitter, jitter)
        brake = np.zeros(len(dist), bool)
        throttle = np.full(len(dist), 100.0)
        for on, off in brakes:
            brake |= (dist >= on + shift) & (dist <= off + shift)
        for lift, back in list(brakes) + list(throttle_off):
            throttle[(dist >= lift + shift) & (dist < back + shift)] = 20.0
        out[f"D{i:02d}"] = pd.DataFrame({"Distance": dist, "Brake": brake, "Throttle": throttle,
                                         "Speed": 250.0})
    return out


def kappa(peaks, width=60.0, height=0.01):
    d = np.arange(0, LAP, 5.0)
    k = np.zeros_like(d)
    for p in peaks:
        k = np.maximum(k, height * np.clip(1 - np.abs(d - p) / width, 0, 1))
    return d, k


def apexes(*dists):
    return [{"number": i + 1, "label": str(i + 1), "dist": float(d)} for i, d in enumerate(dists)]


def test_brake_zones_merge_small_gaps_only():
    d = np.arange(0, 400, 5.0)
    b = np.zeros(len(d), bool)
    b[(d >= 100) & (d <= 150)] = True
    b[(d >= 160) & (d <= 200)] = True       # 10 m gap: same zone
    b[(d >= 300) & (d <= 330)] = True       # far away: new zone
    # Each zone starts at the last sample before braking (the true onset is somewhere in between).
    assert de.brake_zones(d, b) == [(95.0, 200.0), (295.0, 330.0)]


def test_brake_onset_hidden_in_a_data_gap_is_still_enclosed():
    """A 65 m hole in the data right where the driver starts braking."""
    d = np.array([1500.0, 1520.0, 1551.0, 1616.0, 1630.0, 1650.0])
    b = np.array([False, False, False, True, True, True])
    (start, end), = de.brake_zones(d, b)
    assert start == 1551.0 and end == 1650.0     # the unseen onset lies between 1551 and 1616


def test_entry_sits_before_the_earliest_braking_driver():
    tel = lap(brakes=[(800, 960)])
    ev = de.detect_official_events(tel, apexes(1000), LAP)
    corner = next(e for e in ev if e["type"] == "corner")
    earliest = min(t.Distance[t.Brake].min() for t in tel.values())
    assert corner["dist_start"] < earliest
    # and the boundary sits outside every driver's braking: nobody is mid-brake there
    for t in tel.values():
        near = (t.Distance >= corner["dist_start"] - 1) & (t.Distance <= corner["dist_start"] + 1)
        assert not t.Brake[near].any()


def test_later_braker_cannot_leak_time_across_the_boundary():
    """The Barcelona failure: a boundary inside the braking zone."""
    tel = lap(brakes=[(800, 960)], jitter=40.0)
    corner = next(e for e in de.detect_official_events(tel, apexes(1000), LAP) if e["type"] == "corner")
    assert all(not t.Brake[t.Distance <= corner["dist_start"]].iloc[-3:].any() for t in tel.values())


def test_sections_tile_the_lap_without_gaps_or_overlaps():
    tel = lap(brakes=[(800, 960), (1700, 1800)])
    ev = de.detect_official_events(tel, apexes(1000, 1850), LAP)
    assert ev[0]["dist_start"] == 0.0
    assert ev[-1]["dist_end"] == LAP
    for a, b in zip(ev, ev[1:]):
        assert a["dist_end"] == pytest.approx(b["dist_start"])
    assert all(e["length_m"] > 0 for e in ev)


def test_names_follow_official_turn_numbers():
    tel = lap(brakes=[(800, 960), (1700, 1800)])
    ev = de.detect_official_events(tel, apexes(1000, 1850), LAP)
    assert [e["id"] for e in ev] == [
        "Main straight (start, before T1)", "T1", "Between T1-T2", "T2",
        "Main straight (finish, after final corner)",
    ]
    assert all(e["display_id"] == e["id"] for e in ev)       # so circuit_labels leaves them alone
    assert [e["type"] for e in ev] == ["straight", "corner", "straight", "corner", "straight"]


def test_corners_close_together_form_one_sequence():
    tel = lap(brakes=[(800, 900), (1000, 1060)])
    ev = de.detect_official_events(tel, apexes(940, 1100), LAP)
    seq = [e for e in ev if e["type"] == "corner_sequence"]
    assert len(seq) == 1 and seq[0]["id"] == "T1-T2"
    assert seq[0]["n_peaks"] == 2 and seq[0]["peak_dists"] == [940.0, 1100.0]
    assert seq[0]["corner_numbers"] == [1, 2]


def test_flat_corner_falls_back_to_curvature():
    tel = lap()                                        # nobody brakes or lifts
    ev = de.detect_official_events(tel, apexes(1500), LAP, curvature=kappa([1500]))
    corner = next(e for e in ev if e["type"] == "corner")
    assert not corner["braked_entry"]
    assert corner["dist_start"] < 1500 < corner["dist_end"]
    assert corner["length_m"] < 250                    # a bounded zone, not the whole lap


def test_flat_corner_without_curvature_still_gets_a_minimum_zone():
    ev = de.detect_official_events(lap(), apexes(1500), LAP)
    corner = next(e for e in ev if e["type"] == "corner")
    assert corner["dist_start"] <= 1500 - de.MIN_HALF_WIDTH_M
    assert corner["dist_end"] >= 1500 + de.MIN_HALF_WIDTH_M


def test_braking_for_the_previous_corner_is_not_claimed_by_the_next():
    tel = lap(brakes=[(800, 900), (1700, 1800)])
    ev = {e["id"]: e for e in de.detect_official_events(tel, apexes(950, 1850), LAP)}
    assert ev["T2"]["dist_start"] > ev["T1"]["dist_end"]


def test_output_matches_the_geometry_detector_schema():
    tel = lap(brakes=[(800, 960)])
    ev = de.detect_official_events(tel, apexes(1000), LAP, curvature=kappa([1000]))
    needed = {"id", "type", "dist_start", "dist_end", "length_m"}
    assert all(needed <= set(e) for e in ev)
    corner = next(e for e in ev if e["type"] == "corner")
    assert {"n_peaks", "peak_dists", "peak_kappas", "max_kappa"} <= set(corner)
    assert corner["max_kappa"] == pytest.approx(0.01, rel=0.1)
    assert "min_kappa" in next(e for e in ev if e["type"] == "straight")


def test_curvature_profile_finds_a_bend_and_starts_at_the_origin():
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.column_stack([300 * np.cos(th), 300 * np.sin(th)])
    d, k, length = de.curvature_profile(circle, origin=0.0)
    assert length == pytest.approx(2 * np.pi * 300, rel=0.01)
    assert np.median(k) == pytest.approx(1 / 300, rel=0.1)
    d2, _, _ = de.curvature_profile(circle, origin=100.0)
    assert d2[0] == pytest.approx(0.0, abs=5.0) and d2[-1] < length


def _with_brakers(zone_by_driver):
    """Telemetry where only some drivers brake: {driver index: (brake_on, brake_off)}."""
    tel = lap(n_drivers=12, jitter=0.0)
    for i, (on, off) in zone_by_driver.items():
        t = tel[f"D{i:02d}"]
        t["Brake"] = (t.Distance >= on) & (t.Distance <= off)
    return tel


def test_a_minority_who_brake_for_a_flat_corner_are_still_enclosed():
    tel = _with_brakers({0: (1380, 1440), 1: (1395, 1440)})      # 2 of 12 brake; the rest are flat
    corner = next(e for e in de.detect_official_events(tel, apexes(1500), LAP, curvature=kappa([1500]))
                  if e["type"] == "corner")
    assert corner["dist_start"] < 1380


def test_every_driver_is_enclosed_not_just_most():
    """Earliest braker sets the entry: no percentile that leaves the first few outside."""
    tel = lap(brakes=[(800, 960)], jitter=30.0, n_drivers=22, seed=3)
    corner = next(e for e in de.detect_official_events(tel, apexes(1000), LAP) if e["type"] == "corner")
    for t in tel.values():
        first_brake = t.Distance[t.Brake].min()
        assert corner["dist_start"] < first_brake


def test_one_odd_lap_far_from_the_pack_does_not_stretch_the_zone():
    tel = lap(brakes=[(800, 960)], jitter=5.0, n_drivers=12)
    odd = tel["D00"]
    odd["Brake"] = ((odd.Distance >= 500) & (odd.Distance <= 560)) | ((odd.Distance >= 800) & (odd.Distance <= 960))
    corner = next(e for e in de.detect_official_events(tel, apexes(1000), LAP) if e["type"] == "corner")
    assert corner["dist_start"] > 700          # not dragged back to the odd driver's early brake at ~500 m


def test_lift_before_the_brake_zone_belongs_to_the_corner_entry():
    tel = lap(brakes=[(800, 960)], jitter=0.0, n_drivers=12)
    for t in tel.values():                       # everyone coasts from 700 m, then brakes at 800 m
        t.loc[(t.Distance >= 700) & (t.Distance < 800), "Throttle"] = 0.0
    corner = next(e for e in de.detect_official_events(tel, apexes(1000), LAP) if e["type"] == "corner")
    assert corner["dist_start"] < 700


def test_a_lift_from_the_previous_corner_is_not_pulled_in():
    tel = lap(brakes=[(800, 900), (1700, 1800)], jitter=0.0, n_drivers=12)
    ev = {e["id"]: e for e in de.detect_official_events(tel, apexes(950, 1850), LAP)}
    assert ev["T2"]["dist_start"] > ev["T1"]["dist_end"]


CHAIN = [(800, 900), (930, 1000), (1030, 1100), (1130, 1200), (1230, 1300)]
CHAIN_APEXES = (910, 1010, 1110, 1210, 1310)


def test_a_long_chain_of_corners_is_cut_where_nobody_is_braking():
    tel = lap(brakes=CHAIN, jitter=0.0)
    ev = [e for e in de.detect_official_events(tel, apexes(*CHAIN_APEXES), LAP) if e["type"] != "straight"]
    assert len(ev) > 1
    assert all(e["length_m"] <= de.MAX_SEQUENCE_M for e in ev)
    for a, b in zip(ev, ev[1:]):
        cut = b["dist_start"]
        assert a["dist_end"] == pytest.approx(cut)                    # pieces touch: no invented straight
        for t in tel.values():                                        # and nobody is braking at the cut
            near = (t.Distance >= cut - 2) & (t.Distance <= cut + 2)
            assert not t.Brake[near].any()
    assert sorted(n for e in ev for n in e["corner_numbers"]) == [1, 2, 3, 4, 5]


def test_cut_sections_still_tile_the_lap_and_have_distinct_names():
    ev = de.detect_official_events(lap(brakes=CHAIN, jitter=0.0), apexes(*CHAIN_APEXES), LAP)
    assert ev[0]["dist_start"] == 0.0 and ev[-1]["dist_end"] == LAP
    for a, b in zip(ev, ev[1:]):
        assert a["dist_end"] == pytest.approx(b["dist_start"])
    assert all(e["length_m"] > 1 for e in ev)                         # no empty straights between cut pieces
    assert len({e["id"] for e in ev}) == len(ev)


def test_a_chain_with_no_braking_free_point_stays_whole():
    tel = lap(brakes=[(800, 1300)], jitter=0.0)                       # one continuous brake: nowhere to cut safely
    ev = [e for e in de.detect_official_events(tel, apexes(*CHAIN_APEXES), LAP) if e["type"] != "straight"]
    assert len(ev) == 1 and ev[0]["corner_numbers"] == [1, 2, 3, 4, 5]


def test_short_sequences_are_left_alone():
    tel = lap(brakes=[(800, 900), (930, 1000)], jitter=0.0)
    ev = [e for e in de.detect_official_events(tel, apexes(910, 1010), LAP) if e["type"] != "straight"]
    assert len(ev) == 1 and ev[0]["id"] == "T1-T2"
