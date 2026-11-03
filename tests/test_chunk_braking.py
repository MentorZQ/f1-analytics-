"""
Tests for the braking text in telemetry chunks, using synthetic laps.

Two bugs these guard against:
  * braking distance measured to the lowest speed ANYWHERE in the window, so in a
    multi-corner section it ran to a later, slower apex ("braking distance 500m")
  * a second "X braked Ym later" line worked out from braking distance, which
    contradicted the line worked out from the brake-on point

Run from the project root:
    pytest tests/test_chunk_braking.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from events_and_chunking.chunk_telemetry import _braking_analysis, chunk_head_to_head_event  # noqa: E402


def lap_window(brake_on: float, apex1: float = 1700.0, apex2: float = 2100.0, apex1_speed: float = 150.0,
               start: float = 1500.0, end: float = 2300.0) -> pd.DataFrame:
    """Two corners: brake at `brake_on`, slow to apex1, speed back up, then a slower apex2."""
    d = np.arange(start, end, 10.0)
    speed = np.interp(d, [start, brake_on, apex1, apex1 + 150, apex2 - 100, apex2, end],
                      [280, 280, apex1_speed, 250, 250, 100, 200])
    brake = ((d >= brake_on) & (d <= apex1 - 10)) | ((d >= apex2 - 90) & (d <= apex2 - 10))
    return pd.DataFrame({"Distance": d, "Speed": speed, "Brake": brake, "Throttle": np.where(brake, 0.0, 100.0),
                         "nGear": 5.0, "DRS": 0.0})


def test_braking_distance_stops_at_the_first_apex_not_the_slowest_one():
    tel = lap_window(brake_on=1590.0)
    out = _braking_analysis(tel, first_apex=1700.0)
    assert out["brake_start_dist_m"] == 1590.0
    assert out["brake_distance_m"] == pytest.approx(110.0, abs=10)     # to the T-first apex
    assert out["brake_distance_m"] < 200                                # not the ~500 m to the later, slower apex


def test_without_an_apex_it_still_stops_at_the_end_of_the_first_braking_phase():
    out = _braking_analysis(lap_window(brake_on=1590.0))
    assert out["brake_distance_m"] < 200


def test_filled_in_rows_cannot_set_the_brake_on_point():
    tel = lap_window(brake_on=1590.0)
    tel["Interpolated"] = False
    fake = tel.iloc[[3]].copy()                    # 1530 m: a row inserted across a data gap
    fake["Brake"], fake["Interpolated"] = True, True
    tel = pd.concat([tel, fake]).sort_values("Distance").reset_index(drop=True)
    assert _braking_analysis(tel, first_apex=1700.0)["brake_start_dist_m"] == 1590.0


def test_no_braking_is_reported_as_such():
    tel = lap_window(brake_on=1590.0)
    tel["Brake"] = False
    assert _braking_analysis(tel, first_apex=1700.0)["brake_applied"] is False


def _chunk_text(tel_a, tel_b):
    row = lambda code, pos: pd.Series({"Driver": code, "Team": "T", "QualifyingPosition": pos})
    event = {"id": "T4-T5", "display_id": "T4-T5", "type": "corner_sequence", "dist_start": 1500.0,
             "dist_end": 2300.0, "length_m": 800.0, "peak_dists": [1700.0, 2100.0]}
    chunk = chunk_head_to_head_event(row("AAA", 1), row("BBB", 2), tel_a, tel_b, event,
                                     {"circuit": "Test", "year": 2026}, 21.0, 21.1)
    return chunk.text


def test_the_two_braking_lines_can_no_longer_disagree():
    """A brakes 20 m LATER than B but over a LONGER distance: the old text then named B as the later braker too."""
    tel_a = lap_window(brake_on=1610.0, apex1=1740.0)           # starts later, and reaches its minimum later
    tel_b = lap_window(brake_on=1590.0, apex1=1690.0)
    text = _chunk_text(tel_a, tel_b)
    later_claims = [l for l in text.splitlines() if "later" in l]
    assert len(later_claims) == 1, later_claims
    assert "AAA brakes 20m later than BBB" in later_claims[0]
    assert "braked" not in text.replace("brakes", "")
    assert "Braking distance to first apex" in text             # reported as a fact instead


def test_braking_distance_in_the_text_is_to_the_first_apex():
    text = _chunk_text(lap_window(brake_on=1590.0), lap_window(brake_on=1585.0))
    for line in text.splitlines():
        if "braking distance to first apex" in line:
            assert int(line.rsplit("apex", 1)[1].strip().rstrip("m")) < 200


def test_equal_brake_points_say_so_instead_of_inventing_a_difference():
    text = _chunk_text(lap_window(brake_on=1590.0), lap_window(brake_on=1590.0))
    assert "start braking at about the same point" in text
    assert "later" not in text


# ── A hole in the car data right where braking starts ────────────────────────

def with_gap(tel: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Drop every sample strictly between lo and hi, as in the Barcelona RUS lap."""
    return tel[(tel.Distance <= lo) | (tel.Distance >= hi)].reset_index(drop=True)


def test_a_data_gap_at_the_onset_is_reported_as_a_range():
    tel = with_gap(lap_window(brake_on=1580.0), 1550.0, 1620.0)     # really braked at 1580; first sample at 1620
    out = _braking_analysis(tel, first_apex=1700.0)
    assert out["brake_start_dist_m"] == 1620.0
    assert out["brake_start_lo_m"] == 1550.0 and out["brake_gap_m"] == 70.0


def test_no_claim_about_who_brakes_later_when_the_ranges_overlap():
    """The Barcelona case: A's onset hides in a gap (1550-1620m); B brakes at 1593m. The old text said A was 23m later."""
    tel_a = with_gap(lap_window(brake_on=1580.0), 1550.0, 1620.0)
    tel_b = lap_window(brake_on=1590.0)
    text = _chunk_text(tel_a, tel_b)
    assert "AAA: first brakes between 1550m and 1620m (gap in the car data)" in text
    assert "cannot tell who brakes later" in text
    assert "brakes" not in "".join(l for l in text.splitlines() if "→" in l and "later than" in l)
    assert "from 198" not in text                    # no entry speed quoted for a guessed onset


def test_a_later_brake_is_still_claimed_when_it_is_clear_despite_the_gap():
    tel_a = with_gap(lap_window(brake_on=1580.0), 1550.0, 1620.0)   # onset somewhere in 1550-1620
    tel_b = lap_window(brake_on=1680.0)                              # clearly later
    text = _chunk_text(tel_a, tel_b)
    assert "BBB brakes at least " in text and "later than AAA" in text


def test_range_includes_the_braking_distance_uncertainty():
    tel = with_gap(lap_window(brake_on=1580.0), 1550.0, 1620.0)
    line = next(l for l in _chunk_text(tel, lap_window(brake_on=1590.0)).splitlines() if l.strip().startswith("AAA: first brakes"))
    assert "braking distance to first apex" in line and "–" in line


def test_normal_sampling_is_not_treated_as_uncertain():
    text = _chunk_text(lap_window(brake_on=1590.0), lap_window(brake_on=1620.0))
    assert "gap in the car data" not in text and "BBB brakes 30m later than AAA" in text
