"""
Offline tests for the OpenF1 loader's geometry maths. A synthetic oval stands in
for a real circuit, so no network access or API key is needed.

Run from the project root:
    pytest tests/test_openf1_loader.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "RAG_data_layers/F1 Project Thoughts"))

import load_qualifying_openf1 as lo  # noqa: E402

N_POINTS = 800


def oval() -> pd.DataFrame:
    """An asymmetric closed centerline, roughly 3.4 km (a plain oval is symmetric under 180 degrees)."""
    th = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
    x = 700 * np.cos(th) + 120 * np.cos(2 * th)
    y = 330 * np.sin(th) + 60 * np.sin(3 * th)
    return pd.DataFrame({"x_m": x, "y_m": y, "w_tr_right_m": 6.0})


def perimeter(g: pd.DataFrame) -> float:
    xy = g[["x_m", "y_m"]].to_numpy()
    return float(np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T).sum())


def drive(geom: pd.DataFrame, start_m: float, speed_ms: float = 60.0, hz: float = 3.7,
          jitter_m: float = 0.0, seed: int = 0):
    """Samples of a car circling the oval once, starting `start_m` along the centerline."""
    rng = np.random.default_rng(seed)
    center = lo._Centerline(geom[["x_m", "y_m"]].to_numpy())
    lap_s = center.length / speed_ms
    t = np.arange(0.05, lap_s, 1 / hz)
    arc = (start_m + speed_ms * t) % center.length
    xs = np.interp(arc, center.s, center.pts[:, 0])
    ys = np.interp(arc, center.s, center.pts[:, 1])
    xy = np.column_stack([xs, ys]) + rng.normal(0, jitter_m, (len(t), 2))
    tel = pd.DataFrame({
        "t": t, "Speed": speed_ms * 3.6, "Xraw": xy[:, 0] / lo.POSITION_UNIT_M,
        "Yraw": xy[:, 1] / lo.POSITION_UNIT_M, "RPM": 11000.0, "nGear": 7.0,
        "Throttle": 100.0, "Brake": False, "DRS": 0.0,
    })
    return tel, lap_s, center


def test_deleted_laps_are_parsed_from_race_control():
    rc = [
        {"message": "CAR 55 (SAI) TIME 1:18.173 DELETED - TRACK LIMITS AT TURN 2 LAP 6 16:08:58"},
        {"message": "CAR 18 (STR) LAP DELETED - TRACK LIMITS AT TURN 10 LAP 6 (PIT)"},   # no time: ignored
        {"message": "YELLOW IN TRACK SECTOR 3"},
    ]
    deleted = lo.deleted_lap_times(rc)
    assert deleted == {55: [pytest.approx(78.173)]}
    assert lo._is_deleted(55, 78.173, deleted)
    assert not lo._is_deleted(55, 78.2, deleted)
    assert not lo._is_deleted(44, 78.173, deleted)


def test_roll_geometry_moves_origin_without_changing_the_shape():
    g = oval()
    rolled = lo.roll_geometry(g, 30.0)
    assert len(rolled) == len(g)
    assert perimeter(rolled) == pytest.approx(perimeter(g), rel=1e-9)
    # Same points, just a new start: nothing is interpolated, so curvature is untouched.
    assert set(map(tuple, rolled[["x_m", "y_m"]].round(6).to_numpy())) == set(map(tuple, g[["x_m", "y_m"]].round(6).to_numpy()))
    arc_to_new_origin = lo._Centerline(g[["x_m", "y_m"]].to_numpy())
    k = int(np.argmin(np.abs(arc_to_new_origin.s[:-1] - 30.0)))
    assert rolled.iloc[0]["x_m"] == g.iloc[k]["x_m"]
    assert (rolled.columns == g.columns).all()


def test_roll_geometry_offset_wraps_and_handles_negative():
    g = oval()
    L = perimeter(g)
    a = lo.roll_geometry(g, -25.0)
    b = lo.roll_geometry(g, L - 25.0)
    assert a.iloc[0]["x_m"] == b.iloc[0]["x_m"]


def test_registration_recovers_unit_scale_and_small_error():
    g = oval()
    laps = [drive(g, 0, jitter_m=1.0, seed=s)[0][["Xraw", "Yraw"]].to_numpy() for s in range(4)]
    tf = lo.register_to_geometry(laps, g[["x_m", "y_m"]].to_numpy())
    assert tf["scale"] == pytest.approx(1.0, abs=0.01)         # OpenF1 units are decimetres
    assert abs(tf["rotation_deg"]) < 1.0
    assert tf["median_err_m"] < 3.0


def test_registration_handles_a_shifted_and_rotated_frame():
    g = oval()
    tel = drive(g, 0, jitter_m=0.5)[0]
    xy = tel[["Xraw", "Yraw"]].to_numpy() * lo.POSITION_UNIT_M
    th = np.radians(37)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    moved = (xy @ R.T + [5000, -3000]) / lo.POSITION_UNIT_M
    tf = lo.register_to_geometry([moved], g[["x_m", "y_m"]].to_numpy())
    assert tf["median_err_m"] < 3.0


def test_unwrap_is_monotonic_and_handles_a_start_before_the_line():
    L = 1000.0
    arc = np.array([990.0, 995.0, 5.0, 12.0, 20.0])            # crosses the origin
    out = lo._unwrap_lap(arc, L)
    assert out[0] == pytest.approx(-10.0)
    assert (np.diff(out) > 0).all()
    assert out[-1] == pytest.approx(20.0)


def test_fused_distance_is_smooth_and_matches_true_distance_despite_jitter():
    g = oval()
    tel, lap_s, center = drive(g, 0.0, speed_ms=60.0, jitter_m=3.0, seed=1)
    tf = {"scale": 1.0, "R": np.eye(2), "t": np.zeros(2)}
    dist, d0, d1, _ = lo._fused_distance(tel, lap_s, center, tf)
    truth = 60.0 * tel["t"].to_numpy()
    assert np.abs(dist - truth).max() < 3.0                    # raw position jitter is also ~3 m
    step = np.diff(dist)
    assert step.min() > 0
    assert step.std() < 1.0                                    # raw projection would be several metres
    assert d0 == pytest.approx(0.0, abs=3.0)
    assert d1 == pytest.approx(center.length, abs=6.0)


def test_finished_telemetry_spans_exactly_one_lap_for_every_driver():
    g = oval()
    tf = {"scale": 1.0, "R": np.eye(2), "t": np.zeros(2)}
    for speed, seed in [(55.0, 0), (62.0, 1), (70.0, 2)]:
        tel, lap_s, center = drive(g, 0.0, speed_ms=speed, jitter_m=2.0, seed=seed)
        out = lo._finish_telemetry(tel, lap_s, "XXX", center, tf)
        assert out["Distance"].iloc[0] == 0.0
        assert out["Distance"].iloc[-1] == pytest.approx(center.length)
        assert out["Distance"].is_monotonic_increasing
        assert out["Time"].iloc[-1].total_seconds() == pytest.approx(lap_s)
        assert {"Time", "Distance", "Speed", "Throttle", "Brake", "nGear", "DRS", "X", "Y", "Driver"} <= set(out.columns)


def test_drivers_on_different_lines_share_one_distance_axis():
    """Same section boundary, two drivers whose own speed integrals would disagree."""
    g = oval()
    tf = {"scale": 1.0, "R": np.eye(2), "t": np.zeros(2)}
    a, lap_a, center = drive(g, 0.0, speed_ms=60.0, jitter_m=1.0, seed=3)
    b, lap_b, _ = drive(g, 0.0, speed_ms=60.0, jitter_m=1.0, seed=4)
    b["Speed"] *= 1.01                                          # speed channel reads 1% high
    ta = lo._finish_telemetry(a, lap_a, "A", center, tf)
    tb = lo._finish_telemetry(b, lap_b, "B", center, tf)
    # Position pins both to the track: they cross 1500 m at (almost) the same time.
    t_a = np.interp(1500, ta["Distance"], ta["Time"].dt.total_seconds())
    t_b = np.interp(1500, tb["Distance"], tb["Time"].dt.total_seconds())
    assert abs(t_a - t_b) < 0.08
    # Integrating the biased speed alone drifts about 1% (~20-30 m) by the end of the lap.
    fallback = lo._finish_telemetry(b, lap_b, "B", None, None)
    assert abs(fallback["Distance"].iloc[-1] - center.length) > 15


def test_gap_filling_follows_braking_instead_of_assuming_constant_speed():
    # 1.04 s with no samples while braking from 276 to 198 km/h, as seen in the Barcelona data
    tel = pd.DataFrame({
        "t": [20.45, 20.73, 21.17, 22.21, 22.41],
        "Distance": [1496.1, 1517.9, 1551.1, 1616.4, 1626.8],
        "Speed": [288.0, 284.0, 276.0, 198.0, 187.0],
        "Brake": [False, False, False, True, True], "Throttle": [100.0, 100.0, 100.0, 0.0, 0.0],
        "X": 0.0, "Y": 0.0,
    })
    out = lo.densify_gaps(tel)
    gap = out[(out.t > 21.17) & (out.t < 22.21)]
    assert gap.Interpolated.all() and len(gap) >= 8
    assert out.Distance.is_monotonic_increasing and out.t.is_monotonic_increasing
    # nothing outside the gap is touched, and distance still lands on the next real sample
    assert (~out[~out.Interpolated].Interpolated).all() and len(out[~out.Interpolated]) == len(tel)
    assert gap.Speed.is_monotonic_decreasing and 198 < gap.Speed.min() and gap.Speed.max() < 276
    # constant acceleration front-loads nothing: distance covered in the first half is
    # more than half of the total, because the car is faster early in the gap
    mid = np.interp(21.69, out.t, out.Distance)
    assert mid - 1551.1 > (1616.4 - 1551.1) / 2
    # a straight-line interpolation would put the midpoint exactly halfway
    assert abs(mid - (1551.1 + 1616.4) / 2) > 1.0


def test_dense_data_is_left_alone():
    t = np.arange(0, 5, 0.25)
    tel = pd.DataFrame({"t": t, "Distance": t * 60, "Speed": 216.0, "Brake": False, "Throttle": 100.0, "X": 0.0, "Y": 0.0})
    out = lo.densify_gaps(tel)
    assert len(out) == len(tel) and not out.Interpolated.any()
