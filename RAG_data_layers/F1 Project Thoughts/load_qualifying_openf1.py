"""
load_qualifying_openf1.py
-------------------------
Drop-in replacement for load_qualifying.load_qualifying_session that uses only
the free OpenF1 API (no F1 account, no FastF1 cache).

Returns the same dict shape, so the chunkers do not change:
    session_info, driver_fastest_laps, all_laps, weather, telemetry, dnq_drivers

Key difference from the FastF1 loader: lap distance.
FastF1 integrates each driver's own speed trace, so every driver ends up on a
slightly different distance axis. Here, distance is the car's X/Y position
projected onto the track centerline (TUMFTM geometry), so all drivers share one
axis and section boundaries mean the same place for everyone. If no geometry is
given, it falls back to integrating speed.

Usage:
    from load_qualifying_openf1 import load_qualifying_session_openf1
    data = load_qualifying_session_openf1(2026, "Barcelona", geometry=track["geometry"])
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.openf1.org/v1"
POSITION_UNIT_M = 0.1          # OpenF1 X/Y are in decimetres
MIN_SPACING_S = 0.4            # stay well inside OpenF1's rate limit
_last_call = 0.0


# ── HTTP ─────────────────────────────────────────────────────────────────────

def _get(endpoint: str, **params) -> list:
    """GET with throttling and retry. Set OPENF1_CACHE_DIR to cache raw responses."""
    global _last_call
    cache_dir = os.environ.get("OPENF1_CACHE_DIR")
    cache_file = None
    if cache_dir:
        key = re.sub(r"[^A-Za-z0-9_.=-]+", "_", f"{endpoint}-{json.dumps(params, sort_keys=True)}")[:180]
        cache_file = Path(cache_dir) / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

    status = None
    for attempt in range(8):
        wait = MIN_SPACING_S - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)
        if r.status_code == 404:           # OpenF1 uses 404 for "no rows"
            data = []
            break
        status = r.status_code
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(4 * (attempt + 1), 30))
            continue
        r.raise_for_status()
        data = r.json()
        break
    else:
        raise RuntimeError(f"OpenF1 {endpoint} failed after retries (last HTTP status {status})")

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))
    return data


# ── Session lookup ───────────────────────────────────────────────────────────

def find_session(year: int, circuit: str) -> dict:
    sessions = _get("sessions", year=year, session_name="Qualifying")
    needle = circuit.lower()
    hits = [
        s for s in sessions
        if any(needle in str(s.get(k, "")).lower()
               for k in ("circuit_short_name", "location", "country_name"))
    ]
    if not hits:
        names = sorted({s.get("circuit_short_name", "?") for s in sessions})
        raise ValueError(f"No {year} qualifying session matches '{circuit}'. OpenF1 has: {names}")
    return hits[0]


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


# ── Deleted laps ─────────────────────────────────────────────────────────────

_DELETED = re.compile(r"CAR (\d+) \((\w+)\) TIME (\d+):(\d+\.\d+) DELETED", re.I)


def deleted_lap_times(race_control: list) -> dict[int, list[float]]:
    """Map driver number -> lap times (s) that stewards deleted (track limits)."""
    out: dict[int, list[float]] = {}
    for r in race_control:
        m = _DELETED.search(r.get("message") or "")
        if m:
            out.setdefault(int(m.group(1)), []).append(int(m.group(3)) * 60 + float(m.group(4)))
    return out


def _is_deleted(num: int, lap_s: float, deleted: dict[int, list[float]]) -> bool:
    return any(abs(lap_s - d) < 0.002 for d in deleted.get(num, []))


# ── Geometry registration ────────────────────────────────────────────────────

def _resample(xy: np.ndarray, step: float, closed: bool = False) -> np.ndarray:
    pts = np.vstack([xy, xy[:1]]) if closed else xy
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    grid = np.arange(0, s[-1], step)
    return np.column_stack([np.interp(grid, s, pts[:, 0]), np.interp(grid, s, pts[:, 1])])


def _nearest(points: np.ndarray, ref: np.ndarray, chunk: int = 2000):
    idx = np.empty(len(points), int)
    dist = np.empty(len(points))
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk]
        d2 = ((p[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        idx[i:i + chunk] = j
        dist[i:i + chunk] = np.sqrt(d2[np.arange(len(p)), j])
    return idx, dist


def _similarity(src: np.ndarray, dst: np.ndarray):
    """Least-squares scale + rotation + translation mapping src onto dst (Umeyama)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    sign = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ np.diag([1, sign]) @ Vt
    scale = (sig * [1, sign]).sum() / (S ** 2).sum() * len(src)
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def register_to_geometry(laps_xy: list[np.ndarray], geom_xy: np.ndarray) -> dict:
    """
    Find the transform taking OpenF1 X/Y onto the track geometry.

    Pools every driver's fastest-lap path (different racing lines average out
    toward the centerline), resamples evenly by arc length so sample density
    does not bias the fit, then refines with ICP.
    """
    pooled = np.vstack([_resample(p * POSITION_UNIT_M, 2.0) for p in laps_xy])
    ref = _resample(geom_xy, 2.0, closed=True)

    # Coarse start: match centroids, search rotation over a full circle.
    best = None
    for th in np.radians(np.arange(0, 360, 2.0)):
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        moved = (pooled - pooled.mean(0)) @ R.T + ref.mean(0)
        sub = moved[::7]
        _, d = _nearest(sub, ref)
        score = float(np.mean(d ** 2))
        if best is None or score < best[0]:
            best = (score, R)
    R0 = best[1]
    scale, R, t = 1.0, R0, ref.mean(0) - R0 @ pooled.mean(0)

    for _ in range(40):
        moved = scale * pooled @ R.T + t
        idx, d = _nearest(moved, ref)
        keep = d < np.percentile(d, 90)         # ignore pit-lane / off-line outliers
        new_scale, new_R, new_t = _similarity(pooled[keep], ref[idx[keep]])
        if abs(new_scale - scale) < 1e-7 and np.allclose(new_R, R, atol=1e-8):
            scale, R, t = new_scale, new_R, new_t
            break
        scale, R, t = new_scale, new_R, new_t

    moved = scale * pooled @ R.T + t
    _, d = _nearest(moved, ref)
    return {
        "scale": scale, "R": R, "t": t,
        "rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
        "median_err_m": float(np.median(d)),
        "p95_err_m": float(np.percentile(d, 95)),
    }


def _apply(tf: dict, xy_decimetres: np.ndarray) -> np.ndarray:
    return tf["scale"] * (xy_decimetres * POSITION_UNIT_M) @ tf["R"].T + tf["t"]


class _Centerline:
    """Closed centerline with arc length; projects points to distance along the lap."""

    def __init__(self, geom_xy: np.ndarray, origin: float = 0.0):
        self.origin = origin          # metres along the outline where the timing line sits
        self.pts = np.vstack([geom_xy, geom_xy[:1]])
        seg = np.hypot(*np.diff(self.pts, axis=0).T)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        self.length = float(self.s[-1])
        self.n = len(self.pts) - 1

    def arc_of(self, p: np.ndarray) -> float:
        """Distance along the lap of the point on the outline nearest `p` (global search)."""
        a, b = self.pts[:-1], self.pts[1:]
        v = b - a
        vv = (v ** 2).sum(1)
        u = np.clip(((p - a) * v).sum(1) / np.where(vv == 0, 1, vv), 0, 1)
        d2 = (((a + u[:, None] * v) - p) ** 2).sum(1)
        k = int(d2.argmin())
        return float((self.s[k] + u[k] * np.sqrt(vv[k]) - self.origin) % self.length)

    def project(self, xy: np.ndarray, window_m: float = 120.0) -> np.ndarray:
        """Arc length for each point, tracked forward so it never jumps across the circuit."""
        out = np.empty(len(xy))
        i = int(np.argmin(((self.pts[:-1] - xy[0]) ** 2).sum(1)))
        step = self.length / self.n
        span = max(int(window_m / step), 5)
        for k, p in enumerate(xy):
            cand = (np.arange(i - span // 4, i + span) % self.n)
            d2 = ((self.pts[cand] - p) ** 2).sum(1)
            j = int(cand[d2.argmin()])
            # refine on the two adjacent segments
            best_s, best_d = self.s[j], d2.min()
            for a in (j - 1, j):
                a %= self.n
                b = a + 1
                v = self.pts[b] - self.pts[a]
                vv = float(v @ v)
                if vv == 0:
                    continue
                u = float(np.clip(((p - self.pts[a]) @ v) / vv, 0, 1))
                dd = float(((self.pts[a] + u * v - p) ** 2).sum())
                if dd < best_d:
                    best_d, best_s = dd, self.s[a] + u * np.sqrt(vv)
            out[k] = (best_s - self.origin) % self.length
            i = j
        return out


def _unwrap_lap(arc: np.ndarray, length: float) -> np.ndarray:
    """Turn wrapped arc lengths into a monotonic distance starting near 0."""
    arc = arc.copy()
    if arc[0] > length / 2:
        arc[0] -= length                       # started just before the line
    for k in range(1, len(arc)):
        while arc[k] - arc[k - 1] < -length / 2:
            arc[k] += length
        while arc[k] - arc[k - 1] > length / 2:
            arc[k] -= length
    return np.maximum.accumulate(arc)


# ── Telemetry for one lap ────────────────────────────────────────────────────

def _fetch_lap_telemetry(session_key: int, number: int, lap: dict) -> Optional[pd.DataFrame]:
    t0 = _ts(lap["date_start"])
    t1 = t0 + pd.Timedelta(seconds=lap["lap_duration"])
    window = {"date>": t0.isoformat(), "date<": t1.isoformat()}
    car = _get("car_data", session_key=session_key, driver_number=number, **window)
    loc = _get("location", session_key=session_key, driver_number=number, **window)
    if len(car) < 20 or len(loc) < 20:
        return None

    car_df = pd.DataFrame(car)
    car_df["t"] = (pd.to_datetime(car_df["date"], utc=True, format="mixed") - t0).dt.total_seconds()
    loc_df = pd.DataFrame(loc)
    loc_df["t"] = (pd.to_datetime(loc_df["date"], utc=True, format="mixed") - t0).dt.total_seconds()
    car_df = car_df.sort_values("t").reset_index(drop=True)
    loc_df = loc_df.sort_values("t").reset_index(drop=True)

    # Car data is the master timeline; position is interpolated onto it.
    tel = pd.DataFrame({
        "t": car_df["t"],
        "Speed": car_df["speed"].astype(float),
        "RPM": car_df["rpm"].astype(float),
        "nGear": car_df["n_gear"].astype(float),
        "Throttle": car_df["throttle"].astype(float),
        "Brake": pd.to_numeric(car_df["brake"], errors="coerce").fillna(0) > 0,
        "DRS": pd.to_numeric(car_df["drs"], errors="coerce").fillna(0).astype(float),
        "Xraw": np.interp(car_df["t"], loc_df["t"], loc_df["x"].astype(float)),
        "Yraw": np.interp(car_df["t"], loc_df["t"], loc_df["y"].astype(float)),
    })
    tel = tel[(tel["t"] >= 0) & (tel["t"] <= lap["lap_duration"])].reset_index(drop=True)
    return tel


def roll_geometry(geometry: pd.DataFrame, offset_m: float) -> pd.DataFrame:
    """
    Re-origin a closed centerline so index 0 is the point nearest `offset_m` along it.

    Rolls whole points rather than resampling: interpolating new points onto the
    old polyline puts kinks at the old vertices, which corrupts curvature and
    therefore corner detection. The origin lands within half a point spacing
    (~2.5 m) of the requested offset.
    """
    xy = geometry[["x_m", "y_m"]].to_numpy(float)
    seg = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])[:-1]
    k = int(np.argmin(np.abs(s - (offset_m % (s[-1] + seg[-1])))))
    return pd.concat([geometry.iloc[k:], geometry.iloc[:k]], ignore_index=True)


def _fused_distance(tel: pd.DataFrame, lap_s: float, center: _Centerline, tf: dict,
                    sigma_s: float = 1.0):
    """
    Distance along the lap from speed + position.

    Integrated speed is smooth but drifts; projected position has no drift but
    ~3 m of jitter. Take the integrated speed and add a heavily smoothed
    (position - integrated speed) correction, so short-term shape comes from
    speed and long-term placement from position.
    Returns (distance at samples, distance at t=0, distance at t=lap_s).
    """
    t = tel["t"].to_numpy(float)
    v = tel["Speed"].to_numpy(float) / 3.6
    xy = _apply(tf, tel[["Xraw", "Yraw"]].to_numpy())
    arc = _unwrap_lap(center.project(xy), center.length)

    t_ext = np.concatenate([[0.0], t])
    v_ext = np.concatenate([[v[0]], v])
    integ = np.concatenate([[0.0], np.cumsum(np.diff(t_ext) * (v_ext[1:] + v_ext[:-1]) / 2)])
    resid = arc - integ[1:]
    w = np.exp(-0.5 * ((t[:, None] - t[None, :]) / sigma_s) ** 2)
    resid_s = (w @ resid) / w.sum(1)

    dist = integ[1:] + resid_s
    d0 = float(resid_s[0])                                         # integ(0) = 0
    d1 = float(integ[-1] + v[-1] * (lap_s - t[-1]) + resid_s[-1])
    return dist, d0, d1, xy


def densify_gaps(tel: pd.DataFrame, gap_s: float = 0.3, step_s: float = 0.1) -> pd.DataFrame:
    """
    OpenF1 car data is irregular: about 24 stretches per lap are longer than 0.4 s
    with no sample, some right where a car brakes. Interpolating time against
    distance in a straight line across such a gap assumes constant speed, which is
    wrong while braking (up to ~0.04 s) and leaves the first sample of a section
    tens of metres late.

    Insert samples across each gap using the speed at both ends (constant
    acceleration), scaled so distance still lands exactly on the next sample.
    Inserted rows are flagged Interpolated=True; brake/throttle/gear take the
    nearest measured value and should not be treated as measurements.
    Needs columns: t (s), Distance (m), Speed (km/h), plus any others.
    """
    tel = tel.reset_index(drop=True)
    tel["Interpolated"] = False
    rows = []
    t, d, v = tel["t"].to_numpy(float), tel["Distance"].to_numpy(float), tel["Speed"].to_numpy(float) / 3.6
    for i in range(len(tel) - 1):
        gap = t[i + 1] - t[i]
        if gap <= gap_s:
            continue
        n = int(np.ceil(gap / step_s)) - 1
        total = (v[i] + v[i + 1]) / 2 * gap
        for k in range(1, n + 1):
            tau = gap * k / (n + 1)
            model = v[i] * tau + (v[i + 1] - v[i]) * tau ** 2 / (2 * gap)
            frac = model / total if total > 0 else k / (n + 1)
            row = (tel.iloc[i] if tau < gap / 2 else tel.iloc[i + 1]).copy()    # nearest measured values
            row["t"] = t[i] + tau
            row["Distance"] = d[i] + (d[i + 1] - d[i]) * frac
            row["Speed"] = tel.at[i, "Speed"] + (tel.at[i + 1, "Speed"] - tel.at[i, "Speed"]) * tau / gap
            for col in ("X", "Y"):
                if col in tel.columns:
                    row[col] = tel.at[i, col] + (tel.at[i + 1, col] - tel.at[i, col]) * frac
            row["Interpolated"] = True
            rows.append(row)
    if not rows:
        return tel
    return pd.concat([tel, pd.DataFrame(rows)], ignore_index=True).sort_values("t", kind="stable").reset_index(drop=True)


def _finish_telemetry(tel: pd.DataFrame, lap_s: float, driver: str,
                      center: Optional[_Centerline], tf: Optional[dict]) -> pd.DataFrame:
    tel = tel.copy()
    if center is not None and tf is not None:
        dist, d0, d1, xy = _fused_distance(tel, lap_s, center, tf)
        # Line crossings are by definition distance 0 and one lap length; stretch
        # each lap by the few metres it is off so every driver shares one axis.
        dist = (dist - d0) / (d1 - d0) * center.length
        tel["X"], tel["Y"] = xy[:, 0], xy[:, 1]
        first, last = tel.iloc[0].to_dict(), tel.iloc[-1].to_dict()
        tel["Distance"] = np.maximum.accumulate(np.clip(dist, 0.0, center.length))
        tel = pd.concat([
            pd.DataFrame([{**first, "t": 0.0, "Distance": 0.0}]),
            tel,
            pd.DataFrame([{**last, "t": lap_s, "Distance": center.length}]),
        ], ignore_index=True)
    else:
        # Fallback (what FastF1 does): integrate speed.
        dt_s = np.diff(tel["t"], prepend=0.0)
        tel["Distance"] = np.cumsum(tel["Speed"] / 3.6 * dt_s)
        tel["X"], tel["Y"] = tel["Xraw"] * POSITION_UNIT_M, tel["Yraw"] * POSITION_UNIT_M

    tel = densify_gaps(tel)
    tel["Time"] = pd.to_timedelta(tel["t"], unit="s")
    tel["Driver"] = driver
    return tel.drop(columns=["t", "Xraw", "Yraw"]).reset_index(drop=True)


# ── Main entry point ─────────────────────────────────────────────────────────

def _td(seconds) -> pd.Timedelta:
    return pd.NaT if seconds is None or pd.isna(seconds) else pd.Timedelta(seconds=float(seconds))


def load_qualifying_session_openf1(
    year: int,
    circuit: str,
    geometry: Optional[pd.DataFrame] = None,
    verbose: bool = True,
    exact_origin: bool = False,
) -> dict:
    """
    Build the same dict as load_qualifying.load_qualifying_session from OpenF1.

    geometry: track outline DataFrame (x_m, y_m). Strongly recommended —
              it puts every driver on one shared distance axis.
    exact_origin: place distance 0 exactly on the timing line by shifting the
              origin, instead of rolling the outline by whole points. Use this
              with the official outline, whose points are ~10 m apart.
    """
    log = print if verbose else (lambda *a, **k: None)
    sess = find_session(year, circuit)
    sk = sess["session_key"]
    log(f"Loading {year} {circuit} Qualifying from OpenF1 (session_key={sk}) ...")

    drivers = {d["driver_number"]: d for d in _get("drivers", session_key=sk)}
    laps_raw = _get("laps", session_key=sk)
    results = {r["driver_number"]: r for r in _get("session_result", session_key=sk)}
    stints = _get("stints", session_key=sk)
    race_control = _get("race_control", session_key=sk)
    deleted = deleted_lap_times(race_control)

    meeting_name = sess.get("location", circuit)
    try:
        meetings = _get("meetings", meeting_key=sess["meeting_key"])
        if meetings:
            meeting_name = meetings[0].get("meeting_name") or meeting_name
    except Exception:
        pass

    session_info = {
        "year": year,
        "circuit": circuit,
        "event_name": meeting_name,
        "location": sess.get("location", ""),
        "country": sess.get("country_name", ""),
        "date": str(sess.get("date_start", "")),
    }

    laps = pd.DataFrame(laps_raw)
    laps["date_start"] = pd.to_datetime(laps["date_start"], utc=True, format="mixed")
    session_start = laps["date_start"].min()

    # ── tyres: compound + age on each lap ──
    def tyre_for(num: int, lap_no: int):
        for s in stints:
            if s["driver_number"] == num and s["lap_start"] <= lap_no <= s["lap_end"]:
                age = (s.get("tyre_age_at_start") or 0) + (lap_no - s["lap_start"])
                return s.get("compound"), age, (s.get("tyre_age_at_start") or 0) == 0
        return None, None, None

    # ── all laps (timed, not out-laps) ──
    rows = []
    for _, l in laps.iterrows():
        if pd.isna(l["lap_duration"]) or l["is_pit_out_lap"]:
            continue
        num = int(l["driver_number"])
        d = drivers.get(num)
        if d is None:
            continue
        comp, age, fresh = tyre_for(num, int(l["lap_number"]))
        rows.append({
            "Driver": d["name_acronym"], "DriverNumber": str(num), "Team": d["team_name"],
            "LapNumber": int(l["lap_number"]),
            "LapTime": _td(l["lap_duration"]),
            "Sector1Time": _td(l["duration_sector_1"]),
            "Sector2Time": _td(l["duration_sector_2"]),
            "Sector3Time": _td(l["duration_sector_3"]),
            "SpeedI1": l["i1_speed"], "SpeedI2": l["i2_speed"], "SpeedST": l["st_speed"],
            "Compound": comp, "TyreLife": age, "FreshTyre": fresh,
            "LapStartTime": l["date_start"] - session_start,
            "_date_start": l["date_start"], "_num": num,
            "_deleted": _is_deleted(num, float(l["lap_duration"]), deleted),
        })
    all_laps_full = pd.DataFrame(rows)
    all_laps = all_laps_full[~all_laps_full["_deleted"]].copy()
    all_laps["IsPersonalBest"] = (
        all_laps.groupby("Driver")["LapTime"].transform("min") == all_laps["LapTime"]
    )

    # ── fastest valid lap per driver, ordered by official classification ──
    fastest_rows = []
    for drv, grp in all_laps.groupby("Driver"):
        fastest_rows.append(grp.loc[grp["LapTime"].idxmin()])
    fastest = pd.DataFrame(fastest_rows).reset_index(drop=True)

    def classification(num: int):
        r = results.get(num)
        if not r:
            return None, None
        times = r.get("duration") or []
        reached = max((i for i, v in enumerate(times) if v), default=0) + 1 if isinstance(times, list) else None
        return r.get("position"), f"Q{reached}" if reached else None

    pos_q = fastest["_num"].map(classification)
    fastest["QualifyingPosition"] = [p for p, _ in pos_q]
    fastest["QSessionReached"] = [q for _, q in pos_q]
    if fastest["QualifyingPosition"].isna().any():      # no official result: rank by time
        order = fastest["LapTime"].rank(method="first").astype(int)
        fastest["QualifyingPosition"] = fastest["QualifyingPosition"].fillna(order)
    fastest["QualifyingPosition"] = fastest["QualifyingPosition"].astype(int)
    fastest = fastest.sort_values("QualifyingPosition").reset_index(drop=True)

    # ── telemetry for each driver's fastest lap ──
    raw_tel: dict[str, pd.DataFrame] = {}
    lap_by_driver = {}
    for _, row in fastest.iterrows():
        lap = laps[(laps["driver_number"] == row["_num"]) & (laps["lap_number"] == row["LapNumber"])].iloc[0].to_dict()
        lap["date_start"] = lap["date_start"].isoformat()
        lap_by_driver[row["Driver"]] = float(lap["lap_duration"])
        try:
            tel = _fetch_lap_telemetry(sk, int(row["_num"]), lap)
        except Exception as e:
            log(f"  Warning: telemetry fetch failed for {row['Driver']}: {e}")
            continue
        if tel is None:
            log(f"  Warning: not enough telemetry for {row['Driver']}")
            continue
        raw_tel[row["Driver"]] = tel

    center = tf = None
    if geometry is not None and not geometry.empty and raw_tel:
        geom_xy = geometry[["x_m", "y_m"]].to_numpy(float)
        tf = register_to_geometry([t[["Xraw", "Yraw"]].to_numpy() for t in raw_tel.values()], geom_xy)
        center = _Centerline(geom_xy)
        log(f"  Registered OpenF1 positions to track geometry: scale={tf['scale']:.4f} (1.0 = decimetre units), "
            f"rotation={tf['rotation_deg']:.2f} deg, median error {tf['median_err_m']:.1f} m "
            f"(95th pct {tf['p95_err_m']:.1f} m); centerline length {center.length:.0f} m")
        if tf["median_err_m"] > 8:
            log("  WARNING: poor registration — section times may be unreliable")

        # The geometry's index 0 is not the timing line. Estimate how far it is
        # from where every driver's lap starts and ends, then re-origin it.
        starts, ends = [], []
        for d, t in raw_tel.items():
            _, d0, d1, _ = _fused_distance(t, lap_by_driver[d], center, tf)
            starts.append(d0)
            ends.append(d1 - center.length)
        offset = float(np.median(starts + ends))
        log(f"  Timing line sits {offset:.1f} m past geometry origin "
            f"(lap-start estimates {np.median(starts):.1f} m, lap-end {np.median(ends):.1f} m, "
            f"spread {np.std(starts + ends):.1f} m)")
        spread = float(np.std(starts + ends))
        if spread > 15 or abs(offset) > 0.1 * center.length:
            log("  WARNING: lap start/end do not agree on where the timing line is "
                "(registration may have the wrong rotation) — section times are unreliable")
        if exact_origin:
            center = _Centerline(geom_xy, origin=offset)
        else:
            geometry = roll_geometry(geometry, offset)
            geom_xy = geometry[["x_m", "y_m"]].to_numpy(float)
            center = _Centerline(geom_xy)
    elif geometry is None:
        log("  No track geometry: falling back to speed-integrated distance")

    telemetry = {d: _finish_telemetry(t, lap_by_driver[d], d, center, tf) for d, t in raw_tel.items()}

    # ── weather ──
    weather = pd.DataFrame(_get("weather", session_key=sk))
    if not weather.empty:
        weather = weather.rename(columns={
            "air_temperature": "AirTemp", "track_temperature": "TrackTemp",
            "humidity": "Humidity", "rainfall": "Rainfall",
            "wind_speed": "WindSpeed", "wind_direction": "WindDirection",
        })

    # ── drivers with no valid time ──
    red_flags = [_ts(r["date"]) for r in race_control if (r.get("flag") or "").upper() == "RED"]
    have_time = set(fastest["Driver"])
    dnq = []
    for num, d in drivers.items():
        abbr = d["name_acronym"]
        if abbr in have_time:
            continue
        mine = laps[laps["driver_number"] == num].sort_values("date_start")
        red = False
        if red_flags and not mine.empty:
            last = mine.iloc[-1]
            end = last["date_start"] + pd.Timedelta(seconds=float(last["lap_duration"]) if pd.notna(last["lap_duration"]) else 120)
            red = any(last["date_start"] <= rf <= end for rf in red_flags)
        dnq.append({
            "driver": abbr, "full_name": d.get("full_name", abbr), "team": d.get("team_name", "Unknown"),
            "reason": "flying lap interrupted by red flag" if red else "did not complete a timed lap",
            "lap_count": int(len(mine)),
        })
        log(f"  DNQ driver: {abbr} ({d.get('full_name', abbr)}) — {dnq[-1]['reason']}")

    keep = ["Driver", "DriverNumber", "Team", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
            "SpeedI1", "SpeedI2", "SpeedST", "Compound", "TyreLife", "FreshTyre",
            "LapStartTime", "LapNumber", "IsPersonalBest", "QualifyingPosition", "QSessionReached"]
    log(f"  Loaded {len(fastest)} drivers, {len(all_laps)} timed laps, "
        f"{len(all_laps_full) - len(all_laps)} deleted; telemetry for {len(telemetry)}")

    return {
        "session_info": session_info,
        "driver_fastest_laps": fastest[keep],
        "all_laps": all_laps[[c for c in keep if c not in ("QualifyingPosition", "QSessionReached")]],
        "weather": weather,
        "telemetry": telemetry,
        "dnq_drivers": dnq,
        "registration": tf,
        "centerline": center,              # lets callers place official corners on the lap
        "circuit_key": sess.get("circuit_key"),
        "geometry_aligned": geometry,      # use this for event detection: origin is the timing line
    }
