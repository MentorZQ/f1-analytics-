"""
detect_events_official.py
-------------------------
Split a lap into corner and straight sections using the OFFICIAL corner list
(numbers + apex positions) and what the drivers actually did.

Why not the geometry detector (detect_events.py)?
  - It finds corners as curvature peaks on a geometry file. Those files go
    stale when a circuit is rebuilt, and numbering has to be mapped to official
    turn names by hand (circuit_labels.py).
  - Its boundaries are placed by geometry, so they can fall inside a braking
    zone. A driver who brakes later then gains time in one section and gives it
    back in the next, which looks like a large swing that is really one
    braking-point difference.

Here, the official list says WHERE the corners are and what they are called.
The telemetry says WHERE EACH CORNER STARTS AND ENDS:

  entry : where the earliest driver lifts or starts braking, minus a small margin, so
          that no driver is mid-brake at the boundary. Corners with no braking
          use the curvature of the track outline instead.
  exit  : where the last driver is fully back on the throttle, plus a small
          margin. Flat corners use curvature.
  A driver whose point is far from the pack (traffic, a lift for a yellow) is
  ignored, so one odd lap cannot widen a zone for everyone.

Everything between corners is a straight. Corners separated by a short gap are
merged into one sequence (chicanes, esses). A sequence longer than MAX_SEQUENCE_M
is cut between two of its corners, at a point where no driver is braking, so a cut
never lands mid-brake. If there is no such point it stays whole. The sections tile
the lap exactly.

Output matches detect_events.detect_circuit_events, so the chunkers do not change.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

# ── Tunable parameters ───────────────────────────────────────────────────────

BRAKE_GAP_M          = 20.0    # brake samples closer than this are one braking zone
BRAKE_SEARCH_BEFORE  = 300.0   # a braking zone must end within this far before the apex
BRAKE_SEARCH_AFTER   = 25.0    # ...or at most this far after it
BRAKE_MAX_LOOKBACK   = 600.0   # and start no more than this far before the apex
LIFT_MAX_LOOKBACK_M  = 150.0   # a lift this far before the brake zone still belongs to the corner entry
FULL_THROTTLE_PCT    = 98.0
FULL_THROTTLE_HOLD_M = 30.0    # throttle must stay full this long to count as "back on power"
EXIT_SEARCH_M        = 450.0
OUTLIER_M            = 60.0    # ignore a driver's braking/throttle point this far from the pack's median
                               # (traffic, a lift for a yellow); every other driver is enclosed
MARGIN_M             = 10.0
MIN_DRIVER_SHARE     = 0.40    # share of drivers needed to trust a telemetry boundary
KAPPA_EXTENT         = 0.003   # 1/m (radius <= 333 m) counts as "in the corner" for the fallback
MIN_HALF_WIDTH_M     = 25.0
MERGE_GAP_M          = 80.0    # corner zones closer than this form one sequence
MAX_SEQUENCE_M       = 450.0   # a longer sequence is cut where no driver is braking (see _split_group)
MIN_PIECE_M          = 120.0   # ...but never into pieces shorter than this
CUT_GRID_M           = 5.0


# ── Curvature of the track outline (fallback and corner metadata) ────────────

def curvature_profile(outline_xy: np.ndarray, origin: float = 0.0,
                      step: float = 5.0, smooth_m: float = 30.0):
    """
    Curvature (1/m) along the lap. Returns (distance, kappa, lap_length), with
    distance measured from `origin` metres along the outline.
    """
    pts = np.vstack([outline_xy, outline_xy[:1]])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    length = float(s[-1])
    grid = np.arange(0, length, step)
    x, y = np.interp(grid, s, pts[:, 0]), np.interp(grid, s, pts[:, 1])
    heading = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    width = max(int(smooth_m / step), 1) | 1
    kappa = np.abs(np.gradient(uniform_filter1d(heading, width, mode="nearest"), step))
    d = (grid - origin) % length
    order = np.argsort(d)
    return d[order], kappa[order], length


def _curvature_extent(kd: np.ndarray, kk: np.ndarray, apex: float) -> tuple[float, float]:
    i = int(np.argmin(np.abs(kd - apex)))
    lo = hi = i
    while lo > 0 and kk[lo - 1] >= KAPPA_EXTENT:
        lo -= 1
    while hi < len(kd) - 1 and kk[hi + 1] >= KAPPA_EXTENT:
        hi += 1
    return min(kd[lo], apex - MIN_HALF_WIDTH_M), max(kd[hi], apex + MIN_HALF_WIDTH_M)


# ── Per-driver telemetry measurements ────────────────────────────────────────

def brake_zones(dist: np.ndarray, braking: np.ndarray, gap_m: float = BRAKE_GAP_M):
    """
    Contiguous braking zones as (start, end) distances.

    A zone starts at the last sample where the driver was NOT braking, because the
    true brake-on point lies somewhere between that sample and the first braking
    one. With gaps in the data that can be tens of metres, so starting at the
    first braking sample would leave the real onset outside the zone.
    """
    zones, start, last = [], None, None
    prev_d = None
    for d, on in zip(dist, braking):
        if on:
            if start is not None and d - last <= gap_m:
                last = d
            else:
                if start is not None:
                    zones.append((start, last))
                start = prev_d if prev_d is not None else d
                last = d
        prev_d = d
    if start is not None:
        zones.append((start, last))
    return zones


def _entry_for_driver(zones, apex: float, prev_apex: float, dist=None, throttle=None):
    ok = [z for z in zones
          if apex - BRAKE_SEARCH_BEFORE <= z[1] <= apex + BRAKE_SEARCH_AFTER
          and z[0] >= max(prev_apex + 10.0, apex - BRAKE_MAX_LOOKBACK)]
    if not ok:
        return None
    start = min(z[0] for z in ok)
    if dist is None or throttle is None:
        return start
    # Lift-and-coast: a driver who comes off the throttle before braking has begun the
    # corner entry already. Walk back to their last full-throttle sample.
    floor = max(prev_apex + 10.0, start - LIFT_MAX_LOOKBACK_M)
    j = int(np.searchsorted(dist, start, side="right")) - 1
    while j > 0 and throttle[j] < FULL_THROTTLE_PCT and dist[j - 1] >= floor:
        j -= 1
    return float(min(start, dist[j]))


def _exit_for_driver(dist: np.ndarray, throttle: np.ndarray, apex: float):
    full = throttle >= FULL_THROTTLE_PCT
    idx = np.flatnonzero((dist >= apex) & (dist <= apex + EXIT_SEARCH_M))
    for i in idx:
        end = np.searchsorted(dist, dist[i] + FULL_THROTTLE_HOLD_M)
        if end > len(dist):
            break
        if full[i:end].all() and dist[min(end, len(dist)) - 1] - dist[i] >= FULL_THROTTLE_HOLD_M * 0.6:
            return float(dist[i])
    return None


def _trimmed(values, which: str) -> float:
    """Earliest (min) or latest (max) value, ignoring points far from the median."""
    v = np.asarray(values, float)
    kept = v[np.abs(v - np.median(v)) <= OUTLIER_M]
    kept = kept if len(kept) else v
    return float(kept.min() if which == "min" else kept.max())


# ── Main entry point ─────────────────────────────────────────────────────────

def detect_official_events(
    telemetry: dict,
    apexes: list[dict],
    lap_length: float,
    curvature: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[dict]:
    """
    telemetry : {driver: DataFrame with Distance, Brake, Throttle}
    apexes    : [{"number": 1, "label": "1", "dist": 809.0}, ...] official corners
    lap_length: metres
    curvature : (distance, kappa) from curvature_profile(); enables the flat-corner fallback

    Returns events in lap order that tile 0..lap_length, in the same schema as
    detect_events.detect_circuit_events.
    """
    apexes = sorted(apexes, key=lambda a: a["dist"])
    n_drivers = max(len(telemetry), 1)
    def measured(t):
        # Inserted (interpolated) rows carry guessed brake/throttle: use measurements only.
        return t[~t["Interpolated"]] if "Interpolated" in t.columns else t

    tel = {d: (m["Distance"].to_numpy(float), m["Brake"].to_numpy(bool), m["Throttle"].to_numpy(float))
           for d, t in telemetry.items() if t is not None and not t.empty for m in [measured(t)]}
    zones_by_driver = {d: brake_zones(D, B) for d, (D, B, _) in tel.items()}

    # ── one entry/exit per official corner ──
    corners = []
    for k, ap in enumerate(apexes):
        a = ap["dist"]
        prev = apexes[k - 1]["dist"] if k > 0 else -1e9
        entries = [e for d, (D, _, T) in tel.items()
                   if (e := _entry_for_driver(zones_by_driver[d], a, prev, D, T)) is not None]
        exits = [x for d, (D, _, T) in tel.items() if (x := _exit_for_driver(D, T, a)) is not None]

        c_lo, c_hi = _curvature_extent(*curvature, a) if curvature is not None else (a - MIN_HALF_WIDTH_M, a + MIN_HALF_WIDTH_M)
        braked = len(entries) >= MIN_DRIVER_SHARE * n_drivers
        powered = len(exits) >= MIN_DRIVER_SHARE * n_drivers
        # Telemetry decides wherever drivers actually braked / came back to full
        # throttle. Curvature only fills in for flat corners, where there is no
        # braking to see: widening braked corners by curvature would count every
        # gentle bend as part of the corner.
        entry = _trimmed(entries, "min") - MARGIN_M if braked else c_lo
        if entries and not braked:                    # a few drivers brake for a corner most take flat:
            entry = min(entry, _trimmed(entries, "min") - MARGIN_M)   # their braking still belongs inside it
        exit_ = _trimmed(exits, "max") + MARGIN_M if powered else c_hi
        if not braked:
            exit_ = max(exit_, c_hi)
        corners.append({
            **ap,
            "entry": float(max(0.0, min(entry, a - 20.0))),
            "exit": float(min(lap_length, max(exit_, a + 20.0))),
            "braked": braked, "entry_drivers": len(entries), "exit_drivers": len(exits),
        })

    # ── merge corners that run into each other ──
    groups = []
    for c in corners:
        if groups and c["entry"] - groups[-1]["exit"] < MERGE_GAP_M:
            g = groups[-1]
            g["corners"].append(c)
            g["exit"] = max(g["exit"], c["exit"])
        else:
            groups.append({"entry": c["entry"], "exit": c["exit"], "corners": [c]})

    groups = [piece for g in groups for piece in _split_group(g, zones_by_driver, tel)]

    kd, kk = curvature if curvature is not None else (None, None)

    def kappa_at(d):
        return float(np.interp(d, kd, kk)) if kd is not None else 0.0

    events = []
    # lead straight
    if groups[0]["entry"] > 1.0:
        first = groups[0]["corners"][0]
        events.append(_straight(f"Main straight (start, before T{first['label']})",
                                0.0, groups[0]["entry"], kd, kk))
    for i, g in enumerate(groups):
        nums = [f"T{c['label']}" for c in g["corners"]]
        name = "-".join(nums)
        peaks = [c["dist"] for c in g["corners"]]
        kap = [kappa_at(p) for p in peaks]
        events.append({
            "id": name, "display_id": name,
            "type": "corner" if len(nums) == 1 else "corner_sequence",
            "dist_start": g["entry"], "dist_end": g["exit"], "length_m": g["exit"] - g["entry"],
            "n_peaks": len(peaks), "peak_dists": peaks, "peak_kappas": kap, "max_kappa": max(kap),
            "corner_numbers": [c["number"] for c in g["corners"]],
            "braked_entry": g["corners"][0]["braked"],
            "members": [{"label": c["label"], "apex": c["dist"], "entry": c["entry"], "exit": c["exit"],
                         "braked": c["braked"], "entry_drivers": c["entry_drivers"],
                         "exit_drivers": c["exit_drivers"]} for c in g["corners"]],
        })
        if i + 1 < len(groups) and groups[i + 1]["entry"] - g["exit"] > 1.0:     # cuts inside a sequence touch
            nxt = groups[i + 1]
            events.append(_straight(f"Between {nums[-1]}-T{nxt['corners'][0]['label']}",
                                    g["exit"], nxt["entry"], kd, kk))
    # tail straight
    if lap_length - groups[-1]["exit"] > 1.0:
        events.append(_straight("Main straight (finish, after final corner)",
                                groups[-1]["exit"], lap_length, kd, kk))
    return events


def _braking_free_cuts(zones_by_driver: dict, lo: float, hi: float, tel: dict | None = None) -> list[float]:
    """
    One cut per stretch of [lo, hi] where no driver is inside a braking zone: the point
    where the drivers' median throttle is highest (the car is most clearly on the power,
    so nobody has started lifting for the next corner). Without telemetry, the middle.
    """
    if hi <= lo:
        return []
    grid = np.arange(lo, hi + 1e-9, CUT_GRID_M)
    busy = np.zeros(len(grid), bool)
    for zones in zones_by_driver.values():
        for a, b in zones:
            busy |= (grid >= a) & (grid <= b)
    cuts, start = [], None
    for x, is_busy in zip(grid, busy):
        if not is_busy and start is None:
            start = x
        if (is_busy or x == grid[-1]) and start is not None:
            end = x - CUT_GRID_M if is_busy else x
            if end >= start:
                cuts.append(_best_cut(start, end, tel))
            start = None
    return cuts


def _best_cut(start: float, end: float, tel: dict | None) -> float:
    xs = np.arange(start, end + 1e-9, CUT_GRID_M)
    if tel is None or len(xs) < 2:
        return float((start + end) / 2)
    med = np.median([np.interp(xs, D, T) for D, _, T in tel.values() if len(D) > 1], axis=0)
    best = np.flatnonzero(med >= med.max() - 1e-9)
    return float(xs[best[len(best) // 2]])          # the middle of the best stretch


def _split_group(g: dict, zones_by_driver: dict, tel: dict | None = None) -> list[dict]:
    """Cut a long sequence between two of its corners, only where nobody is braking."""
    cs = g["corners"]
    if len(cs) < 2 or g["exit"] - g["entry"] <= MAX_SEQUENCE_M:
        return [g]
    candidates = []
    for k in range(len(cs) - 1):
        for cut in _braking_free_cuts(zones_by_driver, cs[k]["dist"], cs[k + 1]["dist"], tel):
            if cut - g["entry"] >= MIN_PIECE_M and g["exit"] - cut >= MIN_PIECE_M:
                candidates.append((k, cut))
    if not candidates:
        return [g]
    mid = (g["entry"] + g["exit"]) / 2
    k, cut = min(candidates, key=lambda kc: abs(kc[1] - mid))      # the cut that halves the sequence best
    left = {"entry": g["entry"], "exit": cut, "corners": [{**c, "exit": min(c["exit"], cut)} for c in cs[:k + 1]]}
    right = {"entry": cut, "exit": g["exit"], "corners": [{**c, "entry": max(c["entry"], cut)} for c in cs[k + 1:]]}
    return _split_group(left, zones_by_driver, tel) + _split_group(right, zones_by_driver, tel)


def _straight(name: str, start: float, end: float, kd, kk) -> dict:
    if kd is not None:
        m = (kd >= start) & (kd <= end)
        min_kappa = float(kk[m].min()) if m.any() else 0.0
    else:
        min_kappa = 0.0
    return {"id": name, "display_id": name, "type": "straight",
            "dist_start": float(start), "dist_end": float(end), "length_m": float(end - start),
            "min_kappa": min_kappa}
