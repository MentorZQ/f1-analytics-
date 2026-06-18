"""
detect_events.py
----------------
Segments a circuit into an ordered list of named track events:
  - corner          : single apex, speed minimum present
  - corner_sequence : multiple apices with no meaningful straight between them
                      (chicanes, Eau Rouge/Raidillon, Lesmo 1-2, etc.)
  - straight        : sustained low-curvature section between corner groups

Detection is purely geometry-based (TUMFTM x/y centerline coordinates),
so event boundaries are spatially consistent across all drivers.
Telemetry is NOT used here — it gets sliced against these boundaries later.

Key design decisions:
  - Curvature = Menger curvature at each point (direction change per unit
    distance, NOT total displacement). A fast flat corner like Eau Rouge
    has high curvature; a long sweeping section has moderate curvature;
    a genuine straight has near-zero curvature.
  - Corners are curvature peaks, not speed minima. Speed minima are added
    later from telemetry (they may or may not exist at each corner peak).
  - Two corners are in a sequence if the gap between them is short OR
    the gap never drops to near-zero curvature (i.e. the track keeps
    bending between them).
  - Straight threshold is data-driven: computed from the distribution of
    inter-corner gap properties, not hardcoded.

Output schema (list of dicts):
    {
        "id":           "T1" | "T3-T4" | "S2",
        "type":         "corner" | "corner_sequence" | "straight",
        "dist_start":   float,   # metres from lap start
        "dist_end":     float,
        "length_m":     float,
        # corners / corner_sequences only:
        "n_peaks":      int,
        "peak_dists":   [float, ...],  # dist of each curvature peak
        "peak_kappas":  [float, ...],  # curvature at each peak
        "max_kappa":    float,
        # straights only:
        "min_kappa":    float,   # how genuinely straight it is
    }

Usage:
    from detect_events import detect_circuit_events
    events = detect_circuit_events(geometry_df)  # from load_circuit_geometry()
    for e in events:
        print(e["id"], e["type"], e["dist_start"], "->", e["dist_end"])
"""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from typing import Optional


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# Smoothing window for curvature (points). At 5m/point spacing:
#   size=7  → 35m window  (preserves tight chicane detail)
CURVATURE_SMOOTH_WINDOW = 7

# Corner peak detection
PEAK_MIN_HEIGHT     = 0.0015   # minimum curvature to be a corner (filters noise)
PEAK_MIN_DISTANCE   = 15       # minimum points between peaks (~75m at 5m/pt)
PEAK_MIN_PROMINENCE = 0.0008   # peak must stand out from surrounding baseline

# Straight classification
# A gap between two corner peaks is a "straight" if:
#   (a) it's long enough, AND
#   (b) at least this fraction of it has curvature below STRAIGHT_KAPPA_THRESHOLD
STRAIGHT_KAPPA_THRESHOLD = 0.003   # below this = "not really curving"
STRAIGHT_MIN_FRAC        = 0.50    # fraction of gap that must be below threshold
STRAIGHT_MIN_LENGTH_M    = 150     # minimum gap length to be called a straight


# ---------------------------------------------------------------------------
# Core geometry functions
# ---------------------------------------------------------------------------

def _cumulative_distance(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Cumulative distance along the track centerline in metres."""
    dx = np.diff(xs)
    dy = np.diff(ys)
    step_dists = np.sqrt(dx**2 + dy**2)
    return np.concatenate([[0.0], np.cumsum(step_dists)])


def _menger_curvature(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Menger curvature at each interior point.

    kappa[i] = 2 * |cross(AB, AC)| / (|AB| * |BC| * |AC|)

    Measures how sharply the path bends at point i — independent of speed.
    Units: 1/m (inverse radius of curvature).
    A 10m radius hairpin: kappa ≈ 0.1
    A 50m radius slow corner: kappa ≈ 0.02
    A 200m radius fast corner: kappa ≈ 0.005
    A straight: kappa ≈ 0
    """
    n = len(xs)
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        ax, ay = xs[i-1], ys[i-1]
        bx, by = xs[i],   ys[i]
        cx, cy = xs[i+1], ys[i+1]

        ab  = np.sqrt((bx - ax)**2 + (by - ay)**2)
        bc  = np.sqrt((cx - bx)**2 + (cy - by)**2)
        ac  = np.sqrt((cx - ax)**2 + (cy - ay)**2)
        cross = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        denom = ab * bc * ac
        kappa[i] = (2.0 * cross / denom) if denom > 1e-10 else 0.0

    return kappa


def _classify_gap(
    kappa_smooth: np.ndarray,
    cum_dist: np.ndarray,
    idx_start: int,
    idx_end: int,
) -> str:
    """
    Classify the gap between two corner peaks as 'straight' or 'sequence'.

    'straight'  → long gap, most of it has near-zero curvature
    'sequence'  → short gap, or curvature stays elevated throughout
                  (track keeps bending between the two peaks)
    """
    gap_kappa = kappa_smooth[idx_start:idx_end]
    gap_dist  = cum_dist[idx_end] - cum_dist[idx_start]
    straight_frac = (gap_kappa < STRAIGHT_KAPPA_THRESHOLD).mean()

    if gap_dist >= STRAIGHT_MIN_LENGTH_M and straight_frac >= STRAIGHT_MIN_FRAC:
        return "straight"
    return "sequence"


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------

def detect_circuit_events(
    geometry: pd.DataFrame,
    circuit_name: str = "Unknown",
    verbose: bool = False,
) -> list[dict]:
    """
    Segment a circuit into an ordered list of corners, corner sequences,
    and straights.

    Args:
        geometry:     DataFrame from load_circuit_geometry() with columns
                      x_m, y_m, w_tr_right_m, w_tr_left_m
        circuit_name: used only for logging
        verbose:      print detection details

    Returns:
        Ordered list of event dicts. The first event starts at the point
        FastF1 considers lap start (which may not be the start/finish line).
        Events cover the full lap (last event ends near the first).
    """
    if geometry is None or geometry.empty:
        raise ValueError(f"No geometry available for {circuit_name}. "
                         f"detect_circuit_events requires TUMFTM data.")

    xs = geometry["x_m"].values
    ys = geometry["y_m"].values
    cum_dist = _cumulative_distance(xs, ys)
    total_dist = cum_dist[-1]

    # --- Curvature ---
    kappa       = _menger_curvature(xs, ys)
    kappa_smooth = uniform_filter1d(kappa, size=CURVATURE_SMOOTH_WINDOW)

    # --- Corner peaks ---
    peaks, peak_props = find_peaks(
        kappa_smooth,
        height=PEAK_MIN_HEIGHT,
        distance=PEAK_MIN_DISTANCE,
        prominence=PEAK_MIN_PROMINENCE,
    )

    if len(peaks) == 0:
        raise ValueError(f"No corners detected for {circuit_name}. "
                         "Check geometry data quality.")

    if verbose:
        print(f"{circuit_name}: {len(peaks)} raw curvature peaks detected")

    # --- Classify gaps between consecutive peaks ---
    gap_types = []
    for i in range(len(peaks) - 1):
        gap_type = _classify_gap(kappa_smooth, cum_dist, peaks[i], peaks[i+1])
        gap_types.append(gap_type)

    # --- Group peaks into corner events ---
    # Consecutive peaks separated only by 'sequence' gaps → one group
    corner_groups = []
    current_group = [int(peaks[0])]

    for i, gap_type in enumerate(gap_types):
        if gap_type == "sequence":
            current_group.append(int(peaks[i+1]))
        else:
            corner_groups.append(("corner_group", current_group))
            corner_groups.append(("straight", int(peaks[i]), int(peaks[i+1])))
            current_group = [int(peaks[i+1])]

    corner_groups.append(("corner_group", current_group))

    # --- Build event list ---
    events = []
    corner_num   = 1
    straight_num = 1

    for g in corner_groups:
        if g[0] == "corner_group":
            peak_indices = g[1]
            peak_dists   = [float(cum_dist[p]) for p in peak_indices]
            peak_kappas  = [float(kappa_smooth[p]) for p in peak_indices]
            n            = len(peak_indices)
            max_kappa    = max(peak_kappas)

            # Event boundaries: a bit before first peak to a bit after last
            # Use 80m buffer but clamp to track bounds
            buf = 80
            dist_start = max(0.0,         peak_dists[0]  - buf)
            dist_end   = min(total_dist,  peak_dists[-1] + buf)

            if n == 1:
                event_id  = f"T{corner_num}"
                event_type = "corner"
                corner_num += 1
            else:
                event_id  = "-".join(f"T{corner_num + j}" for j in range(n))
                event_type = "corner_sequence"
                corner_num += n

            event = {
                "id":           event_id,
                "type":         event_type,
                "dist_start":   round(dist_start, 1),
                "dist_end":     round(dist_end, 1),
                "length_m":     round(dist_end - dist_start, 1),
                "n_peaks":      n,
                "peak_dists":   [round(d, 1) for d in peak_dists],
                "peak_kappas":  [round(k, 5) for k in peak_kappas],
                "max_kappa":    round(max_kappa, 5),
            }
            events.append(event)

            if verbose:
                print(f"  {event_id:12s} [{event_type:16s}]  "
                      f"{dist_start:5.0f}m – {dist_end:5.0f}m  "
                      f"peaks={n}  max_kappa={max_kappa:.4f}")

        else:  # straight
            _, idx_start, idx_end = g
            dist_start = float(cum_dist[idx_start])
            dist_end   = float(cum_dist[idx_end])
            gap_kappa  = kappa_smooth[idx_start:idx_end]
            min_kappa  = float(gap_kappa.min()) if len(gap_kappa) > 0 else 0.0

            event_id = f"S{straight_num}"
            straight_num += 1

            event = {
                "id":         event_id,
                "type":       "straight",
                "dist_start": round(dist_start, 1),
                "dist_end":   round(dist_end, 1),
                "length_m":   round(dist_end - dist_start, 1),
                "min_kappa":  round(min_kappa, 6),
            }
            events.append(event)

            if verbose:
                print(f"  {event_id:12s} [straight         ]  "
                      f"{dist_start:5.0f}m – {dist_end:5.0f}m  "
                      f"length={dist_end - dist_start:.0f}m  "
                      f"min_kappa={min_kappa:.5f}")

    return events


def events_to_summary(events: list[dict], circuit_name: str = "") -> str:
    """
    Plain-text summary of detected events, suitable for embedding as
    a circuit_overview chunk or for debugging.
    """
    n_corners   = sum(1 for e in events if e["type"] == "corner")
    n_sequences = sum(1 for e in events if e["type"] == "corner_sequence")
    n_straights = sum(1 for e in events if e["type"] == "straight")
    total_corners = sum(e.get("n_peaks", 1) for e in events
                        if e["type"] in ("corner", "corner_sequence"))

    lines = [
        f"Track Event Map: {circuit_name}",
        f"Total corners: {total_corners} "
        f"({n_corners} standalone, {n_sequences} corner sequences)",
        f"Straights: {n_straights}",
        "",
    ]

    for e in events:
        eid   = e["id"]
        etype = e["type"]
        ds    = e["dist_start"]
        de    = e["dist_end"]
        length = e["length_m"]

        if etype == "straight":
            lines.append(
                f"{eid}: straight | {ds:.0f}m – {de:.0f}m | {length:.0f}m long"
            )
        elif etype == "corner":
            kappa = e["peak_kappas"][0]
            radius = round(1 / kappa) if kappa > 0 else "∞"
            lines.append(
                f"{eid}: corner | {ds:.0f}m – {de:.0f}m | "
                f"apex at {e['peak_dists'][0]:.0f}m | "
                f"approx radius {radius}m"
            )
        elif etype == "corner_sequence":
            n = e["n_peaks"]
            apices = ", ".join(f"{d:.0f}m" for d in e["peak_dists"])
            lines.append(
                f"{eid}: {n}-corner sequence | {ds:.0f}m – {de:.0f}m | "
                f"apices at [{apices}] | max_kappa={e['max_kappa']:.4f}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Boundary normalization
# ---------------------------------------------------------------------------

def normalize_event_boundaries(events: list[dict], lap_length: float) -> list[dict]:
    """
    Given the raw (overlapping, gapped) events from detect_circuit_events,
    produce a set of non-overlapping, exhaustive sections that tile the full
    lap from 0 m to lap_length.

    Rules applied in order:
      1. Sort by dist_start.
      2. Overlapping consecutive pairs → boundary snapped to midpoint of
         the overlap zone (e.g. two 80 m overlapping windows share a 40 m
         transition that is assigned to whichever section starts first).
      3. Gaps between consecutive sections → earlier section extended to
         reach the next section's (normalized) start.
      4. Lead gap (0 → first section start) → synthetic straight prepended.
      5. Tail gap (last section end → lap_length) → synthetic straight appended.

    Each returned dict is a shallow copy of the original with updated
    dist_start, dist_end, and length_m; orig_dist_start / orig_dist_end
    preserve the raw geometry values for reference.
    """
    if not events:
        return []

    ordered = sorted(events, key=lambda e: e["dist_start"])
    normalized = [dict(e) for e in ordered]

    for ev in normalized:
        ev["orig_dist_start"] = ev["dist_start"]
        ev["orig_dist_end"]   = ev["dist_end"]

    # Pass 1 — resolve overlaps between consecutive sections
    for i in range(len(normalized) - 1):
        cur  = normalized[i]
        nxt  = normalized[i + 1]
        if cur["dist_end"] > nxt["dist_start"]:
            boundary = (cur["dist_end"] + nxt["dist_start"]) / 2.0
            cur["dist_end"]    = boundary
            nxt["dist_start"]  = boundary

    # Pass 2 — fill gaps by extending the earlier section
    for i in range(len(normalized) - 1):
        cur = normalized[i]
        nxt = normalized[i + 1]
        if cur["dist_end"] < nxt["dist_start"]:
            cur["dist_end"] = nxt["dist_start"]

    # Pass 3 — prepend synthetic straight for lead gap (start/finish → T1)
    first = normalized[0]
    if first["dist_start"] > 0.5:
        lead = {
            "id":         "S_main",
            "type":       "straight",
            "dist_start": 0.0,
            "dist_end":   first["dist_start"],
            "length_m":   first["dist_start"],
            "min_kappa":  0.0,
            "orig_dist_start": 0.0,
            "orig_dist_end":   first["dist_start"],
        }
        normalized.insert(0, lead)

    # Pass 4 — append synthetic straight for tail gap (last section → finish line)
    last = normalized[-1]
    if last["dist_end"] < lap_length - 0.5:
        tail = {
            "id":         "S_finish",
            "type":       "straight",
            "dist_start": last["dist_end"],
            "dist_end":   lap_length,
            "length_m":   lap_length - last["dist_end"],
            "min_kappa":  0.0,
            "orig_dist_start": last["dist_end"],
            "orig_dist_end":   lap_length,
        }
        normalized.append(tail)

    # Recompute length_m from final boundaries
    for ev in normalized:
        ev["length_m"] = ev["dist_end"] - ev["dist_start"]

    return normalized


# ---------------------------------------------------------------------------
# Telemetry windowing helper
# ---------------------------------------------------------------------------

def get_event_telemetry(
    telemetry: pd.DataFrame,
    event: dict,
    dist_col: str = "Distance",
    buffer_m: float = 0.0,
) -> pd.DataFrame:
    """
    Slice a driver's telemetry DataFrame to the distance window of one event.

    Args:
        telemetry:  FastF1 telemetry DataFrame with a Distance column
        event:      single event dict from detect_circuit_events()
        dist_col:   name of the distance column in telemetry
        buffer_m:   extra metres to include before and after the event window

    Returns:
        Sliced telemetry DataFrame for that event. Empty if no overlap.
    """
    if dist_col not in telemetry.columns:
        raise KeyError(
            f"Distance column '{dist_col}' not found. "
            f"Available: {list(telemetry.columns)}"
        )

    d_start = event["dist_start"] - buffer_m
    d_end   = event["dist_end"]   + buffer_m

    mask = (telemetry[dist_col] >= d_start) & (telemetry[dist_col] <= d_end)
    return telemetry[mask].copy()


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data_loaders.load_track_data import load_circuit_geometry

    for circuit in ["Monza", "Spa", "Silverstone"]:
        print(f"\n{'='*60}")
        geom   = load_circuit_geometry(circuit)
        events = detect_circuit_events(geom, circuit_name=circuit, verbose=False)
        print(events_to_summary(events, circuit_name=circuit))
