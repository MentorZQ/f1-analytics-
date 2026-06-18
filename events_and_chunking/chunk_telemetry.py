"""
chunk_telemetry.py
------------------
Produces event-aware telemetry chunks by slicing each driver's FastF1
telemetry against the geometry-based event boundaries from detect_events.py.

Chunk types produced here:
  - telemetry_corner_event   : per driver × corner/corner_sequence
  - telemetry_straight_event : per driver × straight
  - head_to_head_event       : per driver pair × event
  - session_pace_evolution   : per driver, lap time trend across Q1/Q2/Q3

Design principles:
  - All metrics are derived from the telemetry window for that event only,
    not the full lap — so "entry speed into T1" means speed at the start
    of T1's distance window, not some global value.
  - Speed minima are recorded unconditionally. Whether a 285 km/h minimum
    through Eau Rouge is "significant" is for the LLM to reason about.
  - Braking distance is measured from first brake application (Brake=True)
    to the speed minimum within the event window.
  - DRS detection uses the DRS channel value > 8 (FastF1 convention for
    "DRS open"). If DRS column is absent, it's noted as unavailable.
  - Head-to-head event chunks align both drivers at the same spatial window
    and directly compare their metrics — not derived from two separate chunks.

Dependencies:
  - detect_events.py    : event boundary dicts
  - chunk_types.py      : Chunk dataclass and helpers
  - FastF1 telemetry    : from load_qualifying_session()["telemetry"]

Usage:
    from chunking.detect_events import detect_circuit_events
    from chunking.chunk_telemetry import build_telemetry_chunks

    events   = detect_circuit_events(geometry_df, circuit_name="Monza")
    chunks   = build_telemetry_chunks(fastf1_data, events, session_info)
"""

import numpy as np
import pandas as pd
from typing import Optional
from .chunk_types import Chunk, make_chunk_id, CHUNK_TYPES
from .detect_events import get_event_telemetry, normalize_event_boundaries


# ---------------------------------------------------------------------------
# Low-level telemetry metric extractors
# ---------------------------------------------------------------------------

def _time_at_distance(tel: pd.DataFrame, dist_m: float) -> Optional[float]:
    """
    Linearly interpolate the lap-elapsed time (seconds) at an exact distance
    boundary.  Clamps to the nearest telemetry edge rather than returning None,
    so synthetic lap-start (0 m) and lap-end boundaries always produce a value.

    FastF1 convention: Time column is 0.0 s at the lap start for all drivers,
    so clamping to the first sample at the start edge is correct.
    """
    if tel.empty or "Distance" not in tel.columns or "Time" not in tel.columns:
        return None
    d = tel["Distance"]
    d_min, d_max = float(d.min()), float(d.max())

    if dist_m <= d_min:
        return float(tel.iloc[0]["Time"].total_seconds())
    if dist_m >= d_max:
        return float(tel.iloc[-1]["Time"].total_seconds())

    before = (d <= dist_m).values.nonzero()[0]
    after  = (d >= dist_m).values.nonzero()[0]
    if not len(before) or not len(after):
        return None
    i0, i1 = int(before[-1]), int(after[0])
    if i0 == i1:
        return float(tel.iloc[i0]["Time"].total_seconds())
    d0 = float(tel.iloc[i0]["Distance"])
    d1 = float(tel.iloc[i1]["Distance"])
    t0 = float(tel.iloc[i0]["Time"].total_seconds())
    t1 = float(tel.iloc[i1]["Time"].total_seconds())
    alpha = (dist_m - d0) / (d1 - d0)
    return t0 + alpha * (t1 - t0)


def _entry_speed(tel: pd.DataFrame) -> Optional[float]:
    """Speed at the first telemetry sample in the event window."""
    if tel.empty or "Speed" not in tel.columns:
        return None
    return float(tel["Speed"].iloc[0])


def _exit_speed(tel: pd.DataFrame) -> Optional[float]:
    """Speed at the last telemetry sample in the event window."""
    if tel.empty or "Speed" not in tel.columns:
        return None
    return float(tel["Speed"].iloc[-1])


def _min_speed(tel: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    """
    (min_speed_kmh, dist_at_min_m) — recorded unconditionally.
    For a flat-out corner this will be a high value; for a hairpin, low.
    The LLM decides whether the minimum is meaningful.
    Returns (None, None) if speed data is absent.
    """
    if tel.empty or "Speed" not in tel.columns:
        return None, None
    idx = tel["Speed"].idxmin()
    speed = float(tel.loc[idx, "Speed"])
    dist  = float(tel.loc[idx, "Distance"]) if "Distance" in tel.columns else None
    return speed, dist


def _max_speed(tel: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    """(max_speed_kmh, dist_at_max_m)"""
    if tel.empty or "Speed" not in tel.columns:
        return None, None
    idx = tel["Speed"].idxmax()
    speed = float(tel.loc[idx, "Speed"])
    dist  = float(tel.loc[idx, "Distance"]) if "Distance" in tel.columns else None
    return speed, dist


def _braking_analysis(tel: pd.DataFrame) -> dict:
    """
    Analyse braking on corner entry.

    Returns:
        brake_applied       : bool — was brake ever applied in this window?
        brake_start_dist_m  : distance at which braking starts (first Brake=True)
        brake_distance_m    : distance over which brake is applied on entry
                              (from first application to speed minimum)
        entry_speed_at_brake: speed when braking first begins
    """
    result = {
        "brake_applied":        False,
        "brake_start_dist_m":   None,
        "brake_distance_m":     None,
        "entry_speed_at_brake": None,
    }

    if tel.empty or "Brake" not in tel.columns or "Distance" not in tel.columns:
        return result

    brake = tel["Brake"].astype(bool)
    if not brake.any():
        return result

    # First brake application
    first_brake_idx  = brake.idxmax()
    brake_start_dist = float(tel.loc[first_brake_idx, "Distance"])
    brake_start_speed = float(tel.loc[first_brake_idx, "Speed"]) if "Speed" in tel.columns else None

    # Speed minimum after braking starts
    tel_after_brake = tel[tel.index >= first_brake_idx]
    if tel_after_brake.empty or "Speed" not in tel_after_brake.columns:
        return result

    min_idx      = tel_after_brake["Speed"].idxmin()
    min_dist     = float(tel_after_brake.loc[min_idx, "Distance"])
    brake_dist   = max(0.0, min_dist - brake_start_dist)

    result.update({
        "brake_applied":        True,
        "brake_start_dist_m":   round(brake_start_dist, 1),
        "brake_distance_m":     round(brake_dist, 1),
        "entry_speed_at_brake": round(brake_start_speed, 1) if brake_start_speed else None,
    })
    return result


def _throttle_analysis(tel: pd.DataFrame) -> dict:
    """
    Analyse throttle application, primarily for corner exit and straights.

    Returns:
        full_throttle_pct    : % of window at full throttle (≥98%)
        throttle_mean        : average throttle across window
        throttle_pickup_dist : distance from window start at which
                               throttle first reaches ≥98% (corner exit point)
    """
    result = {
        "full_throttle_pct":     None,
        "throttle_mean":         None,
        "throttle_pickup_dist":  None,
    }

    if tel.empty or "Throttle" not in tel.columns:
        return result

    t = tel["Throttle"].dropna()
    if t.empty:
        return result

    full_throttle_mask = t >= 98
    result["full_throttle_pct"] = round(float(full_throttle_mask.mean() * 100), 1)
    result["throttle_mean"]     = round(float(t.mean()), 1)

    # First full throttle application
    if full_throttle_mask.any() and "Distance" in tel.columns:
        first_ft_idx  = full_throttle_mask.idxmax()
        window_start  = float(tel["Distance"].iloc[0])
        ft_dist       = float(tel.loc[first_ft_idx, "Distance"])
        result["throttle_pickup_dist"] = round(ft_dist - window_start, 1)

    return result


def _drs_analysis(tel: pd.DataFrame) -> dict:
    """
    Analyse DRS usage within the event window.

    FastF1 DRS channel: value > 8 means DRS is open.

    Returns:
        drs_available : bool (DRS column present)
        drs_active    : bool (DRS was open at any point)
        drs_dist_m    : distance over which DRS was open
    """
    result = {"drs_available": False, "drs_active": False, "drs_dist_m": None}

    if tel.empty or "DRS" not in tel.columns:
        return result

    result["drs_available"] = True
    drs_open = tel["DRS"] > 8

    if not drs_open.any():
        return result

    result["drs_active"] = True

    # Approximate DRS distance from point spacing
    if "Distance" in tel.columns:
        drs_tel = tel[drs_open]
        if len(drs_tel) > 1:
            drs_dist = float(drs_tel["Distance"].max() - drs_tel["Distance"].min())
            result["drs_dist_m"] = round(drs_dist, 1)

    return result


def _gear_analysis(tel: pd.DataFrame) -> dict:
    """Min, max, and mode gear used in the event window."""
    result = {"gear_min": None, "gear_max": None, "gear_mode": None}
    if tel.empty or "nGear" not in tel.columns:
        return result
    g = tel["nGear"].dropna()
    if g.empty:
        return result
    result["gear_min"]  = int(g.min())
    result["gear_max"]  = int(g.max())
    result["gear_mode"] = int(g.mode().iloc[0])
    return result


def _apex_speeds(tel: pd.DataFrame, peak_dists: list[float], window_m: float = 60.0) -> list[dict]:
    """
    For each geometry-defined apex distance, find the actual speed minimum
    in a ±window_m neighbourhood in the telemetry.

    Returns a list of dicts: [{apex_dist, min_speed_kmh, gear_at_apex}]
    Recorded unconditionally — a flat apex will have a high speed value.
    """
    if tel.empty or "Speed" not in tel.columns or "Distance" not in tel.columns:
        return [{"apex_dist": d, "min_speed_kmh": None, "gear_at_apex": None}
                for d in peak_dists]

    apices = []
    for dist in peak_dists:
        window = tel[
            (tel["Distance"] >= dist - window_m) &
            (tel["Distance"] <= dist + window_m)
        ]
        if window.empty:
            apices.append({"apex_dist": dist, "min_speed_kmh": None, "gear_at_apex": None})
            continue

        min_idx   = window["Speed"].idxmin()
        min_speed = float(window.loc[min_idx, "Speed"])
        gear      = int(window.loc[min_idx, "nGear"]) if "nGear" in window.columns else None
        apices.append({
            "apex_dist":    round(dist, 1),
            "min_speed_kmh": round(min_speed, 1),
            "gear_at_apex":  gear,
        })

    return apices


# ---------------------------------------------------------------------------
# Corner event chunker
# ---------------------------------------------------------------------------

def chunk_corner_event(
    driver_row: pd.Series,
    tel: pd.DataFrame,
    event: dict,
    session_info: dict,
) -> Optional[Chunk]:
    """
    One chunk for one driver at one corner or corner_sequence.

    Covers: entry speed, apex speed(s), exit speed, braking analysis,
    throttle pickup, gear range, DRS status on approach.
    """
    if tel.empty:
        return None

    circuit = session_info["circuit"]
    year    = session_info["year"]
    driver  = driver_row["Driver"]
    team    = driver_row.get("Team", "")
    pos     = int(driver_row["QualifyingPosition"])
    eid     = event["id"]
    etype   = event["type"]

    entry_spd      = _entry_speed(tel)
    exit_spd       = _exit_speed(tel)
    min_spd, _     = _min_speed(tel)
    braking        = _braking_analysis(tel)
    throttle       = _throttle_analysis(tel)
    drs            = _drs_analysis(tel)
    gears          = _gear_analysis(tel)
    apices         = _apex_speeds(tel, event.get("peak_dists", []))

    event_label = "corner sequence" if etype == "corner_sequence" else "corner"

    lines = [
        f"Corner Event: {eid} ({event_label}) — {driver} ({team})",
        f"Event: {circuit} {year} Qualifying | Qualifying position: P{pos}",
        f"Distance window: {event['dist_start']:.0f}m – {event['dist_end']:.0f}m "
        f"({event['length_m']:.0f}m window)",
    ]

    # Speed profile
    if entry_spd:
        lines.append(f"Entry speed: {entry_spd:.1f} km/h")

    # Apex speeds — one line per apex
    if apices:
        if len(apices) == 1:
            a = apices[0]
            spd_str = f"{a['min_speed_kmh']:.1f} km/h" if a["min_speed_kmh"] else "N/A"
            gear_str = f"gear {a['gear_at_apex']}" if a["gear_at_apex"] else ""
            lines.append(f"Apex speed: {spd_str}" + (f" ({gear_str})" if gear_str else ""))
        else:
            lines.append("Apex speeds (per corner in sequence):")
            for i, a in enumerate(apices):
                spd_str  = f"{a['min_speed_kmh']:.1f} km/h" if a["min_speed_kmh"] else "N/A"
                gear_str = f"gear {a['gear_at_apex']}" if a["gear_at_apex"] else ""
                lines.append(
                    f"  Apex {i+1} at {a['apex_dist']:.0f}m: {spd_str}"
                    + (f" ({gear_str})" if gear_str else "")
                )

    if exit_spd:
        lines.append(f"Exit speed: {exit_spd:.1f} km/h")

    # Braking
    if braking["brake_applied"]:
        lines.append(
            f"Braking: starts at {braking['brake_start_dist_m']:.0f}m "
            f"({braking['entry_speed_at_brake']:.1f} km/h) | "
            f"braking distance to apex: {braking['brake_distance_m']:.0f}m"
        )
    else:
        lines.append("Braking: no braking detected in this window (flat or lift-only)")

    # Throttle pickup
    if throttle["full_throttle_pct"] is not None:
        lines.append(
            f"Throttle: {throttle['full_throttle_pct']:.0f}% of window at full throttle"
            + (f" | pickup {throttle['throttle_pickup_dist']:.0f}m into window"
               if throttle["throttle_pickup_dist"] is not None else "")
        )

    # Gears
    if gears["gear_min"] is not None:
        lines.append(
            f"Gears: {gears['gear_min']}–{gears['gear_max']} "
            f"(most common: {gears['gear_mode']})"
        )

    # DRS
    if drs["drs_available"]:
        if drs["drs_active"]:
            dist_str = f" over {drs['drs_dist_m']:.0f}m" if drs["drs_dist_m"] else ""
            lines.append(f"DRS: active{dist_str}")
        else:
            lines.append("DRS: not active in this window")

    return Chunk(
        chunk_id=make_chunk_id("telemetry_corner_event", circuit, year, driver, eid),
        chunk_type="telemetry_corner_event",
        text="\n".join(lines),
        metadata={
            "circuit":             circuit,
            "year":                year,
            "driver":              driver,
            "team":                team,
            "event_id":            eid,
            "event_type":          etype,
            "qualifying_position": pos,
            "entry_speed_kmh":     round(entry_spd, 1) if entry_spd else None,
            "min_speed_kmh":       round(min_spd, 1) if min_spd else None,
            "exit_speed_kmh":      round(exit_spd, 1) if exit_spd else None,
            "brake_distance_m":    braking["brake_distance_m"],
            "full_throttle_pct":   throttle["full_throttle_pct"],
            "gear_min":            gears["gear_min"],
            "gear_max":            gears["gear_max"],
            "n_apices":            len(apices),
        },
        source="fastf1",
    )


# ---------------------------------------------------------------------------
# Straight event chunker
# ---------------------------------------------------------------------------

def chunk_straight_event(
    driver_row: pd.Series,
    tel: pd.DataFrame,
    event: dict,
    session_info: dict,
) -> Optional[Chunk]:
    """
    One chunk for one driver on one straight.

    Covers: entry speed, max speed (top speed + where it occurs),
    exit speed (braking point), DRS usage, gear at max speed.
    """
    if tel.empty:
        return None

    circuit = session_info["circuit"]
    year    = session_info["year"]
    driver  = driver_row["Driver"]
    team    = driver_row.get("Team", "")
    pos     = int(driver_row["QualifyingPosition"])
    eid     = event["id"]

    entry_spd          = _entry_speed(tel)
    exit_spd           = _exit_speed(tel)
    max_spd, max_dist  = _max_speed(tel)
    braking            = _braking_analysis(tel)
    throttle           = _throttle_analysis(tel)
    drs                = _drs_analysis(tel)
    gears              = _gear_analysis(tel)

    lines = [
        f"Straight Event: {eid} — {driver} ({team})",
        f"Event: {circuit} {year} Qualifying | Qualifying position: P{pos}",
        f"Distance window: {event['dist_start']:.0f}m – {event['dist_end']:.0f}m "
        f"({event['length_m']:.0f}m straight)",
    ]

    if entry_spd:
        lines.append(f"Entry speed (from preceding corner): {entry_spd:.1f} km/h")

    if max_spd:
        max_dist_str = f" at {max_dist:.0f}m" if max_dist else ""
        lines.append(f"Top speed: {max_spd:.1f} km/h{max_dist_str}")

    if exit_spd:
        lines.append(f"Exit speed (into next braking zone): {exit_spd:.1f} km/h")

    # Speed gain across the straight
    if entry_spd and max_spd:
        gain = max_spd - entry_spd
        lines.append(f"Speed gain entry→top: +{gain:.1f} km/h")

    # Braking at end of straight
    if braking["brake_applied"]:
        lines.append(
            f"Braking into next corner: begins at {braking['brake_start_dist_m']:.0f}m "
            f"({braking['entry_speed_at_brake']:.1f} km/h) | "
            f"braking distance: {braking['brake_distance_m']:.0f}m"
        )
    else:
        lines.append("Braking: no braking detected within this straight window")

    if throttle["full_throttle_pct"] is not None:
        lines.append(
            f"Throttle: {throttle['full_throttle_pct']:.0f}% of straight at full throttle"
        )

    if gears["gear_max"] is not None:
        lines.append(
            f"Top gear on straight: {gears['gear_max']} "
            f"(range used: {gears['gear_min']}–{gears['gear_max']})"
        )

    if drs["drs_available"]:
        if drs["drs_active"]:
            dist_str = f" | active for {drs['drs_dist_m']:.0f}m" if drs["drs_dist_m"] else ""
            lines.append(f"DRS: OPEN on this straight{dist_str}")
        else:
            lines.append("DRS: closed on this straight")

    return Chunk(
        chunk_id=make_chunk_id("telemetry_straight_event", circuit, year, driver, eid),
        chunk_type="telemetry_straight_event",
        text="\n".join(lines),
        metadata={
            "circuit":             circuit,
            "year":                year,
            "driver":              driver,
            "team":                team,
            "event_id":            eid,
            "qualifying_position": pos,
            "entry_speed_kmh":     round(entry_spd, 1) if entry_spd else None,
            "max_speed_kmh":       round(max_spd, 1) if max_spd else None,
            "exit_speed_kmh":      round(exit_spd, 1) if exit_spd else None,
            "drs_active":          drs["drs_active"],
            "drs_dist_m":          drs["drs_dist_m"],
            "gear_max":            gears["gear_max"],
            "full_throttle_pct":   throttle["full_throttle_pct"],
        },
        source="fastf1",
    )


# ---------------------------------------------------------------------------
# Head-to-head event chunker
# ---------------------------------------------------------------------------

def chunk_head_to_head_event(
    driver_a_row: pd.Series,
    driver_b_row: pd.Series,
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
    event: dict,
    session_info: dict,
    time_a_s: Optional[float] = None,
    time_b_s: Optional[float] = None,
) -> Optional[Chunk]:
    """
    One chunk comparing two drivers at the same track event.

    Directly compares entry/apex/exit speeds, braking distances,
    throttle pickup, and top speed — all at the same spatial window.
    This is the primary chunk for granular diagnostic questions.
    """
    if tel_a.empty or tel_b.empty:
        return None

    circuit = session_info["circuit"]
    year    = session_info["year"]
    drv_a   = driver_a_row["Driver"]
    drv_b   = driver_b_row["Driver"]
    team_a  = driver_a_row.get("Team", "")
    team_b  = driver_b_row.get("Team", "")
    pos_a   = int(driver_a_row["QualifyingPosition"])
    pos_b   = int(driver_b_row["QualifyingPosition"])
    eid     = event["id"]
    etype   = event["type"]
    is_corner = etype in ("corner", "corner_sequence")

    # Extract metrics for both drivers
    # time_a_s / time_b_s are pre-computed via boundary interpolation in build_telemetry_chunks
    time_a  = time_a_s
    time_b  = time_b_s
    entry_a = _entry_speed(tel_a);  entry_b = _entry_speed(tel_b)
    exit_a  = _exit_speed(tel_a);   exit_b  = _exit_speed(tel_b)

    if is_corner:
        min_a, _  = _min_speed(tel_a)
        min_b, _  = _min_speed(tel_b)
        brake_a   = _braking_analysis(tel_a)
        brake_b   = _braking_analysis(tel_b)
        throttle_a = _throttle_analysis(tel_a)
        throttle_b = _throttle_analysis(tel_b)
        apex_a    = _apex_speeds(tel_a, event.get("peak_dists", []))
        apex_b    = _apex_speeds(tel_b, event.get("peak_dists", []))
    else:
        max_a, _  = _max_speed(tel_a)
        max_b, _  = _max_speed(tel_b)
        drs_a     = _drs_analysis(tel_a)
        drs_b     = _drs_analysis(tel_b)
        brake_a   = _braking_analysis(tel_a)
        brake_b   = _braking_analysis(tel_b)

    event_label = "corner sequence" if etype == "corner_sequence" else etype

    lines = [
        f"Head-to-Head at {eid} ({event_label}): {drv_a} vs {drv_b}",
        f"Event: {circuit} {year} Qualifying",
        f"{drv_a} P{pos_a} ({team_a}) | {drv_b} P{pos_b} ({team_b})",
        f"Distance window: {event['dist_start']:.0f}m – {event['dist_end']:.0f}m",
        "",
    ]

    def _delta(a, b, label, unit="km/h", higher_is_better=True):
        """Format a metric comparison line."""
        if a is None or b is None:
            return f"  {label}: {drv_a} N/A | {drv_b} N/A"
        delta = a - b
        if abs(delta) < 0.05:
            advantage = "equal"
        elif (delta > 0) == higher_is_better:
            advantage = f"{drv_a} +{abs(delta):.1f}{unit}"
        else:
            advantage = f"{drv_b} +{abs(delta):.1f}{unit}"
        return f"  {label}: {drv_a} {a:.1f}{unit} | {drv_b} {b:.1f}{unit} | advantage: {advantage}"

    # Time in section — the primary lap-time contribution from this event
    if time_a is not None and time_b is not None:
        time_delta = time_a - time_b  # positive = drv_a slower
        abs_dt     = abs(time_delta)
        if abs_dt < 0.001:
            time_line = f"  Time in section: {drv_a} {time_a:.3f}s | {drv_b} {time_b:.3f}s | equal"
        else:
            faster = drv_b if time_delta > 0 else drv_a
            time_line = (
                f"  Time in section: {drv_a} {time_a:.3f}s | {drv_b} {time_b:.3f}s "
                f"| {faster} faster by {abs_dt:.3f}s"
            )
        lines.append(time_line)

    # Entry speed
    if entry_a and entry_b:
        lines.append(_delta(entry_a, entry_b, "Entry speed"))

    if is_corner:
        # Apex speeds
        if apex_a and apex_b:
            lines.append("Apex speeds:")
            for i, (aa, ab) in enumerate(zip(apex_a, apex_b)):
                sa = aa["min_speed_kmh"]; sb = ab["min_speed_kmh"]
                label = f"    Apex {i+1} at ~{aa['apex_dist']:.0f}m"
                if sa and sb:
                    delta = sa - sb
                    adv   = f"{drv_a} +{abs(delta):.1f}" if delta > 0.5 else \
                            f"{drv_b} +{abs(delta):.1f}" if delta < -0.5 else "equal"
                    lines.append(f"  {label}: {drv_a} {sa:.1f} km/h | {drv_b} {sb:.1f} km/h | advantage: {adv}")

        # Braking comparison
        lines.append("Braking (corner entry):")
        for driver, brake in [(drv_a, brake_a), (drv_b, brake_b)]:
            if brake["brake_applied"]:
                lines.append(
                    f"  {driver}: brakes at {brake['brake_start_dist_m']:.0f}m "
                    f"from {brake['entry_speed_at_brake']:.1f} km/h | "
                    f"braking distance {brake['brake_distance_m']:.0f}m"
                )
            else:
                lines.append(f"  {driver}: no braking detected")

        # Braking distance delta
        bd_a = brake_a["brake_distance_m"]; bd_b = brake_b["brake_distance_m"]
        if bd_a and bd_b:
            delta_bd = bd_a - bd_b
            if abs(delta_bd) > 2:
                later = drv_a if delta_bd < 0 else drv_b
                lines.append(
                    f"  → {later} braked {abs(delta_bd):.0f}m later "
                    f"({'more aggressive' if abs(delta_bd) > 10 else 'slightly later'})"
                )

        # Braking point delta (where they first applied brakes)
        bp_a = brake_a["brake_start_dist_m"]; bp_b = brake_b["brake_start_dist_m"]
        if bp_a and bp_b:
            delta_bp = bp_a - bp_b
            if abs(delta_bp) > 2:
                later  = drv_a if delta_bp > 0 else drv_b
                earlier = drv_b if delta_bp > 0 else drv_a
                lines.append(
                    f"  → {later} brakes {abs(delta_bp):.0f}m later than {earlier} "
                    f"(later braking point = more aggressive entry)"
                )

        # Throttle pickup
        tpu_a = throttle_a.get("throttle_pickup_dist") if is_corner else None
        tpu_b = throttle_b.get("throttle_pickup_dist") if is_corner else None
        if tpu_a is not None and tpu_b is not None:
            delta_tp = tpu_a - tpu_b
            if abs(delta_tp) > 2:
                earlier = drv_a if delta_tp < 0 else drv_b
                lines.append(
                    f"Throttle pickup: {drv_a} at {tpu_a:.0f}m | {drv_b} at {tpu_b:.0f}m "
                    f"| {earlier} gets on throttle earlier (better traction/exit)"
                )

    else:
        # Straight-specific metrics
        if max_a and max_b:
            lines.append(_delta(max_a, max_b, "Top speed"))

        # DRS
        lines.append(f"DRS: {drv_a} {'open' if drs_a['drs_active'] else 'closed'} | "
                     f"{drv_b} {'open' if drs_b['drs_active'] else 'closed'}")

        # End-of-straight braking
        lines.append("Braking point into next corner:")
        for driver, brake in [(drv_a, brake_a), (drv_b, brake_b)]:
            if brake["brake_applied"]:
                lines.append(
                    f"  {driver}: {brake['brake_start_dist_m']:.0f}m "
                    f"at {brake['entry_speed_at_brake']:.1f} km/h"
                )
            else:
                lines.append(f"  {driver}: no braking within window")

    # Exit speed
    if exit_a and exit_b:
        lines.append(_delta(exit_a, exit_b, "Exit speed"))

    return Chunk(
        chunk_id=make_chunk_id("head_to_head_event", circuit, year, drv_a, drv_b, eid),
        chunk_type="head_to_head_event",
        text="\n".join(lines),
        metadata={
            "circuit":      circuit,
            "year":         year,
            "driver_a":     drv_a,
            "driver_b":     drv_b,
            "event_id":     eid,
            "event_type":   etype,
            "pos_a":        pos_a,
            "pos_b":        pos_b,
            "time_delta_s":      round(time_a - time_b, 3) if time_a is not None and time_b is not None else None,
            "entry_speed_delta": round(entry_a - entry_b, 1) if entry_a and entry_b else None,
            "exit_speed_delta":  round(exit_a  - exit_b,  1) if exit_a  and exit_b  else None,
        },
        source="fastf1",
    )


# ---------------------------------------------------------------------------
# Pace evolution chunker
# ---------------------------------------------------------------------------

def chunk_session_pace_evolution(
    driver: str,
    all_laps: pd.DataFrame,
    session_info: dict,
) -> Optional[Chunk]:
    """
    Captures a driver's lap time progression across Q1, Q2, Q3.

    Lets the LLM reason about track evolution:
    - Was their lap set early when the track was cold?
    - Did they improve from Q1 to Q3?
    - How does their Q1 pace compare to their Q3 pace?
    """
    circuit = session_info["circuit"]
    year    = session_info["year"]

    driver_laps = all_laps[all_laps["Driver"] == driver].copy()
    if driver_laps.empty:
        return None

    driver_laps = driver_laps.dropna(subset=["LapTime"]).sort_values("LapStartTime")

    # Identify Q1/Q2/Q3 by lap number ranges (approximate — FastF1 labels them)
    # FastF1 may have a 'Session' or 'Stint' column; fall back to time ordering
    sessions_col = None
    for col in ["Session", "TrackStatus"]:
        if col in driver_laps.columns:
            sessions_col = col
            break

    lines = [
        f"Pace Evolution: {driver}",
        f"Event: {circuit} {year} Qualifying",
        f"Total laps recorded: {len(driver_laps)}",
        "",
        "Lap-by-lap times (chronological):",
    ]

    best_time = None
    for i, (_, row) in enumerate(driver_laps.iterrows()):
        lt = row["LapTime"]
        lt_s = lt.total_seconds()
        is_best = best_time is None or lt_s < best_time
        if is_best:
            best_time = lt_s
        mins    = int(lt_s // 60)
        secs    = lt_s % 60
        lt_str  = f"{mins}:{secs:06.3f}"
        pb_flag = " ← personal best" if is_best else ""
        lines.append(f"  Lap {i+1}: {lt_str}{pb_flag}")

    # Overall trend
    if len(driver_laps) >= 2:
        first_lt = driver_laps["LapTime"].iloc[0].total_seconds()
        best_lt  = driver_laps["LapTime"].min().total_seconds()
        improvement = first_lt - best_lt

        lines += [
            "",
            f"First lap: {first_lt:.3f}s",
            f"Best lap:  {best_lt:.3f}s",
            f"Improvement across session: {improvement:.3f}s "
            f"({'improved' if improvement > 0.05 else 'flat' if abs(improvement) < 0.05 else 'degraded'})",
        ]

        if improvement > 0.5:
            lines.append(
                "Note: substantial improvement suggests track evolution "
                "or setup changes between runs — early laps should be "
                "compared to other drivers' early laps, not Q3 benchmarks."
            )

    return Chunk(
        chunk_id=make_chunk_id("session_pace_evolution", circuit, year, driver),
        chunk_type="session_pace_evolution",
        text="\n".join(lines),
        metadata={
            "circuit":          circuit,
            "year":             year,
            "driver":           driver,
            "total_laps":       len(driver_laps),
            "best_lap_seconds": round(best_time, 3) if best_time else None,
        },
        source="fastf1",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_telemetry_chunks(
    fastf1_data: dict,
    events: list[dict],
    session_info: dict,
    head_to_head_top_n: int = 5,
) -> list[Chunk]:
    """
    Main entry point. Produces all telemetry event chunks for a session.

    Args:
        fastf1_data:        output of load_qualifying_session()
        events:             output of detect_circuit_events()
        session_info:       session_info dict (year, circuit, etc.)
        head_to_head_top_n: generate H2H event chunks for top N + all teammates

    Returns:
        [
            *telemetry_corner_event chunks,
            *telemetry_straight_event chunks,
            *head_to_head_event chunks,
            *session_pace_evolution chunks,
        ]
    """
    fastest   = fastf1_data["driver_fastest_laps"]
    telemetry = fastf1_data.get("telemetry", {})
    all_laps  = fastf1_data.get("all_laps", pd.DataFrame())
    chunks    = []

    # Compute lap length from the longest telemetry trace, then normalize
    # event boundaries so sections are non-overlapping and tile the full lap.
    tel_maxes = [
        float(tel["Distance"].max())
        for tel in telemetry.values()
        if tel is not None and not tel.empty and "Distance" in tel.columns
    ]
    lap_length = max(tel_maxes) if tel_maxes else None
    if lap_length is not None:
        events = normalize_event_boundaries(events, lap_length)

    # --- Per-driver event chunks ---
    for _, row in fastest.iterrows():
        driver = row["Driver"]
        tel    = telemetry.get(driver)
        if tel is None or tel.empty or "Distance" not in tel.columns:
            continue

        for event in events:
            event_tel = get_event_telemetry(tel, event)
            if event_tel.empty:
                continue

            if event["type"] in ("corner", "corner_sequence"):
                c = chunk_corner_event(row, event_tel, event, session_info)
            else:
                c = chunk_straight_event(row, event_tel, event, session_info)

            if c:
                chunks.append(c)

    # --- Head-to-head event chunks ---
    # Top N qualifiers (all pairs)
    top_n    = fastest.head(head_to_head_top_n)
    top_rows = [row for _, row in top_n.iterrows()]

    # All teammate pairs (find by team)
    teammate_pairs = []
    if "Team" in fastest.columns:
        for team in fastest["Team"].dropna().unique():
            team_drivers = fastest[fastest["Team"] == team]
            if len(team_drivers) >= 2:
                mates = [row for _, row in team_drivers.iterrows()]
                for i in range(len(mates)):
                    for j in range(i + 1, len(mates)):
                        pair = (mates[i]["Driver"], mates[j]["Driver"])
                        teammate_pairs.append(pair)

    # Build full pair list (top N combinations + teammates), deduplicated.
    # Normalize tuple order (alphabetical) before dedup to avoid (A,B) vs (B,A) collisions.
    seen_pairs = set()
    all_pairs  = []

    for i in range(len(top_rows)):
        for j in range(i + 1, len(top_rows)):
            da, db = top_rows[i]["Driver"], top_rows[j]["Driver"]
            pair = (da, db) if da < db else (db, da)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                all_pairs.append((top_rows[i], top_rows[j]))

    for drv_a, drv_b in teammate_pairs:
        pair = (drv_a, drv_b) if drv_a < drv_b else (drv_b, drv_a)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            row_a = fastest[fastest["Driver"] == drv_a].iloc[0]
            row_b = fastest[fastest["Driver"] == drv_b].iloc[0]
            all_pairs.append((row_a, row_b))

    for row_a, row_b in all_pairs:
        tel_a = telemetry.get(row_a["Driver"])
        tel_b = telemetry.get(row_b["Driver"])
        if tel_a is None or tel_b is None:
            continue
        if "Distance" not in tel_a.columns or "Distance" not in tel_b.columns:
            continue

        for event in events:
            ev_tel_a = get_event_telemetry(tel_a, event)
            ev_tel_b = get_event_telemetry(tel_b, event)
            if ev_tel_a.empty or ev_tel_b.empty:
                continue

            # Compute section time via linear interpolation at exact boundary
            # distances on the full-lap telemetry — not first/last sample.
            t_start_a = _time_at_distance(tel_a, event["dist_start"])
            t_end_a   = _time_at_distance(tel_a, event["dist_end"])
            t_start_b = _time_at_distance(tel_b, event["dist_start"])
            t_end_b   = _time_at_distance(tel_b, event["dist_end"])

            time_a_s = (t_end_a - t_start_a) if t_start_a is not None and t_end_a is not None else None
            time_b_s = (t_end_b - t_start_b) if t_start_b is not None and t_end_b is not None else None

            c = chunk_head_to_head_event(
                row_a, row_b, ev_tel_a, ev_tel_b, event, session_info,
                time_a_s=time_a_s, time_b_s=time_b_s,
            )
            if c:
                chunks.append(c)

    # --- Pace evolution chunks ---
    if not all_laps.empty:
        for driver in fastest["Driver"]:
            c = chunk_session_pace_evolution(driver, all_laps, session_info)
            if c:
                chunks.append(c)

    return chunks
