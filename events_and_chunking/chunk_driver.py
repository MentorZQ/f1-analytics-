"""
chunk_driver.py
---------------
Produces driver-level chunks from FastF1 qualifying data:

  - driver_lap_summary    : one chunk per driver — full lap overview
  - driver_sector         : one chunk per driver per sector (S1/S2/S3)
  - driver_telemetry_zone : one chunk per driver per track zone (braking/speed detail)
  - head_to_head          : one chunk per driver pair (top N only, to avoid combinatorial explosion)

These are the most retrieval-critical chunks — most qualifying questions
("how did VER do?", "where did LEC lose time?", "compare HAM and RUS")
are answered by fetching one of these chunk types.

Input:  output of load_qualifying_session()
Output: list of Chunk objects
"""

import pandas as pd
import numpy as np
from typing import Optional
from .chunk_types import Chunk, make_chunk_id, format_timedelta, format_delta


# ---------------------------------------------------------------------------
# DNQ / did-not-set-time status chunk
# ---------------------------------------------------------------------------

def chunk_driver_dnq_status(dnq_info: dict, session_info: dict) -> Chunk:
    """
    One chunk per driver who participated in qualifying but set no valid lap time.

    This chunk surfaces first in context when the driver is named in a question,
    so the LLM explains what happened rather than saying 'no data found'.
    """
    year    = session_info["year"]
    circuit = session_info["circuit"]
    driver  = dnq_info["driver"]
    name    = dnq_info.get("full_name", driver)
    team    = dnq_info.get("team", "Unknown")
    reason  = dnq_info.get("reason", "did not complete a timed lap")

    lines = [
        f"QUALIFYING STATUS — {driver} ({name}, {team})",
        f"Event: {circuit} {year} Qualifying",
        f"Outcome: DID NOT SET A LAP TIME — eliminated in Q1",
        f"Reason: {reason.capitalize()}.",
        f"{driver} did not advance beyond Q1 and has no lap time, sector splits, or telemetry "
        f"available in this session. Head-to-head comparisons with {driver} are not possible.",
    ]

    return Chunk(
        chunk_id=make_chunk_id("driver_dnq_status", circuit, year, driver),
        chunk_type="driver_dnq_status",
        text="\n".join(lines),
        metadata={
            "circuit":  circuit,
            "year":     year,
            "driver":   driver,
            "team":     team,
            "outcome":  "dnq",
            "reason":   reason,
            "source":   "fastf1",
        },
    )


# ---------------------------------------------------------------------------
# Driver lap summary
# ---------------------------------------------------------------------------

def chunk_driver_lap_summary(
    driver_row: pd.Series,
    session_info: dict,
    pole_time: pd.Timedelta,
) -> Chunk:
    """
    One chunk per driver summarising their best qualifying lap.

    Covers: lap time, gap to pole, sector splits, speed traps,
    tyre compound, qualifying position.

    This is the first chunk retrieved for broad driver questions:
    "How did Leclerc qualify at Monza?"
    """
    year    = session_info["year"]
    circuit = session_info["circuit"]
    driver  = driver_row["Driver"]
    team    = driver_row.get("Team", "Unknown")
    pos     = int(driver_row["QualifyingPosition"])

    lap_time = driver_row["LapTime"]
    gap      = lap_time - pole_time if pos > 1 else pd.Timedelta(0)

    # Identify their strongest and weakest sector
    sectors = {}
    for s in [1, 2, 3]:
        col = f"Sector{s}Time"
        if col in driver_row.index and not pd.isna(driver_row[col]):
            sectors[s] = driver_row[col]

    lines = [
        f"Qualifying Lap Summary: {driver} ({team})",
        f"Event: {circuit} {year} Qualifying",
        f"Qualifying position: P{pos}",
        f"Best lap time: {format_timedelta(lap_time)}",
        f"Gap to pole: {format_delta(gap) if pos > 1 else 'POLE POSITION'}",
    ]

    # Sector times
    if sectors:
        lines.append("Sector times:")
        for s, t in sectors.items():
            lines.append(f"  Sector {s}: {format_timedelta(t)}")

    # Speed traps
    for col, label in [
        ("SpeedI1", "Speed trap 1 (intermediate 1)"),
        ("SpeedI2", "Speed trap 2 (intermediate 2)"),
        ("SpeedFL", "Speed trap (finish line)"),
        ("SpeedST", "Speed trap (speed trap)"),
    ]:
        if col in driver_row.index and not pd.isna(driver_row.get(col)):
            lines.append(f"{label}: {driver_row[col]:.1f} km/h")

    # Tyre
    compound = driver_row.get("Compound", None)
    tyre_life = driver_row.get("TyreLife", None)
    fresh     = driver_row.get("FreshTyre", None)
    if compound and not pd.isna(compound):
        tyre_str = f"Tyre: {compound}"
        if tyre_life and not pd.isna(tyre_life):
            tyre_str += f" ({int(tyre_life)} laps old)"
        if fresh is not None and not pd.isna(fresh):
            tyre_str += f" | {'Fresh set' if fresh else 'Used set'}"
        lines.append(tyre_str)

    return Chunk(
        chunk_id=make_chunk_id("driver_lap_summary", circuit, year, driver),
        chunk_type="driver_lap_summary",
        text="\n".join(lines),
        metadata={
            "circuit":              circuit,
            "year":                 year,
            "driver":               driver,
            "team":                 team,
            "qualifying_position":  pos,
            "lap_time_seconds":     lap_time.total_seconds() if not pd.isna(lap_time) else None,
            "gap_to_pole_seconds":  gap.total_seconds() if pos > 1 else 0.0,
            "compound":             compound if compound and not pd.isna(compound) else None,
        },
        source="fastf1",
    )


# ---------------------------------------------------------------------------
# Per-sector chunks
# ---------------------------------------------------------------------------

def chunk_driver_sector(
    driver_row: pd.Series,
    sector: int,
    session_info: dict,
    all_drivers: pd.DataFrame,
) -> Optional[Chunk]:
    """
    One chunk per driver per sector.

    Includes the driver's sector time, their rank in that sector
    across all drivers, and the gap to the fastest sector time.

    Answers questions like: "Where did Hamilton lose time compared to Norris?"
    """
    col = f"Sector{sector}Time"
    if col not in driver_row.index or pd.isna(driver_row[col]):
        return None

    year    = session_info["year"]
    circuit = session_info["circuit"]
    driver  = driver_row["Driver"]
    team    = driver_row.get("Team", "Unknown")

    sector_time = driver_row[col]

    # Rank in this sector across all drivers
    valid_sectors = all_drivers[col].dropna()
    best_sector   = valid_sectors.min()
    gap_to_best   = sector_time - best_sector
    sector_rank   = int((valid_sectors < sector_time).sum()) + 1

    # Who set the best sector time?
    best_driver_row = all_drivers.loc[all_drivers[col] == best_sector]
    best_driver = best_driver_row["Driver"].iloc[0] if not best_driver_row.empty else "Unknown"

    lines = [
        f"Sector {sector} Analysis: {driver} ({team})",
        f"Event: {circuit} {year} Qualifying",
        f"Qualifying position: P{int(driver_row['QualifyingPosition'])}",
        f"Sector {sector} time: {format_timedelta(sector_time)}",
        f"Sector {sector} rank: P{sector_rank} of {len(valid_sectors)} drivers",
        f"Gap to fastest sector {sector} ({best_driver}): {format_delta(gap_to_best)}",
        f"Fastest sector {sector} in session: {format_timedelta(best_sector)} ({best_driver})",
    ]

    # Contextual note on what sector position means
    if sector_rank == 1:
        lines.append(f"Note: {driver} set the fastest sector {sector} in the entire session.")
    elif sector_rank <= 3:
        lines.append(f"Note: {driver} was in the top 3 for sector {sector}.")
    elif gap_to_best.total_seconds() > 0.5:
        lines.append(
            f"Note: {driver} lost over 0.5s to the sector {sector} leader — "
            f"a significant gap likely reflecting braking, traction, or track position issues."
        )

    return Chunk(
        chunk_id=make_chunk_id("driver_sector", circuit, year, driver, sector),
        chunk_type="driver_sector",
        text="\n".join(lines),
        metadata={
            "circuit":             circuit,
            "year":                year,
            "driver":              driver,
            "team":                team,
            "sector":              sector,
            "sector_time_seconds": sector_time.total_seconds(),
            "sector_rank":         sector_rank,
            "gap_to_best_seconds": gap_to_best.total_seconds(),
            "qualifying_position": int(driver_row["QualifyingPosition"]),
        },
        source="fastf1",
    )


# ---------------------------------------------------------------------------
# Telemetry zone chunks
# ---------------------------------------------------------------------------

def _describe_telemetry_zone(zone_df: pd.DataFrame, zone_idx: int, n_zones: int) -> dict:
    """
    Compute summary statistics for one distance-based zone of telemetry.
    Returns a dict of named metrics.
    """
    stats = {"zone_index": zone_idx, "zone_of": n_zones}

    if "Speed" in zone_df.columns:
        s = zone_df["Speed"].dropna()
        stats["speed_mean"]  = round(s.mean(), 1)
        stats["speed_max"]   = round(s.max(), 1)
        stats["speed_min"]   = round(s.min(), 1)

    if "Throttle" in zone_df.columns:
        t = zone_df["Throttle"].dropna()
        stats["throttle_mean"]      = round(t.mean(), 1)
        stats["full_throttle_pct"]  = round((t >= 98).sum() / len(t) * 100, 1) if len(t) > 0 else 0

    if "Brake" in zone_df.columns:
        b = zone_df["Brake"].dropna()
        # Brake can be bool or 0/1
        brake_bool = b.astype(bool) if b.dtype != bool else b
        stats["braking_pct"] = round(brake_bool.sum() / len(b) * 100, 1) if len(b) > 0 else 0

    if "nGear" in zone_df.columns:
        g = zone_df["nGear"].dropna()
        stats["gear_mean"] = round(g.mean(), 1)
        stats["gear_min"]  = int(g.min())
        stats["gear_max"]  = int(g.max())

    if "DRS" in zone_df.columns:
        d = zone_df["DRS"].dropna()
        stats["drs_active_pct"] = round((d > 8).sum() / len(d) * 100, 1) if len(d) > 0 else 0

    return stats


def chunk_driver_telemetry_zones(
    driver_row: pd.Series,
    telemetry: pd.DataFrame,
    session_info: dict,
    n_zones: int = 3,
) -> list[Chunk]:
    """
    Divide the driver's pole lap telemetry into n_zones distance-based segments
    and produce one chunk per zone.

    n_zones=3 maps roughly to sectors (though not exactly — FastF1 telemetry
    doesn't include sector boundary distances natively).

    Each chunk describes: speed profile, braking %, full throttle %, gear range, DRS usage.

    Answers questions like: "How was Verstappen's braking in the first sector?"
    or "Did Hamilton use DRS in the final zone?"
    """
    year    = session_info["year"]
    circuit = session_info["circuit"]
    driver  = driver_row["Driver"]

    if telemetry is None or telemetry.empty:
        return []

    # Use Distance column if available, otherwise fall back to index split
    if "Distance" in telemetry.columns:
        total_dist = telemetry["Distance"].max()
        zone_edges = np.linspace(0, total_dist, n_zones + 1)
    else:
        # Split by index
        zone_size  = len(telemetry) // n_zones
        zone_edges = None

    chunks = []
    for zone_idx in range(n_zones):
        if zone_edges is not None:
            zone_df = telemetry[
                (telemetry["Distance"] >= zone_edges[zone_idx]) &
                (telemetry["Distance"] <  zone_edges[zone_idx + 1])
            ]
            dist_start = zone_edges[zone_idx]
            dist_end   = zone_edges[zone_idx + 1]
            dist_str   = f"{dist_start:.0f}m – {dist_end:.0f}m of {total_dist:.0f}m"
        else:
            start   = zone_idx * zone_size
            end     = start + zone_size if zone_idx < n_zones - 1 else len(telemetry)
            zone_df = telemetry.iloc[start:end]
            dist_str = f"zone {zone_idx + 1} of {n_zones} (index-based split)"

        if zone_df.empty:
            continue

        stats = _describe_telemetry_zone(zone_df, zone_idx, n_zones)

        lines = [
            f"Telemetry Zone {zone_idx + 1} of {n_zones}: {driver}",
            f"Event: {circuit} {year} Qualifying — fastest lap",
            f"Distance covered: {dist_str}",
        ]

        if "speed_mean" in stats:
            lines.append(
                f"Speed: avg {stats['speed_mean']} km/h | "
                f"max {stats['speed_max']} km/h | "
                f"min {stats['speed_min']} km/h"
            )

        if "throttle_mean" in stats:
            lines.append(
                f"Throttle: {stats['throttle_mean']:.0f}% average | "
                f"{stats['full_throttle_pct']:.0f}% of zone at full throttle (≥98%)"
            )

        if "braking_pct" in stats:
            lines.append(f"Braking: active {stats['braking_pct']:.0f}% of zone")

        if "gear_mean" in stats:
            lines.append(
                f"Gears: avg {stats['gear_mean']:.1f} | "
                f"range {stats['gear_min']}–{stats['gear_max']}"
            )

        if "drs_active_pct" in stats:
            lines.append(f"DRS: active {stats['drs_active_pct']:.0f}% of zone")

        # Characterise the zone
        if "full_throttle_pct" in stats and "braking_pct" in stats:
            if stats["full_throttle_pct"] > 70:
                char = "high-speed, predominantly full-throttle zone"
            elif stats["braking_pct"] > 30:
                char = "heavy braking zone with significant deceleration"
            elif stats["full_throttle_pct"] > 40:
                char = "mixed zone — alternating throttle and braking"
            else:
                char = "technical zone — low throttle, careful application"
            lines.append(f"Zone character: {char}")

        chunks.append(Chunk(
            chunk_id=make_chunk_id("driver_telemetry_zone", circuit, year, driver, zone_idx),
            chunk_type="driver_telemetry_zone",
            text="\n".join(lines),
            metadata={
                "circuit":            circuit,
                "year":               year,
                "driver":             driver,
                "zone_index":         zone_idx,
                "zone_of":            n_zones,
                "speed_max":          stats.get("speed_max"),
                "braking_pct":        stats.get("braking_pct"),
                "full_throttle_pct":  stats.get("full_throttle_pct"),
                "qualifying_position": int(driver_row["QualifyingPosition"]),
            },
            source="fastf1",
        ))

    return chunks


# ---------------------------------------------------------------------------
# Head-to-head chunks
# ---------------------------------------------------------------------------

def chunk_head_to_head(
    driver_a: pd.Series,
    driver_b: pd.Series,
    session_info: dict,
) -> Chunk:
    """
    One chunk comparing two drivers across lap time and all three sectors.

    Only produce these for meaningful pairs (top N, teammates, etc.) to
    avoid O(n²) chunks across 20 drivers.

    Answers: "Compare Verstappen and Norris at Monza qualifying."
    """
    year    = session_info["year"]
    circuit = session_info["circuit"]
    drv_a   = driver_a["Driver"]
    drv_b   = driver_b["Driver"]
    team_a  = driver_a.get("Team", "")
    team_b  = driver_b.get("Team", "")

    lap_a = driver_a["LapTime"]
    lap_b = driver_b["LapTime"]
    lap_delta = lap_a - lap_b  # positive = A is slower

    lines = [
        f"Head-to-Head Comparison: {drv_a} vs {drv_b}",
        f"Event: {circuit} {year} Qualifying",
        f"",
        f"{drv_a} ({team_a}): P{int(driver_a['QualifyingPosition'])} — {format_timedelta(lap_a)}",
        f"{drv_b} ({team_b}): P{int(driver_b['QualifyingPosition'])} — {format_timedelta(lap_b)}",
        f"Lap time difference: {drv_a} was {format_delta(lap_delta)} vs {drv_b}",
        f"Overall faster: {drv_a if lap_delta.total_seconds() < 0 else drv_b}",
        f"",
        "Sector comparison:",
    ]

    faster_sectors = {drv_a: 0, drv_b: 0}

    for s in [1, 2, 3]:
        col = f"Sector{s}Time"
        t_a = driver_a.get(col)
        t_b = driver_b.get(col)

        if t_a is None or t_b is None or pd.isna(t_a) or pd.isna(t_b):
            lines.append(f"  Sector {s}: data unavailable for one or both drivers")
            continue

        delta = t_a - t_b  # positive = A slower in this sector
        faster = drv_a if delta.total_seconds() < 0 else drv_b
        faster_sectors[faster] += 1

        lines.append(
            f"  Sector {s}: {drv_a} {format_timedelta(t_a)} | "
            f"{drv_b} {format_timedelta(t_b)} | "
            f"Δ {format_delta(delta)} | Faster: {faster}"
        )

    lines += [
        "",
        f"Sectors won: {drv_a} {faster_sectors[drv_a]} — {drv_b} {faster_sectors[drv_b]}",
    ]

    # Narrative summary
    lap_gap_s = lap_delta.total_seconds()
    if abs(lap_gap_s) < 0.05:
        lines.append("Summary: Extremely close — less than 0.05s separating the two drivers overall.")
    elif abs(lap_gap_s) < 0.2:
        faster_drv = drv_a if lap_gap_s < 0 else drv_b
        slower_drv = drv_b if lap_gap_s < 0 else drv_a
        lines.append(
            f"Summary: {faster_drv} edged {slower_drv} by {abs(lap_gap_s):.3f}s. "
            f"Sector analysis reveals where the gap was built."
        )
    else:
        faster_drv = drv_a if lap_gap_s < 0 else drv_b
        slower_drv = drv_b if lap_gap_s < 0 else drv_a
        lines.append(
            f"Summary: {faster_drv} had a clear advantage of {abs(lap_gap_s):.3f}s over {slower_drv}."
        )

    # Tyre comparison
    comp_a = driver_a.get("Compound")
    comp_b = driver_b.get("Compound")
    if comp_a and comp_b and not pd.isna(comp_a) and not pd.isna(comp_b):
        if comp_a != comp_b:
            lines.append(f"Tyre note: {drv_a} on {comp_a}, {drv_b} on {comp_b} — different compounds.")
        else:
            lines.append(f"Tyre: both on {comp_a}.")

    return Chunk(
        chunk_id=make_chunk_id("head_to_head", circuit, year, drv_a, drv_b),
        chunk_type="head_to_head",
        text="\n".join(lines),
        metadata={
            "circuit":    circuit,
            "year":       year,
            "driver_a":   drv_a,
            "driver_b":   drv_b,
            "team_a":     team_a,
            "team_b":     team_b,
            "pos_a":      int(driver_a["QualifyingPosition"]),
            "pos_b":      int(driver_b["QualifyingPosition"]),
            "lap_delta_seconds": lap_delta.total_seconds(),
            "faster_driver":     drv_a if lap_delta.total_seconds() < 0 else drv_b,
        },
        source="fastf1",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_driver_chunks(
    fastf1_data: dict,
    n_telemetry_zones: int = 3,
    head_to_head_top_n: int = 5,
) -> list[Chunk]:
    """
    Main entry point. Produces all driver-level chunks for a session.

    Args:
        fastf1_data:         output of load_qualifying_session()
        n_telemetry_zones:   how many zones to split telemetry into (default 3 = approx sectors)
        head_to_head_top_n:  generate H2H chunks for top N drivers only
                             (avoids 20*19=380 chunks for full grid)

    Returns:
        [*lap_summary_chunks, *sector_chunks, *telemetry_zone_chunks, *h2h_chunks]
    """
    session_info = fastf1_data["session_info"]
    fastest      = fastf1_data["driver_fastest_laps"]
    telemetry    = fastf1_data.get("telemetry", {})
    dnq_drivers  = fastf1_data.get("dnq_drivers", [])

    if fastest.empty:
        return []

    pole_time = fastest.iloc[0]["LapTime"]
    chunks    = []

    # 0. DNQ status chunks — first in list so they surface first in context
    for dnq in dnq_drivers:
        chunks.append(chunk_driver_dnq_status(dnq, session_info))

    # 1. Lap summary — one per driver
    for _, row in fastest.iterrows():
        chunks.append(chunk_driver_lap_summary(row, session_info, pole_time))

    # 2. Sector chunks — one per driver per sector
    for _, row in fastest.iterrows():
        for sector in [1, 2, 3]:
            c = chunk_driver_sector(row, sector, session_info, fastest)
            if c:
                chunks.append(c)

    # 3. Telemetry zone chunks — one per driver per zone
    for _, row in fastest.iterrows():
        driver = row["Driver"]
        tel    = telemetry.get(driver)
        if tel is not None:
            chunks += chunk_driver_telemetry_zones(
                row, tel, session_info, n_zones=n_telemetry_zones
            )

    # 4. Head-to-head — top N drivers, all pairs within that group
    top_n = fastest.head(head_to_head_top_n)
    for i, (_, row_a) in enumerate(top_n.iterrows()):
        for j, (_, row_b) in enumerate(top_n.iterrows()):
            if i >= j:
                continue  # avoid duplicates (A vs B == B vs A)
            chunks.append(chunk_head_to_head(row_a, row_b, session_info))

    return chunks
