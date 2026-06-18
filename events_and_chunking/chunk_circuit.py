"""
chunk_circuit.py
----------------
Produces circuit_overview chunks from track metadata.

These are static chunks — they don't change per session, only per circuit.
They give the LLM spatial and contextual grounding when answering questions
like "why is sector 2 at Monza so fast?" or "what makes Spa difficult?".

Input:  output of load_all_track_data(circuit_name)
Output: list of Chunk objects (usually 1-2 per circuit)
"""

import pandas as pd
import numpy as np
from typing import Optional
from .chunk_types import Chunk, make_chunk_id, CHUNK_TYPES


def chunk_circuit_overview(track_data: dict) -> Chunk:
    """
    Single chunk summarising the circuit's key characteristics.

    Covers: length, altitude, track width stats, first GP year, location.
    This is the foundational context chunk for any question about this circuit.
    """
    meta = track_data["metadata"]
    geom: Optional[pd.DataFrame] = track_data.get("geometry")

    lines = [
        f"Circuit Overview: {meta['name']}",
        f"Location: {meta['location']}",
        f"Track length: {meta['length_m']}m ({meta['length_m'] / 1000:.3f} km)",
        f"Altitude: {meta['altitude_m']}m above sea level",
        f"First Formula 1 Grand Prix: {meta['first_gp_year']}",
        f"Circuit opened: {meta['opened']}",
    ]

    if geom is not None and not geom.empty:
        total_width = geom["w_tr_right_m"] + geom["w_tr_left_m"]
        lines += [
            f"Average track width: {total_width.mean():.1f}m",
            f"Narrowest point: {total_width.min():.1f}m",
            f"Widest point: {total_width.max():.1f}m",
            f"Track geometry: {len(geom)} centerline coordinate points available",
        ]
    else:
        lines.append("Detailed geometry: not available for this circuit (GPS outline only)")

    # GPS bounding box gives a rough sense of circuit footprint
    coords = meta.get("gps_coordinates", [])
    if coords:
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        # 1 degree latitude ≈ 111km; 1 degree longitude ≈ 111km * cos(lat)
        import math
        avg_lat = sum(lats) / len(lats)
        lon_km = (max(lons) - min(lons)) * 111 * math.cos(math.radians(avg_lat))
        lat_km = (max(lats) - min(lats)) * 111
        lines.append(
            f"Circuit footprint: ~{lon_km:.1f}km east-west, "
            f"~{lat_km:.1f}km north-south"
        )

    return Chunk(
        chunk_id=make_chunk_id("circuit_overview", meta["name"]),
        chunk_type="circuit_overview",
        text="\n".join(lines),
        metadata={
            "circuit":       meta["name"],
            "length_m":      meta["length_m"],
            "altitude_m":    meta["altitude_m"],
            "first_gp_year": meta["first_gp_year"],
            "has_geometry":  geom is not None,
        },
        source="bacinger+tumftm",
    )


def chunk_circuit_geometry_zones(track_data: dict, n_zones: int = 3) -> list[Chunk]:
    """
    Divide the circuit geometry into n_zones equal segments and produce
    one chunk per zone describing its width characteristics.

    This gives the LLM zone-level spatial context:
    e.g. "Zone 1 (first third of the lap) has an average width of 12.4m
    and narrows to 9.1m at its tightest — consistent with a technical
    section before a wide braking zone."

    n_zones=3 maps naturally onto F1 sectors (not exact sector boundaries,
    but a reasonable approximation when sector data isn't in the geometry).

    Returns empty list if no geometry is available for this circuit.
    """
    meta = track_data["metadata"]
    geom: Optional[pd.DataFrame] = track_data.get("geometry")

    if geom is None or geom.empty:
        return []

    chunks = []
    geom = geom.copy()
    geom["total_width"] = geom["w_tr_right_m"] + geom["w_tr_left_m"]
    zone_size = len(geom) // n_zones

    for zone_idx in range(n_zones):
        start = zone_idx * zone_size
        end = start + zone_size if zone_idx < n_zones - 1 else len(geom)
        zone = geom.iloc[start:end]

        avg_w  = zone["total_width"].mean()
        min_w  = zone["total_width"].min()
        max_w  = zone["total_width"].max()

        # Rough distance covered in this zone (sum of point-to-point distances)
        dx = zone["x_m"].diff().fillna(0)
        dy = zone["y_m"].diff().fillna(0)
        zone_length = np.sqrt(dx**2 + dy**2).sum()

        # Characterise the zone
        if avg_w < 11:
            width_char = "narrow, technical"
        elif avg_w < 14:
            width_char = "medium-width"
        else:
            width_char = "wide, high-speed"

        lines = [
            f"Circuit: {meta['name']} — Geometry Zone {zone_idx + 1} of {n_zones}",
            f"Approximate zone length: {zone_length:.0f}m of {meta['length_m']}m total",
            f"Zone character: {width_char}",
            f"Average track width: {avg_w:.1f}m",
            f"Narrowest point in zone: {min_w:.1f}m",
            f"Widest point in zone: {max_w:.1f}m",
            f"Width variability (std): {zone['total_width'].std():.2f}m "
            f"({'high variation — mixed corner types' if zone['total_width'].std() > 1.5 else 'consistent width'})",
        ]

        chunks.append(Chunk(
            chunk_id=make_chunk_id("circuit_geometry_zone", meta["name"], zone_idx),
            chunk_type="circuit_overview",
            text="\n".join(lines),
            metadata={
                "circuit":        meta["name"],
                "zone_index":     zone_idx,
                "zone_of":        n_zones,
                "avg_width_m":    round(avg_w, 2),
                "min_width_m":    round(min_w, 2),
                "zone_length_m":  round(zone_length, 1),
            },
            source="tumftm",
        ))

    return chunks


def build_circuit_chunks(track_data: dict, n_zones: int = 3) -> list[Chunk]:
    """
    Main entry point. Produces all circuit-level chunks for a given track.

    Returns:
        [circuit_overview_chunk, *geometry_zone_chunks]
    """
    chunks = [chunk_circuit_overview(track_data)]
    chunks += chunk_circuit_geometry_zones(track_data, n_zones=n_zones)
    return chunks
