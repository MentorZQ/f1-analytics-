"""
load_track_data.py
------------------
Loads static circuit geometry from two open-source repositories:

1. bacinger/f1-circuits (GeoJSON)
   - GPS coordinates of track centerline
   - Circuit metadata: length, altitude, first GP year, location
   - 40 circuits from 1950 to present
   - https://github.com/bacinger/f1-circuits

2. TUMFTM/racetrack-database (CSV)
   - Higher-resolution centerline coordinates (x_m, y_m in meters)
   - Track width (left + right of centerline)
   - Optimized race line
   - Available for ~8 current F1 circuits
   - https://github.com/TUMFTM/racetrack-database

Usage:
    from load_track_data import load_circuit_metadata, load_circuit_geometry
    meta = load_circuit_metadata("Monza")
    geom = load_circuit_geometry("Monza")
"""

import requests
import pandas as pd
import json
from typing import Optional

# --- Sources ---
BACINGER_URL = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/f1-circuits.geojson"
TUMFTM_BASE  = "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks"

# Map common circuit names to TUMFTM filenames (only ~8 available)
TUMFTM_CIRCUITS = {
    "Monza":       "Monza",
    "Silverstone": "Silverstone",
    "Spa":         "Spa",
    "Suzuka":      "Suzuka",
    "Barcelona":   "Catalunya",
    "Catalunya":   "Catalunya",
    "Zandvoort":   "Zandvoort",
    "Austin":      "Austin",
    "COTA":        "Austin",
    "Melbourne":   "Melbourne",
    "Albert Park": "Melbourne",
}

# Map common names to bacinger circuit names (partial match used)
BACINGER_NAME_HINTS = {
    "Monza":       "Monza",
    "Silverstone": "Silverstone",
    "Spa":         "Spa",
    "Monaco":      "Monaco",
    "Bahrain":     "Bahrain",
    "Suzuka":      "Suzuka",
    "Barcelona":   "Barcelona",
    "Catalunya":   "Barcelona-Catalunya",
    "Zandvoort":   "Zandvoort",
    "Jeddah":      "Jeddah",
    "Miami":       "Miami",
    "Baku":        "Baku",
    "Singapore":   "Marina Bay",
    "Austin":      "Americas",
    "COTA":        "Americas",
    "Interlagos":  "Interlagos",
    "Mexico":      "Hermanos",
    "Las Vegas":   "Las Vegas",
    "Abu Dhabi":   "Yas Marina",
    "Melbourne":   "Albert Park",
}


def _fetch_bacinger() -> list:
    """Download and cache bacinger GeoJSON features."""
    r = requests.get(BACINGER_URL, timeout=15)
    r.raise_for_status()
    return r.json()["features"]


def load_circuit_metadata(circuit_name: str) -> Optional[dict]:
    """
    Load circuit metadata from bacinger GeoJSON.

    Returns a dict with:
        name, location, length_m, altitude_m, first_gp_year, opened,
        gps_coordinates (list of [lon, lat] pairs forming track outline)

    Args:
        circuit_name: common name e.g. "Monza", "Silverstone", "Spa"
    """
    hint = BACINGER_NAME_HINTS.get(circuit_name, circuit_name)
    features = _fetch_bacinger()

    # Fuzzy match on Name field
    match = None
    for f in features:
        name = f["properties"].get("Name", "")
        if hint.lower() in name.lower() or name.lower() in hint.lower():
            match = f
            break

    if not match:
        available = [f["properties"]["Name"] for f in features]
        raise ValueError(
            f"Circuit '{circuit_name}' not found in bacinger dataset.\n"
            f"Available: {available}"
        )

    props = match["properties"]
    coords = match["geometry"]["coordinates"]  # [[lon, lat], ...]

    return {
        "name": props.get("Name"),
        "location": props.get("Location"),
        "circuit_id": props.get("id"),
        "length_m": props.get("length"),
        "altitude_m": props.get("altitude"),
        "first_gp_year": props.get("firstgp"),
        "opened": props.get("opened"),
        "gps_coordinates": coords,          # full GPS outline
        "coordinate_count": len(coords),
    }


def load_circuit_geometry(circuit_name: str) -> Optional[pd.DataFrame]:
    """
    Load high-resolution circuit geometry from TUMFTM racetrack database.

    Returns a DataFrame with columns:
        x_m, y_m           — centerline coordinates in meters (relative origin)
        w_tr_right_m        — track width to the right
        w_tr_left_m         — track width to the left

    Returns None if the circuit isn't in the TUMFTM database.

    Available circuits: Monza, Silverstone, Spa, Suzuka, Catalunya,
                        Zandvoort, Austin, Melbourne
    """
    tumftm_name = TUMFTM_CIRCUITS.get(circuit_name)
    if not tumftm_name:
        print(f"  Note: {circuit_name} not in TUMFTM database. "
              f"Available: {list(TUMFTM_CIRCUITS.keys())}")
        return None

    url = f"{TUMFTM_BASE}/{tumftm_name}.csv"
    r = requests.get(url, timeout=15)

    if r.status_code == 404:
        print(f"  TUMFTM file not found for {tumftm_name}")
        return None

    r.raise_for_status()

    # First line is a comment: "# x_m,y_m,w_tr_right_m,w_tr_left_m"
    lines = r.text.strip().split("\n")
    header_line = lines[0].lstrip("# ").strip()
    columns = [c.strip() for c in header_line.split(",")]

    df = pd.read_csv(
        pd.io.common.StringIO("\n".join(lines[1:])),
        header=None,
        names=columns,
    )

    return df


def load_all_track_data(circuit_name: str) -> dict:
    """
    Convenience function: load both metadata and geometry for a circuit.

    Returns:
        {
            "metadata": dict from bacinger (name, length, altitude, GPS coords),
            "geometry": DataFrame from TUMFTM (x/y/width), or None if unavailable
        }
    """
    print(f"\nLoading track data for: {circuit_name}")

    metadata = load_circuit_metadata(circuit_name)
    print(f"  Metadata: {metadata['name']}, {metadata['length_m']}m, "
          f"alt={metadata['altitude_m']}m, {metadata['coordinate_count']} GPS points")

    geometry = load_circuit_geometry(circuit_name)
    if geometry is not None:
        print(f"  Geometry: {len(geometry)} centerline points with track widths")
    else:
        print(f"  Geometry: not available (GPS outline only)")

    return {"metadata": metadata, "geometry": geometry}


def describe_circuit(circuit_name: str) -> str:
    """
    Generate a plain-text description of a circuit suitable for embedding
    as a RAG document chunk.

    This is the text that will go into your vector store so the LLM
    has circuit context when answering questions.
    """
    track = load_all_track_data(circuit_name)
    meta = track["metadata"]
    geom = track["geometry"]

    lines = [
        f"Circuit: {meta['name']}",
        f"Location: {meta['location']}",
        f"Track length: {meta['length_m']}m ({meta['length_m']/1000:.3f} km)",
        f"Altitude: {meta['altitude_m']}m above sea level",
        f"First Formula 1 Grand Prix: {meta['first_gp_year']}",
        f"Circuit opened: {meta['opened']}",
    ]

    if geom is not None:
        avg_width = ((geom["w_tr_right_m"] + geom["w_tr_left_m"]) / 2).mean()
        lines.append(f"Average track width: {avg_width:.1f}m")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with a few circuits
    for circuit in ["Monza", "Silverstone", "Monaco", "Jeddah"]:
        print(f"\n{'='*40}")
        print(describe_circuit(circuit))
