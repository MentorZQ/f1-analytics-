"""
load_circuit_official.py
------------------------
Official corner list and track outline for a circuit, from the MultiViewer
circuit API (the same source FastF1 uses for circuit info). Free, no login.

Per circuit and year it returns:
  corners : official turn number, letter (if any) and apex position
  outline : the track path for that year's layout

Positions are in the same coordinate frame as OpenF1's `location` data
(units of 0.1 m), so no registration is needed. They are converted to metres here.

This is a community API with no service guarantee, so each response is saved
to RAG_data_layers/circuit_data/ and the saved copy is used first. Commit those
files: the pipeline then never depends on the live service for known circuits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://api.multiviewer.app/api/v1/circuits/{key}/{year}"
CACHE_DIR = Path(__file__).resolve().parent.parent / "circuit_data"
UNIT_M = 0.1


def _fetch(circuit_key: int, year: int, cache_dir: Path) -> tuple[dict, int]:
    last = None
    for y in (year, year - 1, year - 2):          # fall back to the nearest earlier layout
        saved = cache_dir / f"multiviewer_{circuit_key}_{y}.json"
        if saved.exists():
            return json.loads(saved.read_text()), y
        r = requests.get(API.format(key=circuit_key, year=y),
                         headers={"User-Agent": "f1-analytics-research/0.1"}, timeout=30)
        if r.status_code == 200:
            cache_dir.mkdir(parents=True, exist_ok=True)
            saved.write_text(json.dumps(r.json()))
            return r.json(), y
        last = r.status_code
    raise LookupError(f"No circuit data for key {circuit_key} ({year} or the two years before); last HTTP {last}")


def load_circuit_official(circuit_key: int, year: int, cache_dir: Path = CACHE_DIR) -> dict:
    raw, year_used = _fetch(circuit_key, year, Path(cache_dir))

    outline = pd.DataFrame({"x_m": np.asarray(raw["x"], float) * UNIT_M,
                            "y_m": np.asarray(raw["y"], float) * UNIT_M})
    corners = []
    for c in sorted(raw["corners"], key=lambda c: (c["number"], c.get("letter") or "")):
        label = f"{c['number']}{c.get('letter') or ''}"
        corners.append({
            "number": c["number"], "label": label,
            "x_m": c["trackPosition"]["x"] * UNIT_M, "y_m": c["trackPosition"]["y"] * UNIT_M,
        })
    return {"circuit_key": circuit_key, "year_used": year_used, "name": raw.get("circuitName"),
            "outline": outline, "corners": corners}
