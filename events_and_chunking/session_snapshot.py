"""
session_snapshot.py
-------------------
A compact, exact copy of what the head-to-head chunks are computed from, saved when a session is ingested:
each driver's fastest-lap telemetry, the driver rows, the final track sections and the session info
(about 1 MB per session).

With it, any pair of drivers can be compared on demand by the same code that builds the stored chunks
(`head_to_head_chunks_for_pair`), so the text is identical and the store no longer needs a chunk for every pair.

Stored as a numpy .npz with no pickled objects: arrays per column, plus one JSON blob. Timedeltas are kept as integer
nanoseconds so a round trip is exact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from events_and_chunking.chunk_telemetry import head_to_head_chunks_for_pair
from events_and_chunking.chunk_types import Chunk

FORMAT_VERSION = 1


def clean_metadata(metadata: dict) -> dict:
    """The metadata a chunk is stored with: no None/NaN, numpy scalars unwrapped, containers as JSON."""
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and value != value:  # NaN
            continue
        if isinstance(value, (list, tuple, set, dict)):
            value = json.dumps(value)
        cleaned[key] = value
    return cleaned


@dataclass
class Snapshot:
    fastest: pd.DataFrame          # one row per driver (Driver, Team, QualifyingPosition, ...)
    telemetry: dict                # {driver: DataFrame}
    events: list                   # final sections, after normalising and labelling
    session_info: dict
    session_telemetry: dict = field(default_factory=dict)   # {"VER@Q1": DataFrame}: a driver's lap from a session other than his best


# ── DataFrame <-> arrays, exactly ──

def _frame_to_arrays(prefix: str, df: pd.DataFrame, arrays: dict, layout: dict) -> None:
    layout[prefix] = {"columns": [], "index": None}
    for col in df.columns:
        values = df[col]
        key = f"{prefix}/{col}"
        if pd.api.types.is_timedelta64_dtype(values):
            arrays[key] = values.to_numpy().astype("timedelta64[ns]").astype("int64")
            kind = "timedelta"
        elif pd.api.types.is_datetime64_any_dtype(values):
            arrays[key] = values.to_numpy().astype("datetime64[ns]").astype("int64")
            kind = "datetime"
        elif pd.api.types.is_bool_dtype(values):
            arrays[key] = values.to_numpy(dtype=bool)
            kind = "bool"
        elif pd.api.types.is_numeric_dtype(values):
            arrays[key] = values.to_numpy()
            kind = str(values.dtype)
        else:
            arrays[key] = np.array(["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in values])
            kind = "str"
        layout[prefix]["columns"].append([str(col), kind])
    layout[prefix]["index"] = [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in df.index]


def _arrays_to_frame(prefix: str, arrays, layout: dict) -> pd.DataFrame:
    data = {}
    for col, kind in layout[prefix]["columns"]:
        raw = arrays[f"{prefix}/{col}"]
        if kind == "timedelta":
            data[col] = pd.to_timedelta(raw, unit="ns")
        elif kind == "datetime":
            data[col] = pd.to_datetime(raw, unit="ns")
        elif kind == "str":
            data[col] = raw.astype(object)
        else:
            data[col] = raw
    return pd.DataFrame(data, index=layout[prefix]["index"])


def save_snapshot(path, fastest: pd.DataFrame, telemetry: dict, events: list, session_info: dict,
                  session_telemetry: Optional[dict] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict = {}
    layout: dict = {}
    _frame_to_arrays("fastest", fastest.reset_index(drop=True), arrays, layout)
    drivers = []
    for driver, tel in telemetry.items():
        if tel is None or tel.empty:
            continue
        drivers.append(driver)
        _frame_to_arrays(f"tel:{driver}", tel.reset_index(drop=True), arrays, layout)
    session_keys = []
    for key, tel in (session_telemetry or {}).items():
        if tel is None or tel.empty:
            continue
        session_keys.append(key)
        _frame_to_arrays(f"tel:{key}", tel.reset_index(drop=True), arrays, layout)
    meta = {"version": FORMAT_VERSION, "layout": layout, "drivers": drivers, "session_laps": session_keys,
            "events": events, "session_info": session_info}
    arrays["meta"] = np.array(json.dumps(meta, default=_json_default))
    np.savez_compressed(path, **arrays)
    return path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    return str(obj)


def load_snapshot(path) -> Snapshot:
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if meta["version"] != FORMAT_VERSION:
            raise ValueError(f"Unsupported snapshot version {meta['version']}")
        layout = meta["layout"]
        fastest = _arrays_to_frame("fastest", z, layout)
        telemetry = {d: _arrays_to_frame(f"tel:{d}", z, layout) for d in meta["drivers"]}
        extra = {k: _arrays_to_frame(f"tel:{k}", z, layout) for k in meta.get("session_laps", [])}
    return Snapshot(fastest=fastest, telemetry=telemetry, events=meta["events"], session_info=meta["session_info"],
                    session_telemetry=extra)


# ── on-demand comparison ──

def driver_row(snapshot: Snapshot, driver: str) -> Optional[pd.Series]:
    rows = snapshot.fastest[snapshot.fastest["Driver"] == driver]
    return rows.iloc[0] if not rows.empty else None


def best_session(row) -> Optional[int]:
    """The qualifying session (1, 2 or 3) a driver's best lap came from, or None when per-session times are not stored."""
    times = {n: row.get(f"Q{n}Time") for n in (1, 2, 3)}
    times = {n: t for n, t in times.items() if t is not None and t == t}
    return min(times, key=times.get) if times else None


def focus_session(times_a: dict, times_b: dict) -> Optional[int]:
    """
    The session to compare two drivers in. When their best laps come from the same session that is simply their best laps
    (None). When they come from different sessions the gap between best laps mixes sessions, so compare them in the last
    session in which both set a time (Q1 for a driver out in Q1; Q3 for two Q3 drivers, even if one was slower there).
    """
    if not times_a or not times_b:
        return None
    shared = sorted(set(times_a) & set(times_b))
    if not shared or min(times_a, key=times_a.get) == min(times_b, key=times_b.get):
        return None
    return shared[-1]


def session_times_of(row) -> dict:
    out = {n: row.get(f"Q{n}Time") for n in (1, 2, 3)}
    return {n: float(t) for n, t in out.items() if t is not None and t == t}


def lap_telemetry(snapshot: Snapshot, driver: str, session: Optional[int]):
    """The driver's telemetry for the lap he set in `session` (None = his best lap); None if that lap was not stored."""
    if session is None:
        return snapshot.telemetry.get(driver)
    row = driver_row(snapshot, driver)
    if row is not None and best_session(row) == session:
        return snapshot.telemetry.get(driver)
    return snapshot.session_telemetry.get(f"{driver}@Q{session}")


def compare_drivers(snapshot: Snapshot, a: str, b: str, session: Optional[int] = None) -> list[Chunk]:
    """
    The head-to-head section chunks for any two drivers in the session, in section order.
    The pair is oriented as the stored chunks are: whichever driver comes first in the session's driver table is
    driver A (the table is ordered by classification, which can differ from best-lap order). Empty if either driver
    has no lap in this session.

    With `session` (1, 2 or 3) each driver's lap from that qualifying session is compared instead of his best lap, which
    is the like-for-like comparison when the two best laps come from different sessions. Empty if either lap was not stored.
    """
    drivers = list(snapshot.fastest["Driver"])
    if a not in drivers or b not in drivers:
        return []
    if drivers.index(a) > drivers.index(b):
        a, b = b, a
    row_a, row_b = driver_row(snapshot, a), driver_row(snapshot, b)
    chunks = head_to_head_chunks_for_pair(
        row_a, row_b, lap_telemetry(snapshot, a, session), lap_telemetry(snapshot, b, session),
        snapshot.events, snapshot.session_info)
    if session is not None:
        for c in chunks:
            c.text = c.text.replace(" Qualifying\n", f" Qualifying ({a} and {b} Q{session} laps)\n", 1)
            c.metadata["lap_session"] = f"Q{session}"
    return chunks
