"""
chunk_types.py
--------------
Defines the Chunk dataclass that every chunker produces.
All chunk types share the same structure so the embedding layer
and vector store don't need to care about where a chunk came from.

Chunk fields:
    chunk_id    — unique identifier (deterministic, based on content)
    chunk_type  — one of the CHUNK_TYPES below
    text        — the plain-text content that gets embedded
    metadata    — structured dict for vector store filtering
                  (driver, circuit, year, sector, chunk_type, etc.)
    source      — which loader produced this chunk
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import hashlib
import json


# All valid chunk types in the pipeline
CHUNK_TYPES = {
    "circuit_overview",           # Static circuit facts (length, altitude, corners)
    "driver_lap_summary",         # One driver's full qualifying lap summary
    "driver_sector",              # One driver, one sector (times + telemetry summary)
    "driver_telemetry_zone",      # One driver, one track zone (braking/acceleration detail)
    "session_weather",            # Weather narrative across the session
    "race_control_event",         # Single flag / message / deletion event
    "head_to_head",               # Two-driver comparison on a metric
    # Event-aware chunk types (require detect_events geometry)
    "telemetry_corner_event",     # One driver, one corner/sequence — entry/apex/exit detail
    "telemetry_straight_event",   # One driver, one straight — top speed, DRS, gear
    "head_to_head_event",         # Two drivers at the same corner or straight
    "session_pace_evolution",     # Lap time progression per driver across Q1/Q2/Q3
}


@dataclass
class Chunk:
    """
    A single embeddable unit of information about a qualifying session.

    text     — what gets embedded and returned to the LLM as context
    metadata — structured fields for filtering before embedding search
               (e.g. retrieve only chunks where driver='VER' and sector=2)
    """
    chunk_id:   str
    chunk_type: str
    text:       str
    metadata:   dict = field(default_factory=dict)
    source:     str  = "unknown"

    def __post_init__(self):
        if self.chunk_type not in CHUNK_TYPES:
            raise ValueError(
                f"Invalid chunk_type '{self.chunk_type}'. "
                f"Must be one of: {sorted(CHUNK_TYPES)}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self):
        preview = self.text[:80].replace('\n', ' ')
        return f"Chunk(type={self.chunk_type}, id={self.chunk_id[:8]}…, text='{preview}…')"


def make_chunk_id(*parts) -> str:
    """
    Generate a deterministic chunk ID from its identifying parts.
    Same inputs always produce the same ID — avoids duplicates on re-runs.

    Usage:
        make_chunk_id("circuit_overview", "Monza", "2024")
        make_chunk_id("driver_sector", "Monza", "2024", "VER", "2")
    """
    key = "|".join(str(p) for p in parts)
    return hashlib.md5(key.encode()).hexdigest()


def format_timedelta(td) -> str:
    """
    Format a pandas Timedelta as a human-readable lap time string.
    e.g. Timedelta('0 days 00:01:19.619000') -> '1:19.619'
    Returns 'N/A' for null values.
    """
    import pandas as pd
    if pd.isna(td):
        return "N/A"
    total_seconds = td.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    if minutes > 0:
        return f"{minutes}:{seconds:06.3f}"
    return f"{seconds:.3f}s"


def format_delta(td) -> str:
    """
    Format a timedelta as a +/- gap string.
    e.g. '+0.342s' or '-0.118s'
    """
    import pandas as pd
    if pd.isna(td):
        return "N/A"
    total = td.total_seconds()
    sign = "+" if total >= 0 else ""
    return f"{sign}{total:.3f}s"
