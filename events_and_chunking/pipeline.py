"""
pipeline.py
-----------
Top-level chunking pipeline. Single entry point that runs all chunkers
in the correct order and returns the complete chunk set for a session.

Usage:
    from chunking.pipeline import build_all_chunks

    chunks = build_all_chunks(
        fastf1_data  = load_qualifying_session(2024, "Monza"),
        openf1_data  = load_openf1_qualifying_context(2024, "monza"),
        track_data   = load_all_track_data("Monza"),
        session_info = {"year": 2024, "circuit": "Monza"},
    )

    print(f"{len(chunks)} chunks produced")
    for c in chunks:
        print(c.chunk_type, c.metadata.get("driver", ""), c.metadata.get("event_id", ""))
"""

import pandas as pd
from .chunk_types import Chunk
from .chunk_circuit import build_circuit_chunks
from .chunk_session import build_session_chunks
from .chunk_driver import build_driver_chunks
from .chunk_telemetry import build_telemetry_chunks
from .detect_events import detect_circuit_events, events_to_summary


def build_all_chunks(
    fastf1_data:  dict,
    openf1_data:  dict,
    track_data:   dict,
    session_info: dict,
    head_to_head_top_n: int = 5,
    verbose: bool = True,
) -> list[Chunk]:
    """
    Run the full chunking pipeline for one qualifying session.

    Args:
        fastf1_data:        output of load_qualifying_session()
        openf1_data:        output of load_openf1_qualifying_context()
        track_data:         output of load_all_track_data()
        session_info:       dict with at minimum {"year": int, "circuit": str}
        head_to_head_top_n: H2H chunks for top N qualifiers + all teammates
        verbose:            print chunk counts per type

    Returns:
        List of Chunk objects ready for embedding and storage.
    """
    all_chunks = []
    circuit    = session_info.get("circuit", "Unknown")
    year       = session_info.get("year", "")

    if verbose:
        print(f"\nBuilding chunks: {circuit} {year} Qualifying")
        print("=" * 55)

    # --- 1. Circuit geometry chunks (static) ---
    circuit_chunks = build_circuit_chunks(track_data)
    all_chunks += circuit_chunks
    if verbose:
        print(f"  circuit_overview        : {len(circuit_chunks)} chunks")

    # --- 2. Detect track events for telemetry windowing ---
    geometry = track_data.get("geometry")
    events   = None

    if geometry is not None and not geometry.empty:
        try:
            events = detect_circuit_events(geometry, circuit_name=circuit, verbose=False)
            if verbose:
                n_corners   = sum(1 for e in events if e["type"] in ("corner", "corner_sequence"))
                n_straights = sum(1 for e in events if e["type"] == "straight")
                print(f"  track events detected   : {len(events)} "
                      f"({n_corners} corner groups, {n_straights} straights)")
        except Exception as e:
            if verbose:
                print(f"  track event detection failed: {e}")
            events = None
    else:
        if verbose:
            print(f"  track events            : skipped (no TUMFTM geometry for {circuit})")

    # --- 3. Session-level chunks (weather + race control) ---
    session_chunks = build_session_chunks(openf1_data, session_info)
    all_chunks += session_chunks
    weather_n = sum(1 for c in session_chunks if c.chunk_type == "session_weather")
    rc_n      = sum(1 for c in session_chunks if c.chunk_type == "race_control_event")
    if verbose:
        print(f"  session_weather         : {weather_n} chunks")
        print(f"  race_control_event      : {rc_n} chunks")

    # --- 4. Driver lap + sector chunks ---
    driver_chunks = build_driver_chunks(
        fastf1_data,
        n_telemetry_zones=3,
        head_to_head_top_n=head_to_head_top_n,
    )
    all_chunks += driver_chunks
    by_type = {}
    for c in driver_chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    if verbose:
        for t, n in sorted(by_type.items()):
            print(f"  {t:24s}: {n} chunks")

    # --- 5. Telemetry event chunks (requires geometry events) ---
    if events is not None:
        tel_chunks = build_telemetry_chunks(
            fastf1_data,
            events,
            session_info,
            head_to_head_top_n=head_to_head_top_n,
        )
        all_chunks += tel_chunks
        by_type = {}
        for c in tel_chunks:
            by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
        if verbose:
            for t, n in sorted(by_type.items()):
                print(f"  {t:24s}: {n} chunks")
    else:
        if verbose:
            print(f"  telemetry event chunks  : skipped (no geometry events)")

    if verbose:
        print(f"  {'─'*40}")
        print(f"  Total                   : {len(all_chunks)} chunks")

        # Breakdown by type
        print(f"\nFull breakdown by chunk_type:")
        type_counts = {}
        for c in all_chunks:
            type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
        for t, n in sorted(type_counts.items()):
            print(f"  {t:32s}: {n}")

    return all_chunks


def chunk_summary(chunks: list[Chunk]) -> str:
    """Return a plain-text summary of a chunk set."""
    type_counts = {}
    for c in chunks:
        type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1

    lines = [f"Total chunks: {len(chunks)}", ""]
    for t, n in sorted(type_counts.items()):
        lines.append(f"  {t}: {n}")
    return "\n".join(lines)


def validate_chunks(chunks: list[Chunk]) -> list[str]:
    """
    Basic validation of a chunk set. Returns list of warning strings.
    Empty list = all good.
    """
    warnings = []

    # Check for duplicate IDs
    ids = [c.chunk_id for c in chunks]
    dupes = [cid for cid in set(ids) if ids.count(cid) > 1]
    if dupes:
        warnings.append(f"{len(dupes)} duplicate chunk IDs found: {dupes[:5]}")

    # Check for empty text
    empty = [c.chunk_id for c in chunks if not c.text.strip()]
    if empty:
        warnings.append(f"{len(empty)} chunks have empty text")

    # Check metadata has minimum required fields
    # circuit_overview chunks are session-independent — year is optional for them
    for c in chunks:
        required = {"circuit"} if c.chunk_type == "circuit_overview" else {"circuit", "year"}
        if not required.issubset(c.metadata.keys()):
            warnings.append(f"Chunk {c.chunk_id[:8]} ({c.chunk_type}) missing required metadata fields")

    return warnings
