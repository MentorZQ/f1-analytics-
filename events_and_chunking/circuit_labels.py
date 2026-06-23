"""
circuit_labels.py
-----------------
Maps internal geometry-derived event IDs to official circuit turn names.

Internal IDs (e.g. "T8-T9-T10-T11") are sequential curvature peaks and
do not align with official FIA turn numbers. This module provides per-circuit
display names so all user-facing content references the turns fans and
engineers recognise.

Rules:
  - If a detected group spans multiple official turns, list all (e.g. "T6-T7-T8")
  - If a detected group spans no official turn (curvy non-numbered section),
    use "Curve/Sweeper before/after/between T{n}" language
  - Straights are labelled with the official turns they connect
  - S_main and S_finish always display as "Main straight"

Official corner distances sourced from FastF1 get_circuit_info():
  Melbourne 2026 : T1–T14 (14 corners)
  Barcelona 2026 : T1–T14 (14 corners)
  Monza 2024     : T1–T11 (11 corners)
"""

# ---------------------------------------------------------------------------
# Per-circuit mappings: internal_event_id -> official display name
# ---------------------------------------------------------------------------

_MELBOURNE = {
    # corners
    "T1-T2":          "T1-T2",
    "T3":             "Curve before T3",
    "T4-T5":          "T3-T4",
    "T6":             "T5",
    "T7":             "Curve between T5-T6",
    "T8-T9-T10-T11":  "T6-T7-T8",
    "T12-T13":        "Curves between T8-T9",
    "T14-T15-T16":    "Sweepers before T9",
    "T17-T18":        "T9-T10",
    "T19-T20":        "Curves between T10-T11",
    "T21":            "T11",
    "T22":            "T12",
    "T23-T24":        "T13-T14",
    # straights
    "S_main":  "Main straight (start, before T1)",
    "S1":      "Between T2-T3",
    "S2":      "Before T3-T4",
    "S3":      "Between T4-T5",
    "S4":      "Between T5 and curve",
    "S5":      "Before T6",
    "S6":      "After T8",
    "S7":      "Between curves after T8",
    "S8":      "Approaching T9",
    "S9":      "After T10",
    "S10":     "Before T11",
    "S11":     "Between T11-T12",
    "S12":     "Between T12-T13",
    "S_finish": "Main straight (finish, after final corner)",
}

_BARCELONA = {
    # corners
    "T1-T2-T3-T4-T5-T6":  "T1-T2-T3-T4-T5",
    "T7-T8-T9":            "T6-T7-T8",
    "T10":                 "T9",
    "T11-T12-T13-T14":     "T10-T11-T12",
    "T15-T16-T17-T18":     "T13-T14",
    # straights
    "S_main":  "Main straight (start, before T1)",
    "S1":      "Between T5-T6",
    "S2":      "Between T8-T9",
    "S3":      "Between T9-T10",
    "S4":      "Between T12-T13",
    "S_finish": "Main straight (finish, after final corner)",
}

_MONZA = {
    # corners
    "T1":      "T1-T2",
    "T2":      "Sweeper between T2-T3",
    "T3-T4":   "T3",
    "T5":      "T4-T5",
    "T6":      "T6",
    "T7":      "T7",
    "T8":      "Sweeper between T7-T8",
    "T9-T10":  "T8-T9-T10",
    "T11-T12": "T11",
    # straights
    "S_main":  "Main straight (start, before T1)",
    "S1":      "Between T2-T3",
    "S2":      "Before T3",
    "S3":      "Before T4-T5",
    "S4":      "Between T5-T6",
    "S5":      "Between T6-T7",
    "S6":      "After T7",
    "S7":      "Before T8-T9-T10",
    "S8":      "Before T11",
    "S_finish": "Main straight (finish, after final corner)",
}

CIRCUIT_LABELS: dict[str, dict[str, str]] = {
    "Melbourne":  _MELBOURNE,
    "Barcelona":  _BARCELONA,
    "Catalunya":  _BARCELONA,  # OpenF1 short name alias
    "Monza":      _MONZA,
}


def relabel_events(events: list[dict], circuit: str) -> list[dict]:
    """
    Add a display_id field to each event with the official circuit turn name.

    The internal "id" is preserved unchanged so chunk IDs (which are hashed
    from the internal ID) remain unique. All user-facing text uses display_id.
    For unknown circuits or unmapped IDs, display_id falls back to "id".
    """
    mapping = CIRCUIT_LABELS.get(circuit)
    relabelled = []
    for ev in events:
        ev = dict(ev)
        official = mapping.get(ev.get("id", "")) if mapping else None
        ev["display_id"] = official if official else ev.get("id", "")
        relabelled.append(ev)
    return relabelled
