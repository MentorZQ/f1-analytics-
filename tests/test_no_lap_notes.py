import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_no_lap_notes as nl  # noqa: E402
import query  # noqa: E402
from events_and_chunking import session_queries as sq  # noqa: E402

START = pd.Timestamp("2026-06-13T14:50:44+00:00")


def samples(trace):
    """[(seconds into the lap, speed)] as OpenF1 car-data rows."""
    return pd.DataFrame({"date": [(START + pd.Timedelta(seconds=t)).isoformat() for t, _ in trace], "speed": [v for _, v in trace]})


def test_a_lap_that_ends_with_the_car_stopped_is_described_as_stopped_on_track_with_the_red_flag():
    car = samples([(0, 100), (10, 320), (20, 250), (26, 80), (28, 0), (60, 0), (100, 0)])
    text = nl.describe_last_lap(car, START, [START + pd.Timedelta(seconds=46)], "Q3")
    assert "was not completed" in text and "slowed to 0 km/h 28 s into the lap" in text and "it stopped on track" in text
    assert "A red flag was shown at 14:51:30, 18 s after it stopped." in text
    assert "crashed" not in text and "does not say why" in text


def test_a_lap_that_ends_back_in_the_pits_is_not_called_a_stop_on_track():
    car = samples([(0, 100), (10, 300), (30, 180), (60, 60), (90, 40)])
    text = nl.describe_last_lap(car, START, [], "Q3")
    assert "did not fall to 0 km/h" in text and "stopped on track" not in text


def test_the_segments_are_the_green_to_chequered_windows():
    rc = [{"date": d, "flag": f} for d, f in [("2026-06-13T14:00:00+00:00", "GREEN"), ("2026-06-13T14:18:00+00:00", "CHEQUERED"),
                                              ("2026-06-13T14:25:00+00:00", "GREEN"), ("2026-06-13T14:40:00+00:00", "CHEQUERED"),
                                              ("2026-06-13T14:47:00+00:00", "GREEN"), ("2026-06-13T15:03:00+00:00", "GREEN"),
                                              ("2026-06-13T15:11:30+00:00", "CHEQUERED")]]
    w = nl.segment_windows(rc)
    assert set(w) == {1, 2, 3} and w[3][0] == pd.Timestamp("2026-06-13T14:47:00+00:00")        # the restart after a red flag does not move the start


def test_the_committed_notes_say_leclerc_stopped_on_track_at_barcelona_and_verstappen_at_melbourne():
    barcelona = json.loads((ROOT / "RAG_data_layers/session_notes/f1_qualifying_2026_barcelona.json").read_text())
    melbourne = json.loads((ROOT / "RAG_data_layers/session_notes/f1_qualifying_2026_melbourne.json").read_text())
    assert "stopped on track" in barcelona["LEC"][0]["text"] and barcelona["LEC"][0]["segment"] == "Q3"
    assert "stopped on track" in melbourne["VER"][0]["text"]
    assert melbourne["SAI"][0]["text"].startswith("The data records no laps")


def test_the_grid_and_the_prompt_context_carry_the_note():
    ctx = {"no_lap_notes": {"LEC": [{"segment": "Q3", "text": "His last lap in Q3 was not completed: it stopped on track."}]},
           "drivers": {"LEC": {"lap_time_seconds": 75.281, "qualifying_position": 10, "team": "Ferrari", "stage": "Q3", "session_times": {1: 75.964, 2: 75.281}}}}
    assert "Why LEC has no time in Q3: His last lap in Q3 was not completed: it stopped on track." in sq.grid_text(ctx)
    chunks = query.extra_context("how did leclerc do in barca", "barcelona", query._get_session_context("barcelona"))
    assert any(c["metadata"]["chunk_type"] == "driver_note" and "stopped on track" in c["text"] for c in chunks)
