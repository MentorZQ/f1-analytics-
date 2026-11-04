"""Who reached Q3, who was last into it, and who just missed it, from the committed session snapshots."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from events_and_chunking import session_queries as sq  # noqa: E402
from events_and_chunking import session_snapshot as snap  # noqa: E402

SNAPSHOTS = ROOT / "RAG_data_layers" / "session_snapshots"


def grid(name: str) -> tuple[dict, str]:
    s = snap.load_snapshot(SNAPSHOTS / f"{name}.npz")
    drivers = {}
    for _, row in s.fastest.iterrows():
        drivers[row["Driver"]] = {"lap_time_seconds": row["LapTime"].total_seconds(), "qualifying_position": int(row["QualifyingPosition"]),
                                  "team": row["Team"], "stage": row["QSessionReached"], "session_times": snap.session_times_of(row)}
    return {"drivers": drivers}, sq.grid_text({"drivers": drivers})


def line(text: str, code: str) -> str:
    return next(l for l in text.splitlines() if f" {code} (" in l)


def test_monza_hamilton_reached_q3_and_alonso_went_out_in_q2():
    _, text = grid("f1_qualifying_2024_monza")
    assert "reached Q3" in line(text, "HAM") and "out in Q2" in line(text, "ALO")


def test_monza_the_slowest_q2_time_among_the_cars_that_advanced():
    ctx, text = grid("f1_qualifying_2024_monza")
    q3 = [c for c, i in ctx["drivers"].items() if i["stage"] == "Q3"]
    slowest = max(q3, key=lambda c: ctx["drivers"][c]["session_times"][2])
    assert len(q3) == 10 and f"the slowest Q2 time among them: {slowest} " in text


def test_barcelona_leclerc_reached_q3_without_a_q3_time_and_lindblad_missed_out():
    ctx, text = grid("f1_qualifying_2026_barcelona")
    assert "reached Q3 but set no Q3 time" in line(text, "LEC") and ctx["drivers"]["LEC"]["qualifying_position"] == 10
    assert "out in Q2" in line(text, "LIN")
    assert "the slowest Q2 time among them: HUL 1:15.768" in text          # not Leclerc, the classified P10 car


def test_melbourne_bortoleto_is_p10_with_no_q3_time_and_still_reached_q3():
    ctx, text = grid("f1_qualifying_2026_melbourne")
    assert ctx["drivers"]["BOR"]["qualifying_position"] == 10 and "reached Q3 but set no Q3 time" in line(text, "BOR")
    assert "the slowest Q2 time among them: BOR 1:20.221" in text


@pytest.mark.parametrize("name", ["f1_qualifying_2024_monza", "f1_qualifying_2026_barcelona", "f1_qualifying_2026_melbourne"])
def test_exactly_ten_cars_reach_q3_in_every_session(name):
    ctx, _ = grid(name)
    assert sum(i["stage"] == "Q3" for i in ctx["drivers"].values()) == 10
