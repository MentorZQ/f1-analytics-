"""Which driver pairs get stored head-to-head comparisons."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from events_and_chunking.chunk_telemetry import head_to_head_pairs  # noqa: E402


def grid(n=12):
    teams = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E", "F", "F"]
    return pd.DataFrame({"Driver": [f"D{i:02d}" for i in range(n)], "Team": teams[:n]})


def names(pairs):
    return {frozenset((a["Driver"], b["Driver"])) for a, b in pairs}


def test_top_n_teammates_and_neighbours_are_all_covered():
    got = names(head_to_head_pairs(grid(), top_n=3))
    assert frozenset(("D00", "D02")) in got          # both in the top 3
    assert frozenset(("D10", "D11")) in got          # teammates at the back
    assert frozenset(("D06", "D07")) in got          # neighbours (also teammates here)
    assert frozenset(("D03", "D04")) in got          # neighbours on different teams
    assert frozenset(("D00", "D09")) not in got      # not top-N, not teammates, not neighbours


def test_no_pair_is_listed_twice_or_with_itself():
    pairs = head_to_head_pairs(grid(), top_n=5)
    keys = [frozenset((a["Driver"], b["Driver"])) for a, b in pairs]
    assert len(keys) == len(set(keys))
    assert all(len(k) == 2 for k in keys)


def test_pair_count_stays_modest():
    # top 10 all pairs = 45; teammates/neighbours add at most ~2 per driver beyond that on a 22-car grid
    n = 22
    teams = [f"T{i // 2}" for i in range(n)]
    df = pd.DataFrame({"Driver": [f"D{i:02d}" for i in range(n)], "Team": teams})
    assert len(head_to_head_pairs(df, top_n=10)) <= 45 + 11 + 21


def test_works_without_a_team_column():
    df = pd.DataFrame({"Driver": ["A", "B", "C", "D"]})
    assert names(head_to_head_pairs(df, top_n=2)) == {frozenset(("A", "B")), frozenset(("B", "C")), frozenset(("C", "D"))}
