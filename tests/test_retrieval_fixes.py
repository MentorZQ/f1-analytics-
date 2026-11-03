"""Offline tests for the two retrieval fixes: nameless pole questions and guaranteed drivers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "RAG_data_layers/F1 Project Thoughts"))

import rag_retrieval as rr  # noqa: E402

CTX = {"standings": ["RUS", "HAM", "ANT", "NOR"], "pole": "RUS", "p2": "HAM",
       "drivers": {"RUS": {"position": 1}, "HAM": {"position": 2}}, "teammate_of": {"RUS": "HAM"}}


def test_nameless_pole_question_compares_pole_with_p2():
    for q in ("Why did the pole sitter beat the field in Spain?", "What made the pole lap quick in Australia?",
              "Who took pole?"):
        assert rr.resolve_context_drivers(q, set(), CTX) == {"RUS", "HAM"}


def test_other_nameless_positions_are_unchanged():
    assert rr.resolve_context_drivers("Who was second?", set(), CTX) == {"HAM"}
    assert rr.resolve_context_drivers("Who finished last?", set(), CTX) == {"NOR"}


def test_named_questions_are_unchanged():
    assert rr.resolve_context_drivers("Russell vs Hamilton", {"RUS", "HAM"}, CTX) == {"RUS", "HAM"}


def chunk(drivers, dist, ctype="telemetry_corner_event", delta=None):
    meta = {"chunk_type": ctype}
    if len(drivers) == 2:
        meta.update(driver_a=drivers[0], driver_b=drivers[1])
    else:
        meta["driver"] = drivers[0]
    if delta is not None:
        meta["time_delta_s"] = delta
    return {"metadata": meta, "distance": dist, "chunk_id": str(drivers) + str(dist)}


def test_a_driver_missing_from_the_top_n_is_swapped_in():
    ranked = [chunk(["HUL"], 0.1), chunk(["HUL"], 0.2), chunk(["HUL"], 0.3), chunk(["ALO"], 0.9)]
    final = rr.ensure_drivers_present(ranked, ranked, {"ALO", "HUL"}, n=3)
    assert len(final) == 3
    assert any("ALO" in rr._chunk_drivers(c) for c in final)
    assert any("HUL" in rr._chunk_drivers(c) for c in final)


def test_the_swap_gives_up_a_duplicate_not_the_only_chunk_of_another_needed_driver():
    ranked = [chunk(["HUL"], 0.1), chunk(["HUL"], 0.15), chunk(["ANT"], 0.2), chunk(["ALO"], 0.9)]
    final = rr.ensure_drivers_present(ranked, ranked, {"ALO", "HUL", "ANT"}, n=3)
    assert {d for c in final for d in rr._chunk_drivers(c)} == {"ALO", "HUL", "ANT"}


def test_nothing_changes_when_everyone_is_present():
    ranked = [chunk(["RUS", "HAM"], 0.1, "head_to_head_event", 0.2), chunk(["RUS"], 0.2), chunk(["HAM"], 0.3)]
    assert rr.ensure_drivers_present(ranked, ranked, {"RUS", "HAM"}, n=3) == ranked


def test_a_driver_absent_from_the_whole_pool_is_left_out_without_error():
    ranked = [chunk(["HUL"], 0.1), chunk(["HUL"], 0.2)]
    assert rr.ensure_drivers_present(ranked, ranked, {"ALO"}, n=2) == ranked


def test_a_missing_driver_is_never_filled_with_a_pair_involving_someone_else():
    stroll_pair = chunk(["ALO", "STR"], 0.2, "head_to_head_event", 0.3)
    ranked = [chunk(["HUL"], 0.1), chunk(["HUL"], 0.15), stroll_pair, chunk(["ALO"], 0.9)]
    final = rr.ensure_drivers_present(ranked, ranked, {"ALO", "HUL"}, n=2)
    assert stroll_pair not in final
    assert any(rr._chunk_drivers(c) == {"ALO"} for c in final)
