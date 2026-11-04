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


# ── driver names: first names, nicknames and three-letter codes ──

def codes(q):
    return rr.extract_driver_codes(q)


def test_first_names_and_nicknames_resolve():
    assert codes("Max vs Lewis at Barcelona") == {"VER", "HAM"}
    assert codes("How did Charles do against Lewis?") == {"LEC", "HAM"}
    assert codes("Mad Max vs Checo") == {"VER", "PER"}
    assert codes("Kimi vs George") == {"ANT", "RUS"}
    assert codes("Hulkenberg versus the Hulk") == {"HUL"}


def test_three_letter_codes_resolve():
    assert codes("LEC vs HAM in Spain") == {"LEC", "HAM"}
    assert codes("Compare NOR and PIA") == {"NOR", "PIA"}
    assert codes("Why was BOT slow?") == {"BOT"}


def test_lowercase_codes_resolve_only_when_set_against_each_other():
    assert codes("lec vs ham") == {"LEC", "HAM"}
    assert codes("ver v nor at monza") == {"VER", "NOR"}
    assert codes("he had a big lap and a bot of luck") == set()      # ordinary words are not drivers
    assert codes("lap time per sector") == set()


def test_max_is_verstappen_only_when_it_is_a_name():
    assert codes("What was the max speed through the Lesmos?") == set()
    assert codes("max throttle application at T12") == set()
    assert codes("Why was Max slower than Lando?") == {"VER", "NOR"}
    assert codes("max vs lewis") == {"VER", "HAM"}


def test_normalize_query_does_not_attach_a_code_to_max_speed():
    assert "VER" not in rr.normalize_query("What was the max speed at Monza?")
    assert "VER" in rr.normalize_query("Why was Max slower?")


def test_max_needs_context():
    # the driver
    assert codes("Max's speed through the Lesmos") == {"VER"}
    assert codes("How fast was Max through the Lesmos?") == {"VER"}
    assert codes("Why was Max slow?") == {"VER"}
    assert codes("Lewis and max in Spain") == {"HAM", "VER"}
    assert codes("Max Verstappen's max speed") == {"VER"}
    # the quantity
    assert codes("What was the max speed through the Lesmos?") == set()
    assert codes("Max speed at Monza?") == set()
    assert codes("his max speed on the straight") == set()
    assert codes("What was the max?") == set()
    assert codes("Who was at max throttle longest?") == set()
    assert codes("Lewis's max speed vs Charles's max speed") == {"HAM", "LEC"}


def test_max_speed_and_maxs_speed_mean_different_things_in_the_query_text():
    assert "VER" in rr.normalize_query("What was Max's speed at Monza?")
    assert "VER" not in rr.normalize_query("What was the max speed at Monza?")


# ── the lap comparison follows the drivers the question is about ──

class FakeCollection:
    def __init__(self):
        self.where = None

    def get(self, where=None, include=None):
        self.where = where
        return {"metadatas": [], "documents": []}                 # nothing stored for any pair


def h2h(a, b):
    return {"metadata": {"chunk_type": "head_to_head_event", "driver_a": a, "driver_b": b, "event_id": "T1"}}


def pair_in(where):
    flat = str(where)
    return {code for code in ("VER", "PER", "NOR") if code in flat}


def test_the_comparison_is_for_the_wanted_pair_not_whatever_was_retrieved():
    sys.path.insert(0, str(ROOT))
    import query
    col = FakeCollection()
    assert query.section_deltas(col, [h2h("VER", "PER")], wanted={"VER", "NOR"}) is None   # no stored VER-NOR pair
    assert pair_in(col.where) == {"VER", "NOR"}                                            # it looked up VER-NOR, not VER-PER


def test_no_named_drivers_means_no_comparison_table():
    import query
    col = FakeCollection()
    assert query.section_deltas(col, [h2h("VER", "PER")], wanted=set()) is None
    assert col.where is None                                                               # nothing was even looked up


def test_without_wanted_the_old_behaviour_is_kept():
    import query
    col = FakeCollection()
    query.section_deltas(col, [h2h("VER", "PER")])
    assert pair_in(col.where) == {"VER", "PER"}
