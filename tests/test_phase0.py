"""Phase 0: session nicknames, team names, the headline lap comparison, and the session-context key fix."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RAG_data_layers/F1 Project Thoughts"))

import query  # noqa: E402
import rag_retrieval as rr  # noqa: E402

CTX = {
    "drivers": {
        "RUS": {"lap_time_seconds": 78.518, "qualifying_position": 1, "team": "Mercedes"},
        "ANT": {"lap_time_seconds": 78.811, "qualifying_position": 2, "team": "Mercedes"},
        "HAD": {"lap_time_seconds": 79.303, "qualifying_position": 3, "team": "Red Bull Racing"},
        "LEC": {"lap_time_seconds": 79.327, "qualifying_position": 4, "team": "Ferrari"},
        "HAM": {"lap_time_seconds": 79.478, "qualifying_position": 7, "team": "Ferrari"},
        "LAW": {"lap_time_seconds": 79.994, "qualifying_position": 8, "team": "Racing Bulls"},
        "LIN": {"lap_time_seconds": 79.971, "qualifying_position": 9, "team": "Racing Bulls"},
        "NOR": {"lap_time_seconds": 79.475, "qualifying_position": 6, "team": "McLaren"},
        "PIA": {"lap_time_seconds": 79.380, "qualifying_position": 5, "team": "McLaren"},
    },
    "standings": ["RUS", "ANT", "HAD", "LEC", "PIA", "NOR", "HAM", "LIN", "LAW"],
    "teammate_of": {"RUS": "ANT", "ANT": "RUS", "LEC": "HAM", "HAM": "LEC", "NOR": "PIA", "PIA": "NOR", "LAW": "LIN", "LIN": "LAW"},
    "pole": "RUS", "p2": "ANT",
}


# ── session nicknames ──

def test_session_nicknames_resolve():
    assert query.detect_session("Compare Isack and Max during Aussie quali") == "melbourne"
    assert query.detect_session("how did the Aus GP quali go") == "melbourne"
    assert query.detect_session("why were many drivers close together in barca") == "barcelona"
    assert query.detect_session("why did leclerc finish so poor in barc") == "barcelona"
    assert query.detect_session("lap at Albert Park") == "melbourne"
    assert query.detect_session("who was fastest at Monza in 2024") == "monza"


def test_short_nicknames_do_not_fire_inside_other_words():
    assert query.detect_session("what was the pause in the ozone of the pit lane") == "barcelona"   # default, not 'oz'
    assert query.detect_session("a barcode scanner at the race") == "barcelona"                     # default, not 'barc'


# ── team names ──

def test_team_names_resolve_to_their_drivers():
    assert rr.team_drivers("How quick were the ferraris in Australia", CTX) == {"LEC", "HAM"}
    assert rr.team_drivers("why was the mercedes pace so dominant", CTX) == {"RUS", "ANT"}
    assert rr.team_drivers("how did mclaren lock out the front row", CTX) == {"NOR", "PIA"}
    assert rr.team_drivers("how did the Red Bulls do", CTX) == {"HAD"}
    assert rr.team_drivers("Racing Bulls vs Ferrari", CTX) == {"LAW", "LIN", "LEC", "HAM"}


def test_red_bull_does_not_catch_racing_bulls_and_vice_versa():
    assert rr.team_drivers("how did the racing bulls do", CTX) == {"LAW", "LIN"}
    assert rr.team_drivers("how did red bull do", CTX) == {"HAD"}


def test_no_team_means_no_drivers_and_no_context_is_safe():
    assert rr.team_drivers("where did each team have an advantage", CTX) == set()
    assert rr.team_drivers("the ferraris", None) == set()


def test_question_drivers_combines_names_and_teams():
    assert rr.question_drivers("Hadjar vs the ferraris", CTX) == {"HAD", "LEC", "HAM"}
    assert rr.question_drivers("LEC vs HAM", CTX) == {"LEC", "HAM"}


# ── the session-context key fix ──

def test_a_driver_slower_than_their_teammate_is_compared_with_the_teammate():
    # HAM is 0.151s slower than LEC: below the 0.3s threshold, so pole. LAW vs LIN is 0.023: pole too.
    assert rr.resolve_context_drivers("Why was Hamilton slow", {"HAM"}, CTX) == {"HAM", "RUS"}
    slow = {**CTX, "drivers": {**CTX["drivers"], "HAM": {"lap_time_seconds": 80.2, "qualifying_position": 10, "team": "Ferrari"}}}
    assert rr.resolve_context_drivers("Why was Hamilton slow", {"HAM"}, slow) == {"HAM", "LEC"}      # >0.3s behind: teammate


def test_both_session_context_key_styles_work():
    old_style = {"drivers": {"HAM": {"position": 10, "lap_time_s": 80.2, "team": "Ferrari"},
                             "LEC": {"position": 4, "lap_time_s": 79.3, "team": "Ferrari"}},
                 "teammate_of": {"HAM": "LEC", "LEC": "HAM"}, "pole": "RUS", "p2": "ANT", "standings": []}
    assert rr.resolve_context_drivers("Why was Hamilton slow", {"HAM"}, old_style) == {"HAM", "LEC"}


# ── the headline lap comparison ──

def test_lap_comparison_states_both_laps_and_the_gap():
    c = query.lap_comparison_chunk({"driver_a": "RUS", "driver_b": "ANT"}, CTX)
    assert c["metadata"]["chunk_type"] == "lap_comparison" and c["rank"] == 0
    assert "RUS: best lap 1:18.518, classified P1" in c["text"]
    assert "ANT: best lap 1:18.811, classified P2" in c["text"]
    assert "RUS was 0.293s quicker than ANT" in c["text"]


def test_lap_comparison_notes_when_classification_and_best_lap_disagree():
    c = query.lap_comparison_chunk({"driver_a": "LAW", "driver_b": "LIN"}, CTX)          # LAW P8 but slower than LIN P9
    assert "LIN was 0.023s quicker than LAW" in c["text"]
    assert "classification order differs" in c["text"]
    assert "classification order differs" not in query.lap_comparison_chunk({"driver_a": "RUS", "driver_b": "ANT"}, CTX)["text"]


def test_lap_comparison_is_skipped_without_a_pair_or_a_lap_time():
    assert query.lap_comparison_chunk(None, CTX) is None
    assert query.lap_comparison_chunk({"driver_a": "RUS", "driver_b": "VER"}, CTX) is None       # VER set no lap


def test_lap_times_are_formatted_as_minutes_and_seconds():
    assert query._lap_text(78.518) == "1:18.518" and query._lap_text(74.679) == "1:14.679" and query._lap_text(61.0) == "1:01.000"
