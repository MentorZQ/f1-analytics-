"""Offline tests for the dataset-level evaluators and the question set used in Phoenix experiments."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import run_experiment as rx  # noqa: E402


def by_name():
    return {f.__name__: f for f in rx.build_evaluators()}


OUT = {"answer": "Russell was 0.125s faster in T12.", "sources": [{"drivers": ["RUS", "HAM"], "section": "T12", "text": "RUS 10.000s | HAM 10.125s"}],
       "deltas": None}


def test_every_evaluator_has_its_own_name():
    names = [f.__name__ for f in rx.build_evaluators()]
    assert len(names) == len(set(names)) == 9


def test_drivers_retrieved_passes_and_fails():
    ev = by_name()["drivers_retrieved"]
    assert ev(OUT, {"drivers": ["RUS", "HAM"]})["label"] == "pass"
    r = ev(OUT, {"drivers": ["RUS", "NOR"]})
    assert r["label"] == "fail" and "NOR" in r["explanation"]
    assert ev(OUT, {"drivers": []})["label"] == "n/a"


def test_unanswerable_questions_must_be_declined():
    ev = by_name()["handles_unanswerable"]
    declined = {**OUT, "answer": "The retrieved context does not contain race results, only qualifying data."}
    invented = {**OUT, "answer": "Russell won the race from pole after a clean start."}
    assert ev(declined, {"answerable": False})["label"] == "pass"
    assert ev(invented, {"answerable": False})["label"] == "fail"
    assert ev(invented, {"answerable": True})["label"] == "n/a"


def test_answerable_questions_must_not_be_refused():
    ev = by_name()["answers_when_answerable"]
    refused = {**OUT, "answer": "The context does not contain enough information."}
    assert ev(refused, {"answerable": True})["label"] == "fail"
    assert ev(OUT, {"answerable": True})["label"] == "pass"
    assert ev(refused, {"answerable": False})["label"] == "n/a"


def test_code_check_evaluators_return_phoenix_results():
    r = by_name()["number_grounding"](OUT, {"answerable": True})
    assert r["label"] in {"pass", "fail", "n/a"} and "explanation" in r


def test_question_set_is_well_formed():
    qs = json.loads((ROOT / "evals" / "questions.json").read_text())
    assert len(qs) >= 30 and len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert q["session"] in {"barcelona", "melbourne", "monza"}
        assert q["question"].strip() and isinstance(q["answerable"], bool)
    assert sum(not q["answerable"] for q in qs) >= 5            # unanswerable cases are part of the set


def test_markdown_does_not_hide_a_decline():
    ev = by_name()["handles_unanswerable"]
    bold = {**OUT, "answer": "The retrieved context does **not contain** any data about a crash."}
    only = {**OUT, "answer": "The retrieved context only contains weather data for the qualifying session. I cannot confirm race-day conditions."}
    assert ev(bold, {"answerable": False})["label"] == "pass"
    assert ev(only, {"answerable": False})["label"] == "pass"


def test_coverage_only_applies_to_broad_questions():
    ev = by_name()["retrieval_coverage"]
    out = {**OUT, "deltas": {"lap_gap_s": 0.06, "sections": [{"section": "T1", "gain_s": 0.3, "retrieved": False},
                                                              {"section": "T2", "gain_s": 0.2, "retrieved": False},
                                                              {"section": "T3", "gain_s": 0.1, "retrieved": False}]}}
    assert ev(out, {"answerable": True}, {"kind": "pair"})["label"] == "fail"
    assert ev(out, {"answerable": True}, {"kind": "section"})["label"] == "n/a"
    assert ev(out, {"answerable": False}, {"kind": "unanswerable"})["label"] == "n/a"


# ── latency and token tracking ──

def test_performance_summarises_latency_tokens_and_cost_when_priced():
    from evals import compare
    runs = [{"output": {"metrics": {"total_s": 10.0, "llm_s": 9.0, "retrieve_s": 0.5, "input_tokens": 2000, "output_tokens": 500}}},
            {"output": {"metrics": {"total_s": 20.0, "llm_s": 19.0, "retrieve_s": 0.5, "input_tokens": 3000, "output_tokens": 1000}}},
            {"output": None}]
    perf = compare.performance(runs, "claude-sonnet-5-5")
    assert perf["n"] == 2 and perf["total_p50"] == 15.0 and perf["in_total"] == 5000 and perf["out_total"] == 1500
    assert abs(perf["cost_total"] - (5000 / 1e6 * 2.0 + 1500 / 1e6 * 10.0)) < 1e-9
    assert compare.performance(runs, "claude-sonnet-4-6")["cost_total"] is None      # no documented price: no invented cost
    assert compare.performance([{"output": {"answer": "x"}}], "claude-sonnet-5-5") is None   # older runs had no metrics


def test_percentile_is_inclusive_of_the_slowest_request():
    from evals import compare
    assert compare.percentile([1, 2, 3, 4, 100], 0.95) == 100


def test_naming_question_set_is_well_formed():
    qs = json.loads((ROOT / "evals" / "questions_naming.json").read_text())
    assert len(qs) >= 10 and len({q["id"] for q in qs}) == len(qs)
    assert all(q["session"] in {"barcelona", "melbourne", "monza"} and q["kind"] == "naming" for q in qs)
    assert any(q["question"].startswith("lec vs") for q in qs) and any("Max vs" in q["question"] for q in qs)
