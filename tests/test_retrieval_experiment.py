import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import run_retrieval_experiment as rr  # noqa: E402

EV = {f.__name__: f for f in rr.build_evaluators()}
OUT = {"session": "monza", "sources": [{"chunk_type": "lap_comparison", "drivers": ["NOR", "PIA"],
                                         "text": "NOR: best lap 1:19.327, classified P1\nPIA: best lap 1:19.436, classified P2\nOver the lap NOR was 0.109s quicker"}]}


def test_the_facts_must_be_in_the_context_to_pass():
    ok = EV["facts_in_context"](OUT, {"must_include": ["1:19.327|79.327", "0.109", "re:\\bNOR\\b|re:\\bnorris\\b"]})
    assert ok["label"] == "pass"
    bad = EV["facts_in_context"](OUT, {"must_include": ["0.463"]})
    assert bad["label"] == "fail" and "0.463" in bad["explanation"]
    assert EV["facts_in_context"](OUT, {})["label"] == "n/a"


def test_drivers_and_pair_sections_are_checked():
    assert EV["drivers_in_context"](OUT, {"drivers": ["NOR", "PIA"]})["label"] == "pass"
    assert EV["drivers_in_context"](OUT, {"drivers": ["NOR", "HAM"]})["label"] == "fail"
    assert EV["pair_sections_present"](OUT, {"drivers": ["NOR", "PIA"]})["label"] == "fail"           # only the headline block, no sections
    assert EV["pair_sections_present"](OUT, {"drivers": ["NOR"]})["label"] == "n/a"


def test_the_runner_removes_the_api_key_before_it_imports_the_pipeline():
    import inspect
    src = inspect.getsource(rr.main)
    assert 'os.environ.pop("ANTHROPIC_API_KEY"' in src and src.index('os.environ.pop("ANTHROPIC_API_KEY"') < src.index("import query")


def test_the_spot_check_file_is_consistent_and_the_agreement_numbers_add_up():
    import json
    from evals import upload_spotcheck as up
    items = json.loads((ROOT / "evals" / "spotcheck_verdicts.json").read_text())
    assert len(items) == len({i["id"] for i in items}) == 13
    assert all(i["human_verdict"] in {"flag_right", "flag_wrong", "clean_ok", "clean_with_issues"} for i in items)
    assert all((i["human_verdict"] == "flag_right") == bool(i["check_flags"]) for i in items)        # a flagged answer was judged right
    a = up.agreement(items)
    assert (a["answers"], a["flagged"], a["flags_right"], a["clean_ok"]) == (13, 2, 2, 1)
    assert a["human_wanted_a_change"] == 12 and a["of_those_flagged"] == 2
    assert a["correctness_problems"] == 4 and a["correctness_flagged"] == 2


def test_every_pipeline_stage_is_a_span_with_its_facts(monkeypatch):
    import contextlib

    import query
    import tracing

    recorded = []

    class FakeSpan:
        def __init__(self, name):
            self.name, self.attrs = name, {}
            recorded.append(self)

        def set_attribute(self, key, value):
            self.attrs[key] = value

    class FakeTracer:
        @contextlib.contextmanager
        def start_as_current_span(self, name):
            yield FakeSpan(name)

    monkeypatch.setattr(tracing, "_tracer", FakeTracer())
    monkeypatch.setattr(tracing, "_initialised", True)
    monkeypatch.setattr(query, "ROUTER_ENABLED", False)          # this tests the model path; the router would answer this question itself
    monkeypatch.setattr(query, "call_llm", lambda prompt, **kw: "First fact. Second fact. Third fact.")
    query.answer_with_sources("What was Bearman's best qualifying lap at Melbourne?", model="m")
    names = [s.name for s in recorded]
    assert names == ["f1.query", "f1.sanitize_question", "f1.understand_question", "f1.session_views", "f1.pair_sections",
                     "f1.retrieve", "f1.lap_comparison", "f1.build_prompt", "f1.generate", "f1.limit_answer"] or \
        [n for n in names if n != "f1.query"] == ["f1.sanitize_question", "f1.understand_question", "f1.session_views", "f1.pair_sections",
                                                  "f1.retrieve", "f1.lap_comparison", "f1.build_prompt", "f1.generate", "f1.limit_answer"]
    by = {s.name: s.attrs for s in recorded}
    assert by["f1.understand_question"]["openinference.span.kind"] == "CHAIN" and by["f1.generate"]["openinference.span.kind"] == "LLM"
    assert "BEA" in by["f1.understand_question"]["f1.drivers_used"] and by["f1.understand_question"]["f1.one_fact_question"] is True
    assert by["f1.build_prompt"]["f1.answer_style"] == "one fact" and by["f1.limit_answer"]["f1.characters_cut"] > 0
    assert by["f1.generate"]["llm.model_name"] == "m"


def _record_spans(monkeypatch):
    import contextlib

    import tracing

    recorded = []

    class FakeSpan:
        def __init__(self, name):
            self.name, self.attrs = name, {}
            recorded.append(self)

        def set_attribute(self, key, value):
            self.attrs[key] = value

    class FakeTracer:
        @contextlib.contextmanager
        def start_as_current_span(self, name):
            yield FakeSpan(name)

    monkeypatch.setattr(tracing, "_tracer", FakeTracer())
    monkeypatch.setattr(tracing, "_initialised", True)
    return recorded


def test_a_routed_question_is_a_short_trace_with_no_model_span(monkeypatch):
    import query

    recorded = _record_spans(monkeypatch)
    monkeypatch.setattr(query, "call_llm", lambda *a, **k: pytest.fail("the model must not be called for a routed question"))
    result = query.answer_with_sources("What was Bearman's best qualifying lap at Melbourne?", model="m")
    assert [s.name for s in recorded if s.name != "f1.query"] == ["f1.sanitize_question", "f1.route"]
    route = next(s for s in recorded if s.name == "f1.route")
    assert route.attrs["f1.routed"] is True and route.attrs["f1.route"] == "lap_time"
    assert result["metrics"]["routed"] == "lap_time" and result["metrics"]["model"] is None and result["metrics"]["input_tokens"] == 0


def test_a_question_the_router_declines_shows_the_decision_then_the_full_pipeline(monkeypatch):
    import query

    recorded = _record_spans(monkeypatch)
    monkeypatch.setattr(query, "call_llm", lambda prompt, **kw: "An answer.")
    query.answer_with_sources("Compare Russell and Hamilton at Barcelona.", model="m")
    names = [s.name for s in recorded if s.name != "f1.query"]
    assert names[:3] == ["f1.sanitize_question", "f1.route", "f1.understand_question"] and "f1.generate" in names
    assert next(s for s in recorded if s.name == "f1.route").attrs["f1.route"] == "model"
