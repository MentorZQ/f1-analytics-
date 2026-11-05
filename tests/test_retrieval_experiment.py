import sys
from pathlib import Path

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
