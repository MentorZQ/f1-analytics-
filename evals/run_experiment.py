"""
Run the question set through the RAG pipeline as a Phoenix experiment, scored by the code checks.

    python -m evals.run_experiment --name baseline            # all questions (uses the Anthropic key)
    python -m evals.run_experiment --name smoke --limit 3

Each run is stored in Phoenix next to earlier ones, so a change to the prompt, the chunk format or the
data can be compared question by question. Needs a running Phoenix server (default http://localhost:6006).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = Path(__file__).resolve().parent / "questions.json"      # --questions chooses another file

_DECLINE = re.compile(
    r"does not (?:contain|include|provide|have)|doesn't (?:contain|include|provide|have)|"
    r"not (?:available|included|contained|provided|in the (?:context|data|retrieved))|no (?:information|data|record)|"
    r"cannot (?:be answered|determine|say)|can't (?:be answered|say)|unable to|isn't (?:in|available)|"
    r"outside (?:the )?(?:scope|data)|only (?:covers|contains|includes|has)|"
    r"cannot (?:confirm|tell|verify)|can't (?:confirm|verify)|no data (?:about|on|for)|there is no|"
    r"\bcontains? (?:no|nothing)\b|\bno [\w ]{0,40}\bdata\b|\bnot present\b|\bnot (?:recorded|stored|measured)\b|"
    r"\b(?:cannot|can't|can not) answer\b|\bnot in the (?:retrieved|provided)\b", re.I)


# Only broad "why / where" questions are expected to touch the pair's biggest-gap sections.
COVERAGE_KINDS = {"pole", "pair", "teammates", "single_slow", "nameless"}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


_CORRECTION = re.compile(r"\w,\s*not\s+\w+(?:\s+\w+)?,|\bactually\b|\bin fact\b|\bnot\b[^.]{0,50}\bbut\b|\brather than\b|\binstead\b|\bwas set by\b|"
                         r"\bwas (?:on|in|from)\b[^.]{0,30}\bnot\b|\bnot (?:the|on|in|at)\b", re.I)


def declines(answer: str) -> bool:
    return bool(_DECLINE.search(answer[:900].replace("*", "").replace("_", " ")))


def build_evaluators():
    from evals import claim_checks
    from evals.code_checks import run_all

    def code_check(name):
        def evaluator(output, input=None, expected=None, metadata=None):
            if not output or "answer" not in output:
                return {"score": 0.0, "label": "fail", "explanation": "no answer produced"}
            question = (input or {}).get("question")
            session = (input or {}).get("session") or output.get("session")
            results = {r.name: r for r in run_all(output["answer"], output["sources"], output.get("deltas"), question, session)}
            r = results[name]
            unanswerable = (expected or {}).get("answerable") is False
            if name in {"completeness", "reversed_winner", "numbers_on_right_driver", "overclaiming"} and unanswerable:
                return {"label": "n/a", "explanation": "unanswerable question"}
            if name.endswith("coverage"):
                if (expected or {}).get("answerable") is False:
                    return {"label": "n/a", "explanation": "unanswerable question"}
                if (metadata or {}).get("kind") not in COVERAGE_KINDS:
                    return {"label": "n/a", "explanation": "narrow question: the biggest gaps are not what it asks about"}
            return {"score": r.score, "label": r.label, "explanation": r.explanation}
        evaluator.__name__ = evaluator.__qualname__ = name
        return evaluator

    def drivers_retrieved(output, expected=None):
        want = set((expected or {}).get("drivers") or [])
        if not want:
            return {"label": "n/a", "explanation": "no drivers expected"}
        sources = output.get("sources", [])
        got = {d for s in sources for d in s.get("drivers", [])}
        # A whole-field block (the grid, a section ranking) names drivers in its text without being "for" one driver:
        # a driver counts as present if his code appears as a word anywhere in the evidence.
        text = "\n".join(s.get("text") or "" for s in sources)
        got |= {d for d in want if re.search(rf"\b{d}\b", text)}
        missing = sorted(want - got)
        return {"score": 1.0 if not missing else 1 - len(missing) / len(want),
                "label": "pass" if not missing else "fail",
                "explanation": "all expected drivers retrieved" if not missing else f"no chunk for: {', '.join(missing)}"}

    def handles_unanswerable(output, expected=None):
        answerable = (expected or {}).get("answerable", True)
        if (expected or {}).get("false_premise"):
            return {"label": "n/a", "explanation": "false-premise question: judged by challenges_false_premise"}
        said_no = declines(output.get("answer", ""))
        if answerable:
            return {"label": "n/a", "explanation": "answerable question"}
        return {"score": 1.0 if said_no else 0.0, "label": "pass" if said_no else "fail",
                "explanation": "declined / said the data does not cover it" if said_no else "answered a question the data cannot answer"}

    def answers_when_answerable(output, expected=None):
        if not (expected or {}).get("answerable", True):
            return {"label": "n/a", "explanation": "unanswerable question"}
        text = output.get("answer", "")
        refused = declines(text) and len(text) < 500
        return {"score": 0.0 if refused else 1.0, "label": "fail" if refused else "pass",
                "explanation": "refused an answerable question" if refused else "gave a substantive answer"}

    def challenges_false_premise(output, expected=None):
        """Questions built on a wrong or unrecorded premise must be challenged, not explained."""
        if not (expected or {}).get("false_premise"):
            return {"label": "n/a", "explanation": "the question's premise is sound"}
        text = output.get("answer", "")[:1100].replace("*", "").replace("_", " ")
        ok = bool(claim_checks._CHALLENGE.search(text) or _CORRECTION.search(text) or declines(text))
        return {"score": 1.0 if ok else 0.0, "label": "pass" if ok else "fail",
                "explanation": "challenges or corrects the premise" if ok else "went along with a premise the data does not support"}

    def states_expected_facts(output, expected=None):
        """Known facts from the stored data that a correct answer must contain (case-insensitive)."""
        must = (expected or {}).get("must_include") or []
        if not must:
            return {"label": "n/a", "explanation": "no required facts"}
        text = output.get("answer", "").lower().replace("*", "").replace(",", "")
        # "a|b" means either spelling is fine ("1:19.327|79.327", "P10|10th|tenth")
        def present(alt: str) -> bool:
            alt = alt.strip()
            if alt.startswith("re:"):                        # "re:\\bp4\\b": a pattern, for facts that need a word boundary
                return re.search(alt[3:], text, re.I) is not None
            return alt.lower() in text

        missing = [m for m in must if not any(present(alt) for alt in m.split("|"))]
        return {"score": 1 - len(missing) / len(must), "label": "pass" if not missing else "fail",
                "explanation": "contains all required facts" if not missing else f"missing: {', '.join(missing)}"}

    def detects_session(output, input=None, metadata=None):
        """When no session is given, the pipeline must work out which circuit the question is about."""
        expected = (metadata or {}).get("expected_session")
        if not expected or (input or {}).get("session"):
            return {"label": "n/a", "explanation": "session was given, or none expected"}
        got = output.get("session")
        return {"score": 1.0 if got == expected else 0.0, "label": "pass" if got == expected else "fail",
                "explanation": f"detected {got}" if got == expected else f"detected {got}, expected {expected}"}

    names = ["uncertainty_compliance", "number_grounding", "total_consistency", "inference_rate",
             "retrieval_coverage", "answer_coverage",
             "reversed_winner", "self_correction", "wrong_driver_or_session", "numbers_on_right_driver",
             "completeness", "false_premise", "overclaiming", "team_attribution"]
    fns = [code_check(n) for n in names] + [drivers_retrieved, handles_unanswerable, answers_when_answerable,
                                            challenges_false_premise, states_expected_facts, detects_session]
    return fns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="experiment name, e.g. baseline")
    ap.add_argument("--endpoint", default="http://localhost:6006")
    ap.add_argument("--questions", default=str(QUESTIONS), help="question file (default: the main 40-question set)")
    ap.add_argument("--dataset", help="Phoenix dataset name (default derived from the question file)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-chunks", type=int, default=6)
    ap.add_argument("--repetitions", type=int, default=1, help="run every question this many times (for self-consistency)")
    ap.add_argument("--chroma-path", help="evaluate a different Chroma store (e.g. the old one)")
    args = ap.parse_args()

    import os
    import sys
    # Opening a Chroma store can rewrite its files (newer chromadb versions migrate them), so work on a
    # throwaway copy and leave the tracked store byte-for-byte as committed.
    import shutil
    import tempfile
    source_store = Path(args.chroma_path) if args.chroma_path else ROOT / "RAG_data_layers" / "chroma_store"
    work_store = Path(tempfile.mkdtemp(prefix="chroma_eval_")) / "store"
    shutil.copytree(source_store, work_store)
    os.environ["CHROMA_PATH"] = str(work_store)
    os.environ.pop("PHOENIX_COLLECTOR_ENDPOINT", None)          # the experiment records its own traces
    sys.path.insert(0, str(ROOT))
    import query
    from phoenix.client import Client

    model = args.model or os.environ.get("F1_MODEL", "claude-sonnet-4-6")
    questions_path = Path(args.questions)
    questions = json.loads(questions_path.read_text())
    dataset_name = args.dataset or ("f1-qualifying-questions-v1" if questions_path == QUESTIONS
                                    else f"f1-{questions_path.stem.replace('_', '-')}-v1")
    if args.limit:                                            # a smaller dataset of its own, never a cut of the full one
        questions = questions[: args.limit]
        dataset_name = f"{dataset_name}-first{args.limit}"
    client = Client(base_url=args.endpoint)
    try:
        dataset = client.datasets.get_dataset(dataset=dataset_name)
    except Exception:
        dataset = client.datasets.create_dataset(
            name=dataset_name,
            inputs=[{"question": q["question"], "session": q["session"]} for q in questions],
            outputs=[{"drivers": q["expected_drivers"], "answerable": q["answerable"],
                      "false_premise": q.get("false_premise", False), "must_include": q.get("must_include", [])}
                     for q in questions],
            metadata=[{"id": q["id"], "kind": q["kind"], "note": q["note"],
                       "expected_session": q.get("expected_session")} for q in questions],
            dataset_description="F1 qualifying RAG questions (answerable and unanswerable)",
        )

    def task(input):
        out = query.answer_with_sources(input["question"], session_key=input["session"],
                                        n_chunks=args.n_chunks, model=model)
        return {"answer": out["answer"], "session": out["session"], "sources": out["sources"], "deltas": out["deltas"],
                "metrics": out.get("metrics")}

    client.experiments.run_experiment(
        dataset=dataset, task=task, evaluators=build_evaluators(), experiment_name=args.name,
        repetitions=args.repetitions,
        experiment_metadata={"name": args.name, "model": model, "repetitions": args.repetitions, "n_chunks": args.n_chunks, "commit": git_commit(),
                             "chroma_path": args.chroma_path or "RAG_data_layers/chroma_store"},
    )


if __name__ == "__main__":
    main()
