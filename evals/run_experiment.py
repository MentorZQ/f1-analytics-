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
    r"cannot (?:confirm|tell|verify)|can't (?:confirm|verify)|no data (?:about|on|for)|there is no", re.I)


# Only broad "why / where" questions are expected to touch the pair's biggest-gap sections.
COVERAGE_KINDS = {"pole", "pair", "teammates", "single_slow", "nameless"}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def declines(answer: str) -> bool:
    return bool(_DECLINE.search(answer[:900].replace("*", "").replace("_", " ")))


def build_evaluators():
    from evals.code_checks import run_all

    def code_check(name):
        def evaluator(output, expected=None, metadata=None):
            if not output or "answer" not in output:
                return {"score": 0.0, "label": "fail", "explanation": "no answer produced"}
            results = {r.name: r for r in run_all(output["answer"], output["sources"], output.get("deltas"))}
            r = results[name]
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
        got = {d for s in output.get("sources", []) for d in s.get("drivers", [])}
        missing = sorted(want - got)
        return {"score": 1.0 if not missing else 1 - len(missing) / len(want),
                "label": "pass" if not missing else "fail",
                "explanation": "all expected drivers retrieved" if not missing else f"no chunk for: {', '.join(missing)}"}

    def handles_unanswerable(output, expected=None):
        answerable = (expected or {}).get("answerable", True)
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

    names = ["uncertainty_compliance", "number_grounding", "total_consistency", "inference_rate",
             "retrieval_coverage", "answer_coverage"]
    fns = [code_check(n) for n in names] + [drivers_retrieved, handles_unanswerable, answers_when_answerable]
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
            outputs=[{"drivers": q["expected_drivers"], "answerable": q["answerable"]} for q in questions],
            metadata=[{"id": q["id"], "kind": q["kind"], "note": q["note"]} for q in questions],
            dataset_description="F1 qualifying RAG questions (answerable and unanswerable)",
        )

    def task(input):
        out = query.answer_with_sources(input["question"], session_key=input["session"],
                                        n_chunks=args.n_chunks, model=model)
        return {"answer": out["answer"], "session": out["session"], "sources": out["sources"], "deltas": out["deltas"],
                "metrics": out.get("metrics")}

    client.experiments.run_experiment(
        dataset=dataset, task=task, evaluators=build_evaluators(), experiment_name=args.name,
        experiment_metadata={"model": model, "n_chunks": args.n_chunks, "commit": git_commit(),
                             "chroma_path": args.chroma_path or "RAG_data_layers/chroma_store"},
    )


if __name__ == "__main__":
    main()
