"""
Run the retrieval and context-building half of the pipeline through Phoenix, with NO model call (free).

    python -m evals.run_retrieval_experiment --name retrieval-final               # every question set
    python -m evals.run_retrieval_experiment --name retrieval-final --only fresh --trace

For each question it builds exactly what the model would be given (the evidence blocks, the grid, the pair sections, the notes)
and scores that: are the expected drivers in it, and are the expected facts (a lap time, a gap, a position, a driver code) in
it? A fact missing from the context cannot be in a correct answer, so this is an upper bound on answer quality that costs nothing
and can be re-run after every retrieval change. Experiments go on the same datasets as the answer experiments, so Phoenix
shows them side by side. --trace also records the pipeline's own spans (f1.query, f1.retrieve) under the project
"f1-rag-retrieval".
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = Path(__file__).resolve().parent

SETS = {   # key: (question file, dataset name used by the answer experiments)
    "main": ("questions.json", "f1-qualifying-questions-v1"),
    "naming": ("questions_naming.json", "f1-questions-naming-v1"),
    "honesty": ("questions_honesty.json", "f1-questions-honesty-v1"),
    "consistency": ("questions_consistency.json", "f1-questions-consistency-v1"),
    "heldout_user": ("questions_heldout_user.json", "f1-questions-heldout-user-v1"),
    "generated": ("questions_heldout_data.json", "heldout-data-v7"),
    "fresh": ("questions_heldout_fresh.json", "heldout-fresh-v1"),
    "safety": ("questions_safety.json", "safety-v1b"),
}


def present(alternative: str, text: str) -> bool:
    alt = alternative.strip()
    if alt.startswith("re:"):
        return re.search(alt[3:], text, re.I) is not None
    return alt.lower() in text.lower()


def build_evaluators():
    def context_text(output):
        return "\n".join(s.get("text") or "" for s in (output or {}).get("sources", [])).replace(",", "")

    def drivers_in_context(output, expected=None):
        want = set((expected or {}).get("drivers") or [])
        if not want:
            return {"label": "n/a", "explanation": "no drivers expected"}
        text = context_text(output)
        got = {d for s in output.get("sources", []) for d in s.get("drivers", [])} | {d for d in want if re.search(rf"\b{d}\b", text)}
        missing = sorted(want - got)
        return {"score": 1.0 - len(missing) / len(want), "label": "pass" if not missing else "fail",
                "explanation": "all expected drivers in the context" if not missing else f"not in the context: {', '.join(missing)}"}

    def facts_in_context(output, expected=None):
        """Every fact a correct answer must state is somewhere in the evidence (an upper bound on answer quality)."""
        must = (expected or {}).get("must_include") or []
        if not must:
            return {"label": "n/a", "explanation": "no required facts"}
        text = context_text(output)
        missing = [m for m in must if not any(present(a, text) for a in m.split("|"))]
        return {"score": 1 - len(missing) / len(must), "label": "pass" if not missing else "fail",
                "explanation": "every expected fact is in the context" if not missing else "not in the context: " + ", ".join(missing)}

    def session_detected(output, input=None, metadata=None):
        expected = (metadata or {}).get("expected_session")
        if not expected or (input or {}).get("session"):
            return {"label": "n/a", "explanation": "session given, or none expected"}
        got = output.get("session")
        return {"score": float(got == expected), "label": "pass" if got == expected else "fail", "explanation": f"detected {got}, expected {expected}"}

    def context_size(output):
        n, chars = len(output.get("sources", [])), sum(len(s.get("text") or "") for s in output.get("sources", []))
        return {"score": float(n), "label": "info", "explanation": f"{n} evidence blocks, {chars} characters"}

    def pair_sections_present(output, expected=None):
        """A two-driver question should come with section-by-section evidence for that pair."""
        drivers = (expected or {}).get("drivers") or []
        if len(drivers) != 2 or not (expected or {}).get("answerable", True):
            return {"label": "n/a", "explanation": "not a two-driver question"}
        have = any(s.get("chunk_type") == "head_to_head_event" for s in output.get("sources", []))
        return {"score": float(have), "label": "pass" if have else "fail", "explanation": "pair sections in the context" if have else "no pair sections"}

    return [drivers_in_context, facts_in_context, session_detected, context_size, pair_sections_present]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--only", help="comma-separated sets: " + ", ".join(SETS))
    ap.add_argument("--endpoint", default="http://localhost:6006")
    ap.add_argument("--trace", action="store_true", help="also record f1.query / f1.retrieve spans (project f1-rag-retrieval)")
    args = ap.parse_args()

    os.environ.pop("ANTHROPIC_API_KEY", None)                 # a real model call is impossible from here
    os.environ.pop("PHOENIX_COLLECTOR_ENDPOINT", None)          # inside an experiment the runner has its own tracer and swallows ours
    os.environ["PHOENIX_PROJECT_NAME"] = "f1-rag-retrieval"
    work = Path(tempfile.mkdtemp(prefix="chroma_retrieval_")) / "store"
    shutil.copytree(ROOT / "RAG_data_layers" / "chroma_store", work)
    os.environ["CHROMA_PATH"] = str(work)
    sys.path.insert(0, str(ROOT))
    import query
    from phoenix.client import Client

    query.call_llm = lambda prompt, **kw: "[context only: no model was called]"
    client = Client(base_url=args.endpoint)
    keys = [k.strip() for k in args.only.split(",")] if args.only else list(SETS)
    for key in keys:
        filename, dataset_name = SETS[key]
        questions = json.loads((EVALS / filename).read_text())
        try:
            dataset = client.datasets.get_dataset(dataset=dataset_name)
        except Exception:
            dataset = client.datasets.create_dataset(
                name=dataset_name,
                inputs=[{"question": q["question"], "session": q["session"]} for q in questions],
                outputs=[{"drivers": q["expected_drivers"], "answerable": q["answerable"], "false_premise": q.get("false_premise", False),
                          "must_include": q.get("must_include", []), "must_not_include": q.get("must_not_include", [])} for q in questions],
                metadata=[{"id": q["id"], "kind": q["kind"], "note": q["note"], "expected_session": q.get("expected_session")} for q in questions],
                dataset_description="F1 qualifying RAG questions")

        def task(input):
            out = query.answer_with_sources(input["question"], session_key=input["session"], n_chunks=6, model="context-only")
            return {"session": out["session"], "sources": out["sources"], "deltas": out["deltas"], "metrics": out.get("metrics")}

        print(f"[{key}] {len(questions)} questions on dataset {dataset_name}")
        client.experiments.run_experiment(dataset=dataset, task=task, evaluators=build_evaluators(), experiment_name=f"{args.name}-{key}",
                                          experiment_metadata={"name": f"{args.name}-{key}", "kind": "context-only (no model)", "set": key})
    if args.trace:
        traced_pass(keys, args.endpoint)


def traced_pass(keys: list[str], endpoint: str) -> None:
    """The same questions through the pipeline in a plain loop, so each one becomes a trace: f1.query with f1.retrieve inside."""
    import time
    import query
    import tracing
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = endpoint
    tracing._initialised = False                    # it initialised itself as "off" during the experiments; start it again, now enabled
    tracing._tracer = None
    n = 0
    for key in keys:
        for q in json.loads((EVALS / SETS[key][0]).read_text()):
            query.answer_with_sources(q["question"], session_key=q["session"], n_chunks=6, model="context-only")
            n += 1
    time.sleep(3)
    print(f"traced pass: {n} questions recorded in the Phoenix project 'f1-rag-retrieval'")


if __name__ == "__main__":
    main()
