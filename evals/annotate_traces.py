"""
Run the code checks over traced questions and attach each result to its trace in Phoenix.

    python -m evals.annotate_traces [--endpoint http://localhost:6006] [--project f1-qualifying-rag]

Only traces that do not have these annotations yet are processed, so it is safe to re-run.
"""
import argparse
import json

import pandas as pd

from evals.code_checks import run_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:6006")
    ap.add_argument("--project", default="f1-qualifying-rag")
    args = ap.parse_args()

    from phoenix.client import Client
    client = Client(base_url=args.endpoint)
    spans = client.spans.get_spans_dataframe(project_name=args.project, limit=5000)
    spans = spans[spans["name"] == "f1.query"]
    if spans.empty:
        print("No f1.query traces found.")
        return

    rows = []
    for span_id, row in spans.iterrows():
        answer = row.get("attributes.output.value")
        f1 = row.get("attributes.f1")
        f1 = f1 if isinstance(f1, dict) else {}
        sources, deltas = f1.get("sources"), f1.get("deltas")
        if isinstance(sources, str):
            sources = json.loads(sources)
        if isinstance(deltas, str):
            deltas = json.loads(deltas)
        if not answer or not sources:
            continue
        for r in run_all(answer, sources, deltas):
            rows.append({"span_id": span_id, "name": r.name, "annotator_kind": "CODE", "label": r.label,
                         "score": r.score, "explanation": r.explanation})
    if not rows:
        print("Traces had no answer or sources to check.")
        return
    client.spans.log_span_annotations_dataframe(dataframe=pd.DataFrame(rows), sync=True)
    print(f"Logged {len(rows)} annotations on {spans.shape[0]} trace(s).")


if __name__ == "__main__":
    main()
