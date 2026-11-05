"""
Put the hand-checked answers (evals/spotcheck_verdicts.json) into Phoenix as a dataset and print how the automatic checks did
against the human verdicts. Local only, no model call.

    python -m evals.upload_spotcheck
"""
import json
from pathlib import Path

FILE = Path(__file__).resolve().parent / "spotcheck_verdicts.json"
DATASET = "spotcheck-v1"


def agreement(items: list[dict]) -> dict:
    flagged = [i for i in items if i["check_flags"]]
    with_issue = [i for i in items if i["human_verdict"] != "clean_ok"]
    correctness = [i for i in items if i["issue_kind"] in ("correctness", "none") and i["human_verdict"] != "clean_ok"]
    return {"answers": len(items), "flagged": len(flagged), "flags_right": sum(i["human_verdict"] == "flag_right" for i in flagged),
            "human_wanted_a_change": len(with_issue), "of_those_flagged": sum(bool(i["check_flags"]) for i in with_issue),
            "correctness_problems": len(correctness), "correctness_flagged": sum(bool(i["check_flags"]) for i in correctness),
            "clean_ok": sum(i["human_verdict"] == "clean_ok" for i in items)}


def main() -> None:
    items = json.loads(FILE.read_text())
    from phoenix.client import Client
    client = Client(base_url="http://localhost:6006")
    try:
        client.datasets.get_dataset(dataset=DATASET)
        print(f"dataset {DATASET} already exists")
    except Exception:
        client.datasets.create_dataset(
            name=DATASET, inputs=[{"question": i["question"]} for i in items],
            outputs=[{"human_verdict": i["human_verdict"], "issue_kind": i["issue_kind"], "issues": i["issues"], "human_note": i["human_note"]} for i in items],
            metadata=[{"id": i["id"], "check_flags": i["check_flags"], "run": i["run"]} for i in items],
            dataset_description="Hand-checked answers: the human verdict beside what the automatic checks said")
        print(f"created dataset {DATASET} with {len(items)} examples")
    for k, v in agreement(items).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
