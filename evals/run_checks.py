"""Run the code checks over the saved examples (web/examples.json) and print a table.

    python -m evals.run_checks [--json out.json] [--verbose]
"""
import argparse
import json
from pathlib import Path

from evals.code_checks import run_all

EXAMPLES = Path(__file__).resolve().parent.parent / "web" / "examples.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--examples", default=str(EXAMPLES))
    args = ap.parse_args()
    rows = []
    for ex in json.loads(Path(args.examples).read_text()):
        for r in run_all(ex["answer"], ex["sources"], ex.get("deltas"), ex["question"], ex.get("session")):
            rows.append({"question": ex["question"], **r.as_dict()})
    names = list(dict.fromkeys(r["name"] for r in rows))
    print(f"{'question':48s} " + " ".join(f"{n[:19]:19s}" for n in names))
    for q in dict.fromkeys(r["question"] for r in rows):
        cells = {r["name"]: r for r in rows if r["question"] == q}
        print(f"{q[:47]:48s} " + " ".join(f"{(cells[n]['label'] + ('' if cells[n]['score'] is None else f' {cells[n][chr(115)+chr(99)+chr(111)+chr(114)+chr(101)]:.2f}')):19s}" for n in names))
    if args.verbose:
        for r in rows:
            print(f"\n[{r['label'].upper()}] {r['question']} / {r['name']}\n    {r['explanation']}")
            if r["label"] == "fail":
                print("    ", json.dumps(r["details"], ensure_ascii=False)[:600])
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
