#!/usr/bin/env python3
"""
Offline evaluation of the lookup router: no model call, no key, no cost.

Runs the router over every question file against a temporary copy of the Chroma store (the repo's store is never
opened), then reports for each question kind how many questions it answered, how many of those contained every
expected value, and which one-fact questions it left to the model.

    venv/bin/python scripts/eval_router.py            # summary
    venv/bin/python scripts/eval_router.py --verbose  # every routed answer and every miss
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp = tempfile.mkdtemp(prefix="router_eval_")
shutil.copytree(ROOT / "RAG_data_layers" / "chroma_store", Path(_tmp) / "store")
os.environ["CHROMA_PATH"] = str(Path(_tmp) / "store")

import query                                  # noqa: E402  (after CHROMA_PATH is set)
from evals import code_checks                 # noqa: E402

# Kinds that ask for one fact; the router is expected to answer these.
ONE_FACT = {"data_lap_time", "data_position", "data_gap_to_pole", "data_speed_trap", "data_fastest_sector",
            "data_last_into_q3", "data_stage", "fresh_lookup"}


def missing_facts(answer: str, must: list[str]) -> list[str]:
    """The same matching the experiments use: 'a|b' is either spelling, 're:' is a pattern, case-insensitive."""
    text = answer.lower().replace("*", "").replace(",", "")

    def present(alt: str) -> bool:
        alt = alt.strip()
        return re.search(alt[3:], text, re.I) is not None if alt.startswith("re:") else alt.lower() in text

    return [m for m in must if not any(present(a) for a in m.split("|"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    stats = defaultdict(lambda: {"n": 0, "routed": 0, "right": 0, "no_expectation": 0})
    wrong, missed, check_fails, routed_total, seen = [], [], [], 0, 0
    for path in sorted(glob.glob(str(ROOT / "evals" / "questions*.json"))):
        for q in json.load(open(path)):
            kind, text = q.get("kind", "?"), q["question"]
            seen += 1
            s = stats[kind]
            s["n"] += 1
            result = query.route_question(text, q.get("session"))
            if result is None:
                if kind in ONE_FACT:
                    missed.append((kind, text))
                continue
            s["routed"] += 1
            routed_total += 1
            must = q.get("must_include") or []
            if not must:
                s["no_expectation"] += 1
            missing = missing_facts(result["answer"], must)
            if not missing:
                s["right"] += 1
            else:
                wrong.append((kind, text, result["answer"], missing))
            for c in code_checks.run_all(result["answer"], result["sources"], None, text, result["session"]):
                if getattr(c, "passed", None) is False or getattr(c, "label", "") == "fail":
                    check_fails.append((text, getattr(c, "name", "?"), getattr(c, "detail", "")))
            if args.verbose:
                print(f"[{result['metrics']['routed']}] {text}\n    -> {result['answer']}")

    print(f"\n{seen} questions, {routed_total} answered by the router (no model call)\n")
    print(f"{'kind':<24}{'asked':>6}{'routed':>8}{'right':>7}")
    for kind, s in sorted(stats.items()):
        if s["routed"] or kind in ONE_FACT:
            print(f"{kind:<24}{s['n']:>6}{s['routed']:>8}{s['right']:>7}" + ("   (no expected values for some)" if s["no_expectation"] else ""))
    print(f"\nrouted but missing an expected value: {len(wrong)}")
    for kind, text, answer, missing in wrong:
        print(f"  [{kind}] {text}\n      {answer}\n      missing: {missing}")
    print(f"\none-fact questions left to the model: {len(missed)}")
    for kind, text in missed:
        print(f"  [{kind}] {text}")
    print(f"\ncode checks failing on a routed answer: {len(check_fails)}")
    for text, name, detail in check_fails[:20]:
        print(f"  {name}: {text} {detail}")
    shutil.rmtree(_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
