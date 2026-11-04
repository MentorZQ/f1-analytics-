"""
Self-consistency: ask the same question several times and measure how much the answers agree.

    python -m evals.run_experiment --name consistency-1 --questions evals/questions_consistency.json --repetitions 3
    python -m evals.consistency --experiment consistency-1

One answer is one draw. If the same question gives the same figures, the same sections and the same
winners every time, a single run of the evaluation can be trusted more; if it does not, differences
between two runs of the pipeline may be noise rather than a real change.

Per question, across its repetitions:
  numbers      average pairwise overlap (Jaccard) of the numbers-with-units each answer states
  sections     overlap of the sections each answer discusses
  winners      of the section winners the answers name, the share on which they agree
  length       spread of answer length (coefficient of variation)
  flaky checks the checks whose pass/fail verdict changes between repetitions
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

from evals import claim_checks as cl
from evals import code_checks as cc

LOW_AGREEMENT = 0.6


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean_pairwise(sets: list[set]) -> float | None:
    if len(sets) < 2:
        return None
    return statistics.mean(jaccard(a, b) for a, b in itertools.combinations(sets, 2))


def number_set(answer: str) -> set:
    """Numbers with units, rounded so 152 and 152.0 are one number."""
    return {(round(v, 2), u) for v, u in cc.numbers_with_units(answer)}


def sections_discussed(answer: str, deltas: dict | None, sources: list[dict]) -> set:
    names = [s["section"] for s in (deltas or {}).get("sections", [])] or [s["section"] for s in sources if s.get("section")]
    starts = {s["section"]: s.get("start_m") for s in (deltas or {}).get("sections", [])}
    return {n for n in dict.fromkeys(names) if cc.section_mentioned(answer, n, starts.get(n))}


def winners_named(answer: str, deltas: dict | None) -> dict[str, str]:
    """{section: driver the answer says was faster}, for the sections whose claim can be read cleanly."""
    if not deltas or not deltas.get("sections") or not deltas.get("driver_a"):
        return {}
    a, b = deltas["driver_a"], deltas["driver_b"]
    out = {}
    for heading, unit in cl._units(answer):
        times = [v for v, u in cc.numbers_with_units(unit) if u == "s"]
        matched = [s for s in deltas["sections"] if abs(s["gain_s"]) >= 0.0005
                   and any(abs(t - abs(s["gain_s"])) <= cc.TOL["s"] for t in times)
                   and (cc.section_mentioned(unit, s["section"], s.get("start_m"))
                        or (heading and cc.section_mentioned(heading, s["section"], s.get("start_m"))))]
        if len(matched) == 1:
            claimed = cl.claimed_winner_for_time(unit, abs(matched[0]["gain_s"]), a, b)
            if claimed:
                out[matched[0]["section"]] = claimed
    return out


def winner_agreement(per_run: list[dict[str, str]]) -> tuple[float | None, int]:
    """Share of (section) winner claims, made in at least two runs, on which every claiming run agrees."""
    by_section: dict[str, list[str]] = {}
    for run in per_run:
        for sec, d in run.items():
            by_section.setdefault(sec, []).append(d)
    repeated = {sec: ds for sec, ds in by_section.items() if len(ds) >= 2}
    if not repeated:
        return None, 0
    agree = sum(len(set(ds)) == 1 for ds in repeated.values())
    return agree / len(repeated), len(repeated)


def analyse(question: str, session: str | None, runs: list[dict]) -> dict:
    """runs: [{"answer", "sources", "deltas"}] for one question."""
    answers = [r["answer"] for r in runs]
    nums = mean_pairwise([number_set(a) for a in answers])
    secs = mean_pairwise([sections_discussed(r["answer"], r.get("deltas"), r["sources"]) for r in runs])
    wins, n_claims = winner_agreement([winners_named(r["answer"], r.get("deltas")) for r in runs])
    lengths = [len(a) for a in answers]
    cv = statistics.pstdev(lengths) / statistics.mean(lengths) if len(lengths) > 1 and statistics.mean(lengths) else 0.0
    verdicts: dict[str, list[str]] = {}
    for r in runs:
        for res in cc.run_all(r["answer"], r["sources"], r.get("deltas"), question, session):
            if res.label in {"pass", "fail"}:
                verdicts.setdefault(res.name, []).append(res.label)
    flaky = sorted(n for n, labels in verdicts.items() if len(set(labels)) > 1)
    return {"question": question, "runs": len(runs), "numbers": nums, "sections": secs, "winners": wins,
            "winner_claims": n_claims, "length_cv": round(cv, 3), "flaky_checks": flaky}


def summarise(rows: list[dict]) -> dict:
    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.mean(vals) if vals else None
    return {"questions": len(rows), "numbers": avg("numbers"), "sections": avg("sections"), "winners": avg("winners"),
            "questions_with_flaky_checks": sum(bool(r["flaky_checks"]) for r in rows),
            "low_agreement": [r["question"] for r in rows
                              if any(r[k] is not None and r[k] < LOW_AGREEMENT for k in ("numbers", "sections", "winners"))]}


def load_runs(client, dataset: str, experiment_name: str) -> tuple[dict, dict]:
    """Group a Phoenix experiment's answers by question: {example_id: [run, ...]}, {example_id: input}."""
    ds = client.datasets.get_dataset(dataset=dataset)
    inputs = {e["id"]: e["input"] for e in ds.examples}
    ex = None
    for info in client.experiments.list(dataset_id=ds.id):
        candidate = client.experiments.get_experiment(experiment_id=info["id"])
        if (candidate.get("experiment_metadata") or {}).get("name") == experiment_name:
            ex = candidate                                  # the newest experiment with this name wins
    if ex is None:
        raise SystemExit(f"No experiment named {experiment_name!r} on dataset {dataset!r}.")
    grouped: dict[str, list[dict]] = {}
    for tr in ex["task_runs"]:
        out = tr.get("output")
        if out and out.get("answer"):
            grouped.setdefault(tr["dataset_example_id"], []).append(out)
    return grouped, inputs


def fmt(x) -> str:
    return "  n/a" if x is None else f"{x:5.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--dataset", default="f1-questions-consistency-v1")
    ap.add_argument("--endpoint", default="http://localhost:6006")
    ap.add_argument("--json", help="write the per-question results here")
    args = ap.parse_args()
    from phoenix.client import Client
    grouped, inputs = load_runs(Client(base_url=args.endpoint), args.dataset, args.experiment)
    rows = [analyse(inputs[eid]["question"], inputs[eid].get("session"), runs) for eid, runs in grouped.items() if len(runs) >= 2]
    print(f"{'question':58s} runs numbers sections winners  len-cv  flaky checks")
    for r in rows:
        print(f"{r['question'][:57]:58s} {r['runs']:4d}  {fmt(r['numbers'])}   {fmt(r['sections'])}   {fmt(r['winners'])}   {r['length_cv']:5.2f}  {', '.join(r['flaky_checks']) or '-'}")
    s = summarise(rows)
    print(f"\nOverall over {s['questions']} questions: numbers {fmt(s['numbers'])}, sections {fmt(s['sections'])}, winners {fmt(s['winners'])}; "
          f"{s['questions_with_flaky_checks']} question(s) had a check that flipped between runs.")
    if s["low_agreement"]:
        print("Below", LOW_AGREEMENT, "agreement:", "; ".join(q[:60] for q in s["low_agreement"]))
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "summary": s}, indent=2))


if __name__ == "__main__":
    main()
