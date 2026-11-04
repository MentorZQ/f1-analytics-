"""
Print the experiments on a dataset side by side: pass / fail counts per check.

    python -m evals.compare [--dataset f1-qualifying-questions-v1]
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

PRICES = json.loads((Path(__file__).resolve().parent / "prices.json").read_text())


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def performance(task_runs: list, model: str | None) -> dict | None:
    """Latency, tokens and (when the model's price is known) cost, from the metrics stored with each answer."""
    ms = [(tr["output"] or {}).get("metrics") for tr in task_runs]
    ms = [m for m in ms if m]
    if not ms:
        return None
    total = [m["total_s"] for m in ms]
    tin = [m["input_tokens"] for m in ms if m.get("input_tokens") is not None]
    tout = [m["output_tokens"] for m in ms if m.get("output_tokens") is not None]
    out = {"n": len(ms), "total_p50": statistics.median(total), "total_p95": percentile(total, 0.95),
           "llm_mean": statistics.mean(m["llm_s"] for m in ms), "retrieve_mean": statistics.mean(m["retrieve_s"] for m in ms),
           "in_mean": statistics.mean(tin) if tin else None, "out_mean": statistics.mean(tout) if tout else None,
           "in_total": sum(tin), "out_total": sum(tout), "cost_total": None, "cost_per_q": None}
    price = PRICES.get(model or "")
    if price and tin and tout:
        out["cost_total"] = out["in_total"] / 1e6 * price["input_per_mtok"] + out["out_total"] / 1e6 * price["output_per_mtok"]
        out["cost_per_q"] = out["cost_total"] / len(ms)
    return out


def summarise(client, dataset_name: str) -> dict:
    ds = client.datasets.get_dataset(dataset=dataset_name)
    out = {}
    for info in client.experiments.list(dataset_id=ds.id):
        ex = client.experiments.get_experiment(experiment_id=info["id"])
        meta = ex["experiment_metadata"] or {}
        latest = {}                                        # newest verdict per (question, check)
        for r in ex["evaluation_runs"]:
            latest[(r.experiment_run_id, r.name)] = r
        agg = collections.defaultdict(list)
        for (_, name), r in latest.items():
            res = r.result or {}
            agg[name].append((res.get("label"), res.get("score")))
        out[info["id"]] = {"name": info.get("name") or meta.get("name"), "meta": meta,
                           "n": len(ex["task_runs"]), "agg": agg,
                           "perf": performance(ex["task_runs"], meta.get("model"))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:6006")
    ap.add_argument("--dataset", default="f1-qualifying-questions-v1")
    ap.add_argument("--min-questions", type=int, default=10, help="hide smoke runs smaller than this")
    args = ap.parse_args()
    from phoenix.client import Client
    runs = {k: v for k, v in summarise(Client(base_url=args.endpoint), args.dataset).items() if v["n"] >= args.min_questions}
    names = sorted({n for v in runs.values() for n in v["agg"]})
    cols = [(v["meta"].get("chroma_path", "?").split("/")[-1], v) for v in runs.values()]
    print(f"{'':26s}" + "".join(f"{c[0]:>22s}" for c in cols))
    for n in names:
        row = f"{n:26s}"
        for _, v in cols:
            labels = [l for l, _ in v["agg"].get(n, [])]
            p, f = labels.count("pass"), labels.count("fail")
            scores = [s for _, s in v["agg"].get(n, []) if s is not None]
            row += f"{(f'{p} pass / {f} fail' if p + f else f'mean {statistics.mean(scores):.2f}' if scores else 'n/a'):>22s}"
        print(row)

    print("\nSpeed and spend (per question):")
    rows = [("latency p50 (s)", "total_p50", "{:.1f}"), ("latency p95 (s)", "total_p95", "{:.1f}"),
            ("  of which Claude call (mean s)", "llm_mean", "{:.1f}"), ("  of which retrieval (mean s)", "retrieve_mean", "{:.2f}"),
            ("input tokens (mean)", "in_mean", "{:,.0f}"), ("output tokens (mean)", "out_mean", "{:,.0f}"),
            ("tokens in/out (total)", None, None), ("cost per question ($)", "cost_per_q", "{:.4f}"), ("cost, whole run ($)", "cost_total", "{:.2f}")]
    for label, key, fmt in rows:
        row = f"{label:34s}"
        for _, v in cols:
            perf = v.get("perf")
            if perf is None:
                cell = "n/a (not recorded)"
            elif key is None:
                cell = f"{perf['in_total']:,} / {perf['out_total']:,}"
            elif perf.get(key) is None:
                cell = "n/a (no price)" if key.startswith("cost") else "n/a"
            else:
                cell = fmt.format(perf[key])
            row += f"{cell:>22s}"
        print(row)


if __name__ == "__main__":
    main()
