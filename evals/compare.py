"""
Print the experiments on a dataset side by side: pass / fail counts per check.

    python -m evals.compare [--dataset f1-qualifying-questions-v1]
"""
import argparse
import collections
import statistics


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
                           "n": len(ex["task_runs"]), "agg": agg}
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


if __name__ == "__main__":
    main()
