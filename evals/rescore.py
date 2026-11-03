"""
Re-run the current evaluators over experiments that already exist in Phoenix, without calling Claude again.
Use it after changing a check, so old and new runs are scored by the same rules.

    python -m evals.rescore                 # every experiment on the default dataset
    python -m evals.rescore --dataset f1-qualifying-questions-v1
"""
import argparse

from evals.run_experiment import build_evaluators


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:6006")
    ap.add_argument("--dataset", default="f1-qualifying-questions-v1")
    args = ap.parse_args()
    from phoenix.client import Client
    client = Client(base_url=args.endpoint)
    dataset = client.datasets.get_dataset(dataset=args.dataset)
    for info in client.experiments.list(dataset_id=dataset.id):
        experiment = client.experiments.get_experiment(experiment_id=info["id"])
        print(f"re-scoring {info['id']}")
        client.experiments.evaluate_experiment(experiment=experiment, evaluators=build_evaluators(), print_summary=False)


if __name__ == "__main__":
    main()
