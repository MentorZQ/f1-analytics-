"""
For every head-to-head section chunk in the store, rebuild it from the session snapshot and compare the text.

    python scripts/check_snapshot_parity.py [--store PATH]

Run it on a COPY of the store (opening Chroma can rewrite its files). Exits non-zero on any mismatch.
"""
import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb  # noqa: E402

from events_and_chunking import session_snapshot as snap  # noqa: E402

SESSIONS = ["f1_qualifying_2024_monza", "f1_qualifying_2026_barcelona", "f1_qualifying_2026_melbourne"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(ROOT / "RAG_data_layers" / "chroma_store"))
    args = ap.parse_args()
    client = chromadb.PersistentClient(path=args.store)
    bad_total = 0
    for name in SESSIONS:
        snapshot = snap.load_snapshot(ROOT / "RAG_data_layers" / "session_snapshots" / f"{name}.npz")
        stored = client.get_collection(name).get(where={"chunk_type": "head_to_head_event"}, include=["documents", "metadatas"])
        by_pair: dict = {}
        for cid, doc, meta in zip(stored["ids"], stored["documents"], stored["metadatas"]):
            by_pair.setdefault((meta["driver_a"], meta["driver_b"]), {})[cid] = doc
        same = missing = differ = 0
        examples = []
        for (a, b), docs in by_pair.items():
            rebuilt = {c.chunk_id: c.text for c in snap.compare_drivers(snapshot, a, b)}
            for cid, doc in docs.items():
                if cid not in rebuilt:
                    missing += 1
                    examples.append((cid, "not rebuilt", ""))
                elif rebuilt[cid] == doc:
                    same += 1
                else:
                    differ += 1
                    diff = "\n".join(list(difflib.unified_diff(doc.splitlines(), rebuilt[cid].splitlines(), lineterm="", n=0))[:8])
                    examples.append((cid, "text differs", diff))
        print(f"{name}: {len(by_pair)} pairs, {same + missing + differ} stored chunks: {same} identical, {differ} differ, {missing} not rebuilt")
        for cid, why, diff in examples[:3]:
            print(f"   {cid}: {why}\n{diff}")
        bad_total += differ + missing
    print("PARITY OK" if not bad_total else f"PARITY FAILED: {bad_total} chunks")
    return 1 if bad_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
