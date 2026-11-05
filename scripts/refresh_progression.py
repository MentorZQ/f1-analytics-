"""
Rewrite the 'Qualifying progression' lines (and the per-session best times) in the stored driver lap summaries from the
session snapshots, without re-ingesting everything. Uses the same helper as the pipeline (chunk_driver.progression_lines),
and re-embeds only the chunks whose text changed.

    python scripts/refresh_progression.py [--chroma-path RAG_data_layers/chroma_store]

Run reingest.py --snapshots-only first so the snapshots carry the per-session times.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RAG_data_layers" / "F1 Project Thoughts"))

import chromadb  # noqa: E402

from events_and_chunking import session_snapshot as snap  # noqa: E402
from events_and_chunking.chunk_driver import progression_lines  # noqa: E402

_OLD = re.compile(r"Qualifying progression:[^\n]*\n(?:Best time in each session:[^\n]*\n)?")


def refreshed(doc: str, row) -> str:
    new = "\n".join(progression_lines(row)) + "\n"
    if _OLD.search(doc):
        return _OLD.sub(lambda _: new, doc, count=1)
    return doc.replace("\nBest lap time:", "\n" + new + "Best lap time:", 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-path", default=str(ROOT / "RAG_data_layers" / "chroma_store"))
    args = parser.parse_args()
    from reingest import embed_text

    client = chromadb.PersistentClient(path=args.chroma_path)
    for info in client.list_collections():
        name = info.name if hasattr(info, "name") else info
        col = client.get_collection(name)
        snapshot = snap.load_snapshot(ROOT / "RAG_data_layers" / "session_snapshots" / f"{name}.npz")
        rows = {r["Driver"]: r for _, r in snapshot.fastest.iterrows()}
        got = col.get(where={"chunk_type": "driver_lap_summary"}, include=["documents", "metadatas"])
        ids, docs = [], []
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
            row = rows.get(meta.get("driver"))
            if row is None:
                continue
            new = refreshed(doc, row)
            if new != doc:
                ids.append(cid)
                docs.append(new)
        if ids:
            col.update(ids=ids, documents=docs, embeddings=[embed_text(d) for d in docs])
        print(f"{name}: {len(ids)} of {len(got['ids'])} lap summaries rewritten")


if __name__ == "__main__":
    main()
