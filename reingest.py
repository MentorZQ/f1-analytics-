"""
reingest.py
-----------
Rebuild both ChromaDB collections from scratch using the current chunk pipeline.

Run from the project root:
    python reingest.py

Rebuilds:
  - f1_qualifying_2024_monza
  - f1_qualifying_2026_barcelona

Both sessions use the updated pipeline with:
  - normalize_event_boundaries : non-overlapping sections that tile the full lap
  - _time_at_distance          : interpolated section times at exact boundaries
"""

import sys
import json
from pathlib import Path

import chromadb
import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "RAG_data_layers/F1 Project Thoughts"))

from load_qualifying import load_qualifying_session
from load_openf1 import load_openf1_qualifying_context
from load_track_data import load_all_track_data
from events_and_chunking import build_all_chunks, chunk_summary, validate_chunks

# ── Embed model ───────────────────────────────────────────────────────────────

_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")


def embed_text(text: str) -> list[float]:
    return next(_model.embed([text])).tolist()


def clean_metadata(metadata: dict) -> dict:
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (list, tuple, set, dict)):
            value = json.dumps(value)
        cleaned[key] = value
    return cleaned


# ── Sessions to rebuild ───────────────────────────────────────────────────────

SESSIONS = [
    {"year": 2024, "circuit": "Monza",     "short": "monza"},
    {"year": 2026, "circuit": "Barcelona", "short": "Catalunya"},
    {"year": 2026, "circuit": "Melbourne", "short": "Melbourne"},
]

CHROMA_PATH = ROOT / "RAG_data_layers/chroma_store"

# ── Main ──────────────────────────────────────────────────────────────────────

def ingest_session(client: chromadb.PersistentClient, cfg: dict) -> None:
    year    = cfg["year"]
    circuit = cfg["circuit"]
    short   = cfg["short"]

    print(f"\n{'='*60}")
    print(f"  {circuit} {year} Qualifying")
    print(f"{'='*60}")

    # Load source data
    track       = load_all_track_data(circuit)
    session_data = load_qualifying_session(year=year, circuit=circuit)
    openf1_ctx  = load_openf1_qualifying_context(year=year, circuit_short_name=short)

    # Build chunks (normalization happens inside build_all_chunks → build_telemetry_chunks)
    chunks = build_all_chunks(
        fastf1_data=session_data,
        openf1_data=openf1_ctx,
        track_data=track,
        session_info=session_data["session_info"],
        head_to_head_top_n=5,
        verbose=True,
    )

    warnings = validate_chunks(chunks)
    if warnings:
        print("\nValidation warnings:")
        for w in warnings:
            print(f"  {w}")

    # (Re)create Chroma collection
    collection_name = f"f1_qualifying_{year}_{circuit.lower().replace(' ', '_')}"
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        print(f"\nDropping existing collection: {collection_name}")
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed and add
    print(f"\nEmbedding {len(chunks)} chunks …")
    ids         = []
    documents   = []
    embeddings  = []
    metadatas   = []

    for i, chunk in enumerate(chunks):
        ids.append(chunk.chunk_id)
        documents.append(chunk.text)
        embeddings.append(embed_text(chunk.text))
        metadatas.append(clean_metadata({"chunk_type": chunk.chunk_type, "source": chunk.source, **chunk.metadata}))
        if (i + 1) % 50 == 0:
            print(f"  … {i+1}/{len(chunks)}")

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"\nStored {collection.count()} documents in '{collection_name}'")


def main() -> None:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    for cfg in SESSIONS:
        ingest_session(client, cfg)

    print("\n\nAll collections rebuilt.")
    for c in client.list_collections():
        print(f"  {c.name}: {client.get_collection(c.name).count()} documents")


if __name__ == "__main__":
    main()
