"""
reingest.py
-----------
Rebuild both ChromaDB collections from scratch using the current chunk pipeline.

Run from the project root:
    python reingest.py                      # OpenF1 data (free, no login)
    python reingest.py --source fastf1      # FastF1 (needs an F1 TV subscription)
    python reingest.py --chroma-path /tmp/store --only Barcelona

Rebuilds:
  - f1_qualifying_2024_monza
  - f1_qualifying_2026_barcelona

Both sessions use the updated pipeline with:
  - normalize_event_boundaries : non-overlapping sections that tile the full lap
  - _time_at_distance          : interpolated section times at exact boundaries
"""

import argparse
import os
import sys
import json
from pathlib import Path

import chromadb
import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "RAG_data_layers/F1 Project Thoughts"))

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
        if isinstance(value, float) and value != value:  # NaN
            continue
        if isinstance(value, (list, tuple, set, dict)):
            value = json.dumps(value)
        cleaned[key] = value
    return cleaned


# ── Sessions to rebuild ───────────────────────────────────────────────────────

SESSIONS = [
    {"year": 2024, "circuit": "Monza",     "short": "monza",     "h2h_top_n": 5},
    {"year": 2026, "circuit": "Barcelona", "short": "Catalunya", "h2h_top_n": 10},
    {"year": 2026, "circuit": "Melbourne", "short": "Melbourne", "h2h_top_n": 10},
]

CHROMA_PATH = ROOT / "RAG_data_layers/chroma_store"

# ── Main ──────────────────────────────────────────────────────────────────────

def load_session_data(cfg: dict, track: dict, source: str) -> dict:
    """Load lap + telemetry data from the chosen source.

    OpenF1: distance comes from position on the official track outline, and the
    section boundaries come from the official corner list plus where drivers
    brake and get back on the throttle (stored in track["events"]).
    """
    if source == "openf1":
        from load_qualifying_openf1 import find_session, load_qualifying_session_openf1
        from load_circuit_official import load_circuit_official
        from events_and_chunking.detect_events_official import detect_official_events, curvature_profile

        sess = find_session(cfg["year"], cfg["circuit"])
        try:
            official = load_circuit_official(sess["circuit_key"], cfg["year"])
        except LookupError as exc:
            print(f"  Official circuit data unavailable ({exc}); using the TUMFTM geometry instead")
            official = None

        if official is not None:
            data = load_qualifying_session_openf1(cfg["year"], cfg["circuit"],
                                                  geometry=official["outline"], exact_origin=True)
            lap = data["centerline"]
            apexes = [{"number": c["number"], "label": c["label"],
                       "dist": lap.arc_of(np.array([c["x_m"], c["y_m"]]))} for c in official["corners"]]
            kd, kk, _ = curvature_profile(official["outline"][["x_m", "y_m"]].to_numpy(float), origin=lap.origin)
            track["events"] = detect_official_events(data["telemetry"], apexes, lap.length, curvature=(kd, kk))
            print(f"  Official layout: {len(official['corners'])} corners (circuit data for {official['year_used']}) "
                  f"-> {len(track['events'])} sections")
            return data

        data = load_qualifying_session_openf1(cfg["year"], cfg["circuit"], geometry=track["geometry"])
        if data.get("geometry_aligned") is not None:
            track["geometry"] = data["geometry_aligned"]
        return data
    from load_qualifying import load_qualifying_session   # needs an F1 TV subscription
    return load_qualifying_session(year=cfg["year"], circuit=cfg["circuit"])


def ingest_session(client: chromadb.PersistentClient, cfg: dict, source: str = "openf1") -> None:
    year    = cfg["year"]
    circuit = cfg["circuit"]
    short   = cfg["short"]

    print(f"\n{'='*60}")
    print(f"  {circuit} {year} Qualifying")
    print(f"{'='*60}")

    # Load source data
    track       = load_all_track_data(circuit)
    session_data = load_session_data(cfg, track, source)
    openf1_ctx  = load_openf1_qualifying_context(year=year, circuit_short_name=short)

    # Build chunks (normalization happens inside build_all_chunks → build_telemetry_chunks)
    chunks = build_all_chunks(
        fastf1_data=session_data,
        openf1_data=openf1_ctx,
        track_data=track,
        session_info=session_data["session_info"],
        head_to_head_top_n=cfg.get("h2h_top_n", 5),
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
    parser = argparse.ArgumentParser(description="Rebuild the ChromaDB collections.")
    parser.add_argument("--source", choices=["openf1", "fastf1"], default="openf1",
                        help="telemetry source (default: openf1, free and needs no login)")
    parser.add_argument("--chroma-path", help="write the store here instead of RAG_data_layers/chroma_store")
    parser.add_argument("--only", help="comma-separated circuits to rebuild, e.g. Barcelona,Monza")
    args = parser.parse_args()

    sessions = SESSIONS
    if args.only:
        wanted = {w.strip().lower() for w in args.only.split(",")}
        sessions = [c for c in SESSIONS if c["circuit"].lower() in wanted]

    chroma_path = Path(args.chroma_path) if args.chroma_path else CHROMA_PATH
    chroma_host = os.environ.get("CHROMA_HOST")
    if chroma_host:
        client = chromadb.HttpClient(host=chroma_host, port=int(os.environ.get("CHROMA_PORT", "8000")))
    else:
        chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_path))

    for cfg in sessions:
        ingest_session(client, cfg, args.source)

    print("\n\nAll collections rebuilt.")
    for c in client.list_collections():
        print(f"  {c.name}: {client.get_collection(c.name).count()} documents")


if __name__ == "__main__":
    main()
