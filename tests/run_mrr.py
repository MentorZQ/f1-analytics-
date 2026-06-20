"""
run_mrr.py
----------
Computes MRR@6 for two retrieval methods and prints a per-question comparison.

Baseline  : raw cosine search, no h2h-first fetch, no delta reranking
            (simulates the system before the retrieval fixes)

Current   : h2h-first fetch + delta_weight=3.0 + DNQ injection
            (the live system as deployed)

MRR@6 = mean reciprocal rank of the first expected chunk within the top-6
        results. If no expected chunk appears, reciprocal rank = 0.

Run from the project root:
    python tests/run_mrr.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "RAG_data_layers/F1 Project Thoughts"))

import chromadb
from rag_retrieval import (
    smart_retrieve, retrieve_chunks, rerank_by_significance,
    make_embed_fn, build_driver_where, extract_driver_codes,
    resolve_context_drivers,
)
sys.path.insert(0, str(ROOT / "tests"))
from eval_set import EVAL_SET

# ── Session setup ─────────────────────────────────────────────────────────────

COLLECTIONS = {
    "barcelona": "f1_qualifying_2026_barcelona",
    "melbourne": "f1_qualifying_2026_melbourne",
}

client   = chromadb.PersistentClient(path=str(ROOT / "RAG_data_layers/chroma_store"))
embed_fn = make_embed_fn()

def get_session_ctx(col):
    r = col.get(where={"chunk_type": "driver_lap_summary"}, include=["metadatas"])
    drivers, teams = {}, {}
    for m in r["metadatas"]:
        if "driver" not in m:
            continue
        d = m["driver"]
        drivers[d] = {
            "qualifying_position": m.get("qualifying_position"),
            "lap_time_s": m.get("lap_time_seconds"),
            "team": m.get("team", ""),
        }
        t = m.get("team", "")
        if t:
            teams.setdefault(t, []).append(d)
    standings = sorted(drivers.items(), key=lambda x: x[1]["qualifying_position"] or 99)
    teammate_of = {}
    for drvs in teams.values():
        for d in drvs:
            mates = [x for x in drvs if x != d]
            if mates:
                teammate_of[d] = mates[0]
    return {
        "drivers":     drivers,
        "standings":   [d for d, _ in standings],
        "teammate_of": teammate_of,
        "pole":        standings[0][0] if standings else None,
        "p2":          standings[1][0] if len(standings) > 1 else None,
    }

cols, ctxs = {}, {}
for key, name in COLLECTIONS.items():
    cols[key] = client.get_collection(name)
    ctxs[key] = get_session_ctx(cols[key])

# ── Matching logic ────────────────────────────────────────────────────────────

def chunk_matches(chunk: dict, expected_type: str, expected_eid) -> bool:
    meta = chunk.get("metadata", {})
    if meta.get("chunk_type") != expected_type:
        return False
    if expected_eid is not None:
        return meta.get("event_id") == expected_eid
    return True  # type match only (dnq_status, lap_summary — any driver match is fine)

def reciprocal_rank(chunks: list[dict], expected: list[tuple]) -> float:
    """Return 1/rank of the first expected chunk found, or 0 if none in top 6."""
    for rank, chunk in enumerate(chunks[:6], 1):
        for exp_type, exp_eid in expected:
            if chunk_matches(chunk, exp_type, exp_eid):
                return 1.0 / rank
    return 0.0

# ── Baseline retrieval ────────────────────────────────────────────────────────

def baseline_retrieve(col, question: str, n: int = 6) -> list[dict]:
    """
    Raw cosine search with driver filter but no h2h-first fetch and no
    significance reranking. Mirrors the original system before the fixes.
    """
    codes = extract_driver_codes(question)
    where = build_driver_where(codes) if codes else None
    return retrieve_chunks(col, question, embed_fn, n_results=n, where=where,
                           normalize=True, auto_filter=False)

# ── Run eval ──────────────────────────────────────────────────────────────────

N = 6
print(f"{'#':<4} {'Q':<52} {'BASE':>6} {'CURR':>6}  {'DELTA':>6}  first-expected-in")
print("-" * 95)

base_scores, curr_scores = [], []

for i, item in enumerate(EVAL_SET, 1):
    col = cols[item["session"]]
    ctx = ctxs[item["session"]]
    q   = item["question"]
    exp = item["expected_chunks"]

    base_chunks = baseline_retrieve(col, q, n=N)
    curr_chunks = smart_retrieve(col, q, embed_fn, n_results=N, session_ctx=ctx)

    base_rr = reciprocal_rank(base_chunks, exp)
    curr_rr = reciprocal_rank(curr_chunks, exp)

    base_scores.append(base_rr)
    curr_scores.append(curr_rr)

    # Show which chunk first matched (or miss)
    match_label = "MISS"
    for rank, chunk in enumerate(curr_chunks[:N], 1):
        for exp_type, exp_eid in exp:
            if chunk_matches(chunk, exp_type, exp_eid):
                eid = chunk["metadata"].get("event_id") or chunk["metadata"].get("chunk_type","")
                match_label = f"rank {rank} → {eid[:28]}"
                break
        else:
            continue
        break

    delta = curr_rr - base_rr
    delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
    q_short = q[:51]
    print(f"{i:<4} {q_short:<52} {base_rr:>6.3f} {curr_rr:>6.3f}  {delta_str:>6}  {match_label}")

base_mrr = sum(base_scores) / len(base_scores)
curr_mrr = sum(curr_scores) / len(curr_scores)
lift     = curr_mrr - base_mrr

print("-" * 95)
print(f"{'MRR@6':<57} {base_mrr:>6.3f} {curr_mrr:>6.3f}  {lift:>+6.3f}")
print()
print(f"Baseline MRR@6 : {base_mrr:.3f}")
print(f"Current  MRR@6 : {curr_mrr:.3f}")
print(f"Absolute lift  : {lift:+.3f}  ({lift/base_mrr*100:+.1f}% relative)" if base_mrr > 0 else "")
