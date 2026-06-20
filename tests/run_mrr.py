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

# ── Retrieval variants ────────────────────────────────────────────────────────

def baseline_retrieve(col, question: str, n: int = 6) -> list[dict]:
    """Raw cosine, no h2h fetch, no delta reranking."""
    codes = extract_driver_codes(question)
    where = build_driver_where(codes) if codes else None
    return retrieve_chunks(col, question, embed_fn, n_results=n, where=where,
                           normalize=True, auto_filter=False)


def h2h_only_retrieve(col, question: str, ctx: dict, n: int = 6) -> list[dict]:
    """H2h-first fetch with NO significance reranking (delta_weight=0)."""
    from rag_retrieval import extract_driver_codes, resolve_context_drivers, build_driver_where
    named = extract_driver_codes(question)
    all_codes = resolve_context_drivers(question, named, ctx)
    where = build_driver_where(all_codes) if all_codes else None

    if len(all_codes) >= 2:
        codes_list = sorted(all_codes)
        pair_conds = []
        for i, a in enumerate(codes_list):
            for b in codes_list[i+1:]:
                pair_conds.append({"$and": [{"driver_a": a}, {"driver_b": b}]})
                pair_conds.append({"$and": [{"driver_a": b}, {"driver_b": a}]})
        pair_filter = pair_conds[0] if len(pair_conds) == 1 else {"$or": pair_conds}
        h2h_where = {"$and": [{"chunk_type": "head_to_head_event"}, pair_filter]}
        chunks = retrieve_chunks(col, question, embed_fn, n_results=n * 3,
                                 where=h2h_where, normalize=True, auto_filter=False)
        seen = {c["chunk_id"] for c in chunks}
        for c in retrieve_chunks(col, question, embed_fn, n_results=n, where=where,
                                 normalize=True, auto_filter=False):
            if c["chunk_id"] not in seen:
                chunks.append(c); seen.add(c["chunk_id"])
    else:
        chunks = retrieve_chunks(col, question, embed_fn, n_results=n * 3,
                                 where=where, normalize=True, auto_filter=False)
    # No reranking — return by raw cosine order
    return sorted(chunks, key=lambda c: c["distance"])[:n]


def delta_only_retrieve(col, question: str, ctx: dict, n: int = 6) -> list[dict]:
    """Delta reranking (weight=3.0) but NO h2h-first fetch — standard cosine pool."""
    from rag_retrieval import extract_driver_codes, resolve_context_drivers, build_driver_where
    named = extract_driver_codes(question)
    all_codes = resolve_context_drivers(question, named, ctx)
    where = build_driver_where(all_codes) if all_codes else None
    chunks = retrieve_chunks(col, question, embed_fn, n_results=n * 3,
                             where=where, normalize=True, auto_filter=False)
    return rerank_by_significance(chunks)[:n]


# ── Run eval ──────────────────────────────────────────────────────────────────

N = 6
print(f"{'#':<4} {'Q':<44} {'BASE':>6} {'H2H':>6} {'DELT':>6} {'CURR':>6}")
print("-" * 74)

base_scores, h2h_scores, delt_scores, curr_scores = [], [], [], []

for i, item in enumerate(EVAL_SET, 1):
    col = cols[item["session"]]
    ctx = ctxs[item["session"]]
    q   = item["question"]
    exp = item["expected_chunks"]

    base_chunks = baseline_retrieve(col, q, n=N)
    h2h_chunks  = h2h_only_retrieve(col, q, ctx, n=N)
    delt_chunks  = delta_only_retrieve(col, q, ctx, n=N)
    curr_chunks = smart_retrieve(col, q, embed_fn, n_results=N, session_ctx=ctx)

    base_rr = reciprocal_rank(base_chunks, exp)
    h2h_rr  = reciprocal_rank(h2h_chunks,  exp)
    delt_rr = reciprocal_rank(delt_chunks,  exp)
    curr_rr = reciprocal_rank(curr_chunks, exp)

    base_scores.append(base_rr); h2h_scores.append(h2h_rr)
    delt_scores.append(delt_rr); curr_scores.append(curr_rr)

    q_short = q[:43]
    print(f"{i:<4} {q_short:<44} {base_rr:>6.3f} {h2h_rr:>6.3f} {delt_rr:>6.3f} {curr_rr:>6.3f}")

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

base_mrr = sum(base_scores)  / len(base_scores)
h2h_mrr  = sum(h2h_scores)  / len(h2h_scores)
delt_mrr = sum(delt_scores) / len(delt_scores)
curr_mrr = sum(curr_scores) / len(curr_scores)

print("-" * 74)
print(f"{'MRR@6':<48} {base_mrr:>6.3f} {h2h_mrr:>6.3f} {delt_mrr:>6.3f} {curr_mrr:>6.3f}")
print()
print("Ablation summary:")
print(f"  Baseline (raw cosine)             : {base_mrr:.3f}")
print(f"  + h2h-first fetch only            : {h2h_mrr:.3f}  ({h2h_mrr-base_mrr:+.3f})")
print(f"  + delta reranking only            : {delt_mrr:.3f}  ({delt_mrr-base_mrr:+.3f})")
print(f"  Full system (h2h fetch + delta)   : {curr_mrr:.3f}  ({curr_mrr-base_mrr:+.3f})")
