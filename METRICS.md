# Performance Metrics

Tracked metrics for the F1 qualifying RAG system. Updated as improvements are made.

---

## Retrieval Quality

### MRR@6
*Mean Reciprocal Rank of first expected chunk in top-6 results. Run: `python tests/run_mrr.py`*

| Variant | MRR@6 | Delta vs baseline |
|---|---|---|
| Baseline (raw cosine, no h2h fetch, no delta weight) | 0.486 | — |
| + h2h-first fetch only (no delta reranking) | 0.468 | -0.019 |
| + delta reranking only (no h2h fetch) | 1.000 | +0.514 |
| Full system (h2h fetch + delta_weight=3.0) | 0.972 | +0.486 |

**Key finding:** Delta reranking drives the lift. H2h-first fetch alone slightly hurts MRR (it changes the candidate pool order without the ranking signal to compensate), but is still necessary — it's what guarantees mid-grid pairs (e.g. BOR vs LAW) enter the pool at all. The one regression (Q17, HUL Q-miss question) pulls full system below delta-only.

### Corner Retrieval Recall
*Whether the most diagnostic section for a driver pair surfaces in top-6. Observed on Barcelona RUS vs HAM.*

| Version | T11-T14 (T10-T11-T12) Recall |
|---|---|
| Before fix | 0% (cosine rank 8+, never retrieved at n=6) |
| After h2h-first fetch | 100% (rank 2) |

---

## Data Accuracy

### Section time accuracy
*Section times for any driver pair should sum to within ~1ms of their actual lap time gap.*

| Status | Value |
|---|---|
| Current | Within 1ms for all top-10 driver pairs at Barcelona and Melbourne |
| How verified | Summed `time_delta_s` across all sections per pair, compared to `lap_time_seconds` gap from driver_lap_summary chunks |
| Bug that motivated this | Raw geometry detection produced overlapping 80m zones at every corner-to-straight boundary — each boundary region was being double-counted, producing a spurious 0.4s advantage on a single straight for drivers only 0.064s apart overall |

---

## Test Suite

### Retrieval unit tests
| Status | Count |
|---|---|
| Passing | 24 / 24 |
| Run with | `python -m pytest tests/test_retrieval.py` |

---

## Pending Metrics

| Metric | Why | How to measure |
|---|---|---|
| NDCG@6 | Weighted recall — rewards surfacing highest time-delta sections at higher ranks | Add to `tests/run_mrr.py` alongside MRR |
| RAGAS faithfulness | Does the LLM hallucinate facts outside the retrieved context? | `pip install ragas`, run eval question set through it |
| Ablation table | Empirically justify each design decision (h2h fetch, delta weight, etc.) | Run MRR@6 with one component removed at a time |
| Latency p50/p95 | Deployability signal for applied AI roles | Time retrieval + LLM call across 20 questions, report percentiles |
