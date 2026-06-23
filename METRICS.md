# Performance Metrics

Tracked metrics for the F1 qualifying RAG system. Updated as improvements are made.

---

## Retrieval Quality

### MRR@6
*Mean Reciprocal Rank of first expected chunk in top-6 results. Run: `python tests/run_mrr.py`*

| Variant | MRR@6 | Delta vs baseline |
|---|---|---|
| Baseline (raw cosine, no h2h fetch, no delta weight) | 0.486 | — |
| Full system (h2h-first fetch + delta_weight=3.0) | 0.972 | +0.486 |

**Key finding:** Delta reranking is the primary driver of the lift. H2h-first fetch is architecturally necessary — it guarantees mid-grid pairs (e.g. BOR vs LAW at Melbourne) enter the candidate pool — but the ranking signal is what turns those candidates into rank-1 results.

### Corner Retrieval Recall
*Whether the most diagnostic section for a driver pair surfaces in top-6.*

| Version | Most diagnostic section recall |
|---|---|
| Before fix | 0% — T10-T11-T12 at Barcelona (Russell's biggest gain) was cosine rank 8+, never retrieved at n=6 |
| After h2h-first fetch + delta reranking | 100% — surfaces at rank 2 |

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

## Answer Quality

### RAGAS Faithfulness
*Fraction of claims in the LLM answer that are directly supported by retrieved context.*
*Run: `python tests/run_ragas.py` (answers cached in `tests/ragas_cache.json` — use `--regen` to refresh)*

| Version | RAGAS Faithfulness | Per-claim audit |
|---|---|---|
| Current (claude-sonnet-4-6, n=6 chunks) | 0.872 | 0.941 (312 claims, 18 unsupported) |

**What the 0.872 actually reflects:** Manual per-claim audit via `python tests/run_faithfulness_audit.py` found zero fabricated facts across 18 questions. All 18 unsupported claims fell into two categories:

- **Causal inference (10/18):** The LLM connects sequential data points into an explanation — e.g. "exit speed deficit fed into the following straight." This is physically correct and derivable from the telemetry, but the causal link is not literally stated in the context. RAGAS penalises it; a domain expert would not.
- **Speculation / paraphrase (8/18):** Either the LLM speculates about engineering causes ("suggests a Red Bull power unit advantage") or rephrases the chunk's own speculative language into a different conclusion. No training-data knowledge was introduced.

The 0.872 vs 0.941 gap is explained by RAGAS decomposing answers into more atomic statements and applying stricter NLI verification without leniency for valid arithmetic or derivable inferences. Both numbers are reported because together they give a complete picture: the standard metric for comparison against published systems, and the per-claim breakdown showing what it actually caught.

---

## Pending Metrics

| Metric | Why | How to measure |
|---|---|---|
| NDCG@6 | Weighted recall — rewards surfacing highest time-delta sections at higher ranks | Add graded relevance scoring to `tests/run_mrr.py` |
| Latency p50/p95 | Deployability signal for applied AI roles | Time retrieval + LLM call across 20 questions, report percentiles |
