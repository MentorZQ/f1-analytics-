# Performance Metrics

Tracked metrics for the F1 qualifying RAG system. Updated as improvements are made.

---

## Retrieval Quality

### MRR@6
*Mean Reciprocal Rank of first expected chunk in top-6 results. Run: `python tests/run_mrr.py`*

| Variant | MRR@6 | Delta vs baseline |
|---|---|---|
| Baseline (raw cosine, no h2h fetch, no delta weight) | 0.377 | — |
| Full system (h2h-first fetch + delta_weight=3.0) | 0.935 | +0.558 |

*Caveat: expected chunks are the top-3 |time delta| sections per pair read from the current store, the same signal the reranker uses, so this measures how well the pipeline finds its own largest deltas, not independent relevance. Labels were regenerated after the OpenF1 / official-corner rebuild.*

**Key finding:** Delta reranking is the primary driver of the lift. H2h-first fetch is architecturally necessary — it guarantees mid-grid pairs (e.g. BOR vs LAW at Melbourne) enter the candidate pool — but the ranking signal is what turns those candidates into rank-1 results.

### Corner Retrieval Recall
*Whether the most diagnostic section for a driver pair surfaces in top-6.*

| Version | Most diagnostic section recall |
|---|---|
| Before fix (earlier data build) | 0% — the biggest-gain corner section at Barcelona was cosine rank 8+, never retrieved at n=6 |
| After h2h-first fetch + delta reranking | Measured on the earlier data build only; re-measure on the current build |

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

### Section boundaries (OpenF1 + official corners)
*Run on the fastest laps of Barcelona 2026, Melbourne 2026 and Monza 2024.*

| Check | Old geometry sections | Current |
|---|---|---|
| Drivers mid-brake at a boundary | ~20% mean, worst 100% | 0% on all three circuits |
| Sections vs official turns | 24 detected at Melbourne vs 14 official | official turn numbers |
| Implausible section timings (average speed above top speed) | 18% at Melbourne | 0.3–0.7% |
| Barcelona lap gap RUS–HAM | n/a | 0.063 s (official 0.064 s) |
| Corner comparisons with an unseen brake-on point (car-data gap) | not measured | 21–39%; reported as a range or "cannot tell" |

---

## Experiment results (Phoenix, deterministic checks)

*40 questions (34 answerable, 6 unanswerable) run through the full pipeline with the same prompt and model (claude-sonnet-4-6, 6 chunks), once on the earlier data build (FastF1-era geometry sections) and once on the current build (OpenF1 + official corners). Scored by the code checks in `evals/code_checks.py`. Reproduce: `python -m evals.run_experiment --name ...` then `python -m evals.compare`.*

| Check | Earlier data | Current data |
|---|---|---|
| Answer is grounded: every number in the chunks (number_grounding) | 32 pass / 2 fail | 36 pass / 0 fail |
| Totals consistent with the section gains (total_consistency) | 4 / 0 | 5 pass / 1 fail |
| No unhedged claim where the data has a gap (uncertainty_compliance) | n/a (old chunks never reported gaps) | 24 pass / 1 fail |
| Biggest-gap sections retrieved (retrieval_coverage, broad questions only) | 19 / 5 | 22 pass / 2 fail |
| Biggest-gap sections discussed (answer_coverage, broad questions only) | 16 / 8 | 21 pass / 3 fail |
| Expected drivers retrieved | 37 / 1 | 37 pass / 1 fail |
| Unanswerable questions declined | 4 / 2 | 6 pass / 0 fail |
| Answerable questions answered | 34 / 0 | 34 pass / 0 fail |
| Share of sentences with causal wording (inference_rate, not pass/fail) | 0.15 | 0.18 |

**What the failures are (current data):**
- *Real:* a Barcelona answer called one section's 0.147 s gain "behind Russell across the measured sections" (the lap gap was 0.321 s); another said Hamilton's "later braking" was a genuine advantage across T1-T2, T4 and T10-T11, but T4's braking comparison is unresolved because of a gap in the car data.
- *Real retrieval gaps:* Alonso vs Hulkenberg at Monza retrieves no Alonso chunk (head-to-head chunks exist only for the top 10); nameless questions ("why did the pole sitter win?") retrieve too few of the biggest-gap sections.

### After the retrieval fix (nameless pole questions, guaranteed named drivers)

All three runs below were scored with the same final checks (the earlier two were re-scored).

| Check | Earlier data | Current data | + retrieval fix |
|---|---|---|---|
| Biggest-gap sections retrieved (broad questions) | 19 / 5 fail | 22 / 2 | **24 / 0** |
| Biggest-gap sections discussed in the answer | 17 / 7 | 21 / 3 | **24 / 0** |
| Expected drivers retrieved | 37 / 1 | 37 / 1 | **38 / 0** |
| Numbers grounded | 32 / 2 | 36 / 0 | 36 / 0 |
| Totals consistent | 3 / 1 | 5 / 1 | 7 / 0 |
| No unhedged claim at a data gap | n/a | 24 / 1 | 27 / 0 |
| Unanswerable questions declined | 4 / 2 | 6 / 0 | 6 / 0 |

**Reading this carefully:**
- The retrieval fix is real and measurable without any LLM: expected drivers retrieved went from 33/34 to 34/34 and both nameless pole questions went from 1 of 3 to 3 of 3 biggest-gap sections (retrieval-only check, `evals` scripts).
- Zero failures across the board does not mean the system is flawless. The checks were corrected about a dozen times while reading these 40 questions' failures, so they are tuned to this set. They need a held-out set of new questions before the clean sheet means much.
- Runs vary: the one real unhedged braking claim found in the previous run did not recur in this one with the same prompt, so single runs of 40 cannot separate small improvements from noise. Repetitions are needed.
- Alonso vs Hulkenberg (no stored head-to-head pair) now gets an honest answer from lap and sector data and says corner-level data is not available, rather than an unrelated pair.

**How much to trust this:**
- The checks were corrected several times while reading the first run's failures (a thousands-separator parsing bug, "overall" read as a lap total, claims blamed on the wrong section or on a gap chunk about other drivers, bold text read as a heading, decline detection tripping on markdown, "caution" not counted as a hedge) and once to catch an overclaim they had missed (a claim after a list of sections). Each fix has a regression test, and old and new runs were re-scored with the same final rules, so the comparison is like for like.
- "Earlier data" vs "current data" differs in sections, braking text and gap handling at once, so this shows the rebuild did not hurt and where it helped (mainly retrieval and answer coverage of the big-gap sections), not which part helped.
- One run per condition, 40 questions: differences of one or two questions are noise.
- Coverage uses the stored time deltas as its reference, which is partly circular for retrieval. The checks do not judge whether causal explanations are sound (see inference_rate).

---

## Claim checks, honesty and consistency (Phoenix, deterministic)

*Seven checks added after the first experiments (reversed winners, wrong driver/session, numbers on the wrong driver, completeness, false premises, overclaiming, leaked self-correction). They were calibrated on 174 stored real answers: every flag was read, false alarms were fixed and given regression tests, and five real errors remained.*

**Real errors the checks found in stored answers**
- Antonelli vs Russell, T1-T2: the chunk says ANT faster by 0.060 s; the answer wrote "ANT faster by 0.060s - wait, corrected: RUS faster by 0.060s here". The model "corrected" itself into the wrong answer.
- Two answers revise themselves in the text ("wait, notably Hamilton's exit speed is higher").
- "Who had the better exit from the final corner in Melbourne": the answers call T13 the final corner; the lap ends at T14 (flagged in 2 of 3 runs).

**Latest 40-question run (current data, retrieval fix, all checks)** - 4 flags in 40 answers: one answer that revises itself, the T14 "final corner" miss, one coverage miss (Hamilton vs Verstappen) and one ungrounded figure. `reversed_winner`, `wrong_driver_or_session`, `numbers_on_right_driver`, `overclaiming` and `uncertainty_compliance` had no failures.

**Honesty set (19 questions)**
| | First run | With question-type routing |
|---|---|---|
| False premises challenged or corrected (7: crash, penalty, failure, spin, wrong pole-sitter, wrong tyre, Alonso in Q3) | 7 / 7 | 7 / 7 |
| Near-miss unanswerables declined (7: tyre pressure, engine mode, wing level, tow, brake temperature...) | 7 / 7 | 7 / 7 |
| Tricky answerables answered with the stored facts (5: tyre compound, track temperature change, session improvement 0.584 s, weather, sector rank) | 3 / 5 | 5 / 5 |

The first run refused two answerable questions (tyre compound, improvement across the session) because only head-to-head chunks were retrieved. Routing by question type (tyre -> lap summary, "improve" -> pace evolution, weather, flags, sector rank) fixed both. Two of the first run's failures were the decline detector missing "contains no ... data" wording.

**Self-consistency (8 questions x 3 runs)**
| Measure | Result |
|---|---|
| Same numbers stated | 0.69 |
| Same sections discussed | 0.84 |
| Section winners named agree | 1.00 |
| Questions where a check flipped between runs | 3 of 8 |

In one run in three each, two questions claimed "more aggressive braking" at T4, a section whose braking the data cannot resolve. A single run would have missed this. Section winners never flipped.

**Speed and tokens (40-question run, claude-sonnet-4-6)**: median 14.3 s per question (p95 21.7 s), of which the Claude call is 14.7 s on average and retrieval 0.02 s; about 2,160 input and 680 output tokens per question (86,488 in / 27,369 out in total). Cost is not computed because no documented price for this model is in `evals/prices.json`.

**Limits**
- The checks were tuned against these same questions; a held-out question set is needed before a clean sheet means much.
- 3 repetitions of 8 questions is a small consistency sample.
- `reversed_winner` stays silent when a sentence covers several sections or uses a pronoun; it cannot catch a reversal it cannot attribute.
- Coverage still uses stored time deltas as its reference.

---

## Test Suite

### Retrieval unit tests
| Status | Count |
|---|---|
| Passing | 17 / 24 (`tests/test_retrieval.py`). The 7 failures (Q3-cutoff and slow-driver cases) predate the data rebuild: they fail identically on the old store, since commit e98553c removed Q3-cutoff inference and the tests were not updated |
| Offline suite | all passing (`tests/test_server.py test_openf1_loader.py test_detect_events_official.py test_chunk_braking.py test_code_checks.py test_experiment_evaluators.py`) |

---

## Answer Quality

### RAGAS Faithfulness
*Measured on the previous data build; not yet re-run on the current one.*
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
