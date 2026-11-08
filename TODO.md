# To do

Ordered roughly by value. Items marked (measured) came out of an experiment; the rest are ideas.

## Retrieval and data
- [x] **Functions instead of stored pair chunks** (option B) - DONE: snapshots, compare_drivers, grid/team/section views, slim store. Original note: Persist each session's compact per-driver lap
      telemetry and events at ingest; generate head-to-head sections on demand with the existing
      `chunk_head_to_head_event`; add `grid()`, `team_summary()`, `section_ranking()`. Removes about 60% of the store and
      the "only stored pairs work" limit. Parity test: generated text must equal the stored chunk text for every stored pair.
- [x] **Team names -> drivers** ("the ferraris", "mclaren lock out"). Three of ten held-out questions failed on this (measured).
- [x] **Session aliases**: Aussie, Oz, barca, barc (measured: "Aussie quali" was answered from Barcelona).
- [x] **Headline margin in the prompt**: pass the pair's lap gap and lap times into the context. Answers explain sections but
      never state 0.006 / 0.064 / 0.335 s (measured).
- [x] **Whole-grid view** (positions, teams, gaps to pole and to the car ahead) for "why were drivers so close" questions.
- [x] Fix "final corner": named corners are resolved, pinned into the pair's sections, and the first/last corner is stated (65dbf40).

- [x] Q3 entry and like-for-like laps: a top-ten car with no Q3 time still reached Q3; each driver's Q1/Q2/Q3 best is stored; "last into Q3" uses Q2 times; a pair whose best laps come from different sessions also gets the last shared session's gap.
- [x] Head-to-head sections compare the like-for-like session's laps when the best laps come from different sessions (snapshots store each driver's lap from every Q session; the comparison block says which laps the sections use).
- [x] Grid "+x to the car ahead" uses one session (both Q3 times, both Q2 times, ...) and says so; cars that went out in different sessions get no gap.
- [~] Haiku guessing a name before reading the context: prompt line added, but it still appears on "last into Q3" questions (2 of 3 Haiku runs); Sonnet 4.6 (the site's model) did not do it in 3 of 3. Only matters for measuring on Haiku.
- [x] Team ranking claims (`team_ranking` check: rank by best car or by average of the cars, from the grid chunk).
- [ ] Pair slots are the top 4 sections by time gap, so `retrieval_coverage` is now close to true by construction for pair questions; rely on `answer_coverage` instead.

## Tool calling (saved for later, wanted for richer Phoenix traces)
- [ ] **Option C**: give Claude the option-B functions as tools so it chooses and chains calls. Costs roughly +5 to 15 s per
      question and adds run-to-run variation; measure against the deterministic router with the same experiments.
      Needs checks on the tool calls (right function, right arguments) and tool results counted as evidence.

## Evaluation
- [x] Independent batch written by a fresh model (`evals/questions_heldout_fresh.json`, 60 questions): run, read and fixed; see METRICS.md. (Original note on part 2 follows.)
- [~] **Held-out set, part 2**: the data-generated half is DONE (`evals/make_heldout.py`, 57 questions, new seed = fresh set). Still to do: the batch written by a fresh model that has not seen the checks. Original note: data-generated questions (ground truth computed from the store) plus a batch written by a
      fresh model that has not seen the checks. Freeze before running; never tune on it. Part 1 (ten user-written
      questions, `evals/questions_heldout_user.json`) is already frozen and was run once: 3 clean, 7 with problems.
- [x] Tyre-claim check (`tyre_claims`: fails when the answer blames tyres but the evidence shows the same tyres, or none).
      tyres (every fastest lap in all three sessions was a fresh SOFT).
- [ ] **Parameter sweep, retrieval-only first**: rerank weight (3.0), chunks per question (6), routing. Needs the rerank
      weight exposed as a parameter. Costs no API calls.
- [x] Safety set (`evals/questions_safety.json`, 25 questions: prompt extraction, secrets, role hijack, off-topic, harmful, injection inside a real question, false authority, long input; `leaked_or_hijacked` check). 24 / 25 clean on Haiku and on Sonnet 4.6 (Haiku still ends with PWNED when told to).
- [x] Inner spans: a traced pass records f1.query / f1.retrieve per question in the Phoenix project `f1-rag-retrieval` (`--trace` on the context-only runner). Still open: spans of the Claude call inside the answer experiments (the experiment runner has its own tracer).
- [x] First spot-check of 13 answers done and loaded into Phoenix as `spotcheck-v1`. Next: a second one on 15 "why" answers from the main and honesty sets (where the checks are weakest).
- [ ] More repetitions for self-consistency (8 questions x 3 runs is a small sample).

## Housekeeping
- [ ] No documented price for `claude-sonnet-4-6` in the API reference, so `evals/prices.json` has none (runs on it report tokens and latency, not cost). Options: leave it, or move the site to Claude Sonnet 5.5 ($2 / $10 per MTok, documented) after measuring it on the safety and held-out sets (it needs thinking handled differently from Haiku).
- [x] Stale tests: the six Q3-cutoff tests were replaced by `tests/test_q3_boundary.py` (grid view and stage from the committed snapshots); `tests/test_retrieval.py` now passes. Note it opens the committed Chroma store, which rewrites files: `git checkout -- RAG_data_layers/chroma_store` afterwards (it is not in CI).
- [ ] Push the `openf1-official-corners` branch and deploy (Render blueprint is in `render.yaml`).

- [ ] Re-run the data-generated set with a new seed (`python -m evals.make_heldout --seed N`, then run_experiment with a new `--dataset` name) whenever the checks or retrieval change materially.

- [ ] Starting grid after penalties is not available from OpenF1 (`starting_grid` repeats the qualifying classification); answers say "qualified Pn, before any grid penalties". Would need another source (the race results page) to give the true grid.
- [ ] Run the spot-check again on 15 answers from the main 40-question set and the honesty set (more "why" answers, where the checks are weakest).

- [ ] Corner names exist only for Monza (first chicane, Lesmo, Ascari, Parabolica). Barcelona 2026 and Albert Park names are not written down, so a question that uses them without a number is not understood.
- [ ] "Fastest down the main straight" is answered by time through the section, not top speed; the speed ranking exists for corners and could be used for straights.
- [ ] A check for wrong or missing evidence in lookup answers (roughly 7 of 12 real problems on the independent batch were content errors the checks could not see): for example compare the driver named in the answer with the driver the evidence ranks first.

- [ ] Section-gap accuracy: raw cumulative gaps are off by 0.02 to 0.10 s on average against the official sector gaps (METRICS.md). A speed-trace alignment and a clock-stretch to the sector times were both tried and neither reliably helped. A better source of position (or fusing the car's speed integral with the track map differently at ingest) is the next thing to try; reingest is free (OpenF1 only).
- [ ] Re-run the independent batch and the spot-check on the aligned data (costs credits: use Haiku).

## Future: driver coaching
- [ ] Tell a driver where they are losing time: braking position, throttle pickup, or timing (built on the corner braking/throttle data; keep the 15 m accuracy rule and no cause guesses).

## Lookup router (built; follow-ups)
- [x] Answer one-fact questions from the data with no model call (`events_and_chunking/lookup_router.py`, `scripts/eval_router.py`).
- [ ] Measure it on a fresh, frozen question set run once (the current numbers were tuned on the sets it was built against).
- [ ] Cover what it hands to the model today: a driver's sector time, top speed (telemetry, not the speed trap), tyre set.
- [ ] Add routed vs model cost and latency to the Phoenix comparison (`evals/compare.py` already counts routed answers as zero tokens).

## TypeScript layer in production
- [x] Gateway, shared schemas and React frontend (`ts/`), Render blueprint with both services, trusted visitor address between them.
- [ ] Push, sync the Render Blueprint, and open the gateway URL (see `ts/README.md`); check `/health` on it returns 200.
- [ ] Decide whether the Python service becomes a private service once the gateway is live.
- [ ] Token streaming: a streaming endpoint in `server.py` relayed by the gateway.

