# To do

Ordered roughly by value. Items marked (measured) came out of an experiment; the rest are ideas.

## Retrieval and data
- [ ] **Functions instead of stored pair chunks** (decided: option B). Persist each session's compact per-driver lap
      telemetry and events at ingest; generate head-to-head sections on demand with the existing
      `chunk_head_to_head_event`; add `grid()`, `team_summary()`, `section_ranking()`. Removes about 60% of the store and
      the "only stored pairs work" limit. Parity test: generated text must equal the stored chunk text for every stored pair.
- [ ] **Team names -> drivers** ("the ferraris", "mclaren lock out"). Three of ten held-out questions failed on this (measured).
- [ ] **Session aliases**: Aussie, Oz, barca, barc (measured: "Aussie quali" was answered from Barcelona).
- [ ] **Headline margin in the prompt**: pass the pair's lap gap and lap times into the context. Answers explain sections but
      never state 0.006 / 0.064 / 0.335 s (measured).
- [ ] **Whole-grid chunk** (positions, teams, gaps to pole and to the car ahead) for "why were drivers so close" questions.
- [ ] Fix "final corner": the model calls T13 the last corner at Melbourne; the lap ends at T14 (measured, recurs).

## Tool calling (saved for later, wanted for richer Phoenix traces)
- [ ] **Option C**: give Claude the option-B functions as tools so it chooses and chains calls. Costs roughly +5 to 15 s per
      question and adds run-to-run variation; measure against the deterministic router with the same experiments.
      Needs checks on the tool calls (right function, right arguments) and tool results counted as evidence.

## Evaluation
- [ ] **Held-out set, part 2**: data-generated questions (ground truth computed from the store) plus a batch written by a
      fresh model that has not seen the checks. Freeze before running; never tune on it. Part 1 (ten user-written
      questions, `evals/questions_heldout_user.json`) is already frozen and was run once: 3 clean, 7 with problems.
- [ ] **Tyre-claim check**: fail when an answer blames tyre compound/age for a gap while the evidence shows identical
      tyres (every fastest lap in all three sessions was a fresh SOFT).
- [ ] **Parameter sweep, retrieval-only first**: rerank weight (3.0), chunks per question (6), routing. Needs the rerank
      weight exposed as a parameter. Costs no API calls.
- [ ] **Safety set** before any public deploy: prompt-injection questions, off-topic requests, very long inputs.
- [ ] Record the inner retrieval and Claude spans inside experiment runs (currently only our own metrics).
- [ ] Spot-check 15 flagged and unflagged answers by hand to measure how often the checks are wrong.
- [ ] More repetitions for self-consistency (8 questions x 3 runs is a small sample).

## Housekeeping
- [ ] Add a documented price for `claude-sonnet-4-6` to `evals/prices.json` (or move the app to a model with a documented price).
- [ ] Update the 7 stale tests in `tests/test_retrieval.py` (Q3-cutoff and slow-driver cases).
- [ ] Push the `openf1-official-corners` branch and deploy (Render blueprint is in `render.yaml`).
