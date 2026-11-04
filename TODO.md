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
- [ ] Head-to-head sections still compare each driver's overall best lap (telemetry is stored for that lap only). For a Q1-vs-Q3 pair the sections are therefore not like-for-like; the comparison block says so. Storing each Q session's best-lap telemetry would fix it.
- [ ] **Team-level claims**: the answer called Ferrari the "fourth quickest team" when the best car of each team puts them third; no check catches ranking claims yet.
- [ ] Pair slots are the top 4 sections by time gap, so `retrieval_coverage` is now close to true by construction for pair questions; rely on `answer_coverage` instead.

## Tool calling (saved for later, wanted for richer Phoenix traces)
- [ ] **Option C**: give Claude the option-B functions as tools so it chooses and chains calls. Costs roughly +5 to 15 s per
      question and adds run-to-run variation; measure against the deterministic router with the same experiments.
      Needs checks on the tool calls (right function, right arguments) and tool results counted as evidence.

## Evaluation
- [~] **Held-out set, part 2**: the data-generated half is DONE (`evals/make_heldout.py`, 57 questions, new seed = fresh set). Still to do: the batch written by a fresh model that has not seen the checks. Original note: data-generated questions (ground truth computed from the store) plus a batch written by a
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

- [ ] Re-run the data-generated set with a new seed (`python -m evals.make_heldout --seed N`, then run_experiment with a new `--dataset` name) whenever the checks or retrieval change materially.
