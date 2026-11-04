# F1 Qualifying RAG

A retrieval-augmented generation system for analysing F1 qualifying sessions. Ask natural language questions about lap times, sector performance, braking points, and head-to-head driver comparisons — answers are grounded in telemetry data from OpenF1, and where the car data has a gap the system says so instead of guessing.

**Sessions included (pre-built, no re-ingestion needed; stored head-to-head comparisons cover the top 10, every teammate pair and every pair of grid neighbours):**
- Melbourne 2026 Qualifying — 1498 chunks
- Barcelona 2026 Qualifying — 1604 chunks
- Monza 2024 Qualifying — 1431 chunks

---

## Design Decisions & Contributions

This project started with raw telemetry data and no framework for what "context" means for an F1 question. The core intellectual work was defining that — what information an LLM actually needs to reason about a qualifying lap, and how to structure it so retrieval surfaces the right things.

**Defining the unit of analysis: the track section**:
The first non-obvious decision was that a lap should be broken into spatial sections — not by time, not by sector, but by what happens on track. Corners that run into each other are grouped as sequences (T1–T2 at Barcelona has shared entry, two apices and one exit) because that's how drivers and engineers think about them. A sequence longer than 450 m is cut between two of its corners, at a point where no driver is braking. Straights are the recovery zones between them. This framing — derived from domain knowledge of how lap time is built — determined the entire chunk structure.

**Using braking and throttle as the structural signals**:
The official corner list gives where each corner is. Where it starts and ends comes from the drivers: a corner begins before the earliest driver lifts or brakes, and ends after the last driver is back on full throttle, so every driver's braking and throttle application sit inside the section. Curvature of the track outline is only a fallback for corners nobody brakes for. The LLM receives braking point (distance and entry speed), throttle pickup distance, and per-apex minimum speeds for every section. This gives it the information a race engineer would use to identify where time was gained or lost — not just that one driver was faster, but whether they carried more speed into the apex, braked later, or got on the throttle earlier on exit.

**Curating what data matters**:
Every field in every chunk was a deliberate choice about what an LLM needs to reason about lap time: entry speed tells you how much energy a driver carried from the previous section, apex speed tells you how much they scrubbed off, exit speed tells you how well they converted that into acceleration. DRS status, gear range, and braking distance are included because they explain the *why* behind the numbers. Fields that don't change the answer were left out.

**Catching the overlap bug through domain knowledge**:
During development, the section breakdown showed a 0.4s advantage for one driver on a single straight — but their overall lap gap was only 0.064s. That number was immediately suspicious: no driver gains four tenths on one straight against someone they only beat by six hundredths total. Investigation confirmed that the raw geometry detection was producing overlapping 80-metre zones at every corner-to-straight boundary, so each boundary region was being counted twice. The fix required normalising every boundary to a shared midpoint, interpolating times at the exact boundary distance rather than snapping to the nearest telemetry sample, and adding synthetic sections for the uncovered start and end of the lap. After the fix, section times sum to within 1ms of the actual lap time gap for every driver pair — the numbers close.

**Retrieval shaped by intent, not just keywords**:
A question like "Why was Bottas slow?" names one driver. The system detects that the question is about pace relative to someone, infers a comparison driver from session standings (teammate, pole-sitter, or Q3 boundary depending on question type), and expands the retrieval filter to include that driver before the vector search runs. For two-driver questions, the filter restricts results to chunks where exactly those two drivers are compared head-to-head — preventing chunks about unrelated driver pairs from filling the context window.

**Time delta as a ranking signal**:
Chunks where two drivers differ most in section time are promoted in ranking above chunks where they are nearly equal. A chunk showing a 0.3s difference through a corner complex is more likely to explain lap time than one showing 0.01s — so it surfaces first. This was the insight that moved the system from retrieving *related* content to retrieving *diagnostic* content.

**Fixing retrieval bias against corner sections**:
Vector similarity search for questions like "Why did Russell take pole?" systematically ranked straight sections above corner sections — because the embedding model associates pole position with top speed and DRS language. Corner chunks, which use braking/apex/throttle vocabulary, ranked lower even when they explained more of the lap time. The fix was two-part: for driver pair queries, the initial fetch is restricted to `head_to_head_event` chunks only (preventing per-driver telemetry chunks from consuming the candidate pool), and the time-delta reranking weight was raised from 0.5 to 3.0 so sections with large time gaps are strongly promoted regardless of their cosine rank. The same change surfaces the sections that explain the Russell–Hamilton Barcelona lap (T12, where Russell gained 0.125s, and T4 and T10–T11 where Hamilton did) in the top six.

**Sections that follow the official turn numbers and the drivers**:
The first detector found corners from the shape of an old track outline, which produced 24 "corners" at Melbourne against 14 official ones, put boundaries in the middle of braking zones (about 20% of drivers were mid-brake at a boundary), and was out of date for rebuilt layouts (Barcelona, Melbourne). Sections now come from the MultiViewer official corner list and outline for the right year, registered onto the car positions (median error 0.6 m), with each corner's entry and exit set from the drivers' own brake and throttle data. Result: 0% of drivers mid-brake at any boundary on all three circuits, and section names are the official turn numbers with no label mapping.

**Being honest about gaps in the car data**:
OpenF1 car data is about 3.7 Hz and has holes. In 21–39% of corner comparisons the exact brake-on point of at least one driver falls inside a hole. The chunk then reports a range ("first brakes between 1551m and 1616m, gap in the car data"), only claims who braked later when the ranges are clearly separate, and otherwise says it cannot tell. The answering prompt tells the model to say so in one clause and keep explaining from entry, apex, exit speed and time delta.

**The result**:
The project moved from a state where the vector database had no principled structure and retrieval returned whatever was semantically closest, to one where every retrieved chunk contains quantitatively precise, engineer-readable information, section sums match actual lap times, corner sections surface correctly alongside straights, and the system can answer questions like "where exactly did Russell beat his teammate" with official turn numbers, specific distances, speeds, and time deltas — derived from stored telemetry.

---

## Quickstart

```bash
git clone https://github.com/MentorZQ/f1-analytics-.git
cd f1-analytics-

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements_core.txt

cp .env.example .env
# Open .env and set: ANTHROPIC_API_KEY=sk-ant-...

python query.py "Why did Russell take pole at Barcelona 2026?"
```

The ChromaDB vector store ships with the repo — no data download or re-ingestion step required.

---

## Usage

```bash
# Single question
python query.py "Where did Antonelli lose time to Russell at Barcelona?"

# Force a specific session (auto-detected from question by default)
python query.py --session monza "How did Norris beat Piastri?"

# More chunks = more context for complex questions (default: 6)
python query.py --chunks 10 "Compare braking into T1 between Hamilton and Leclerc"

# See which chunks were retrieved before the answer
python query.py --verbose "Who just missed Q3 at the 2026 Spanish Grand Prix?"

# Interactive mode — ask multiple questions without restarting
python query.py
```

Session is auto-detected from the question text. Mentions of "Melbourne", "Australia" route to the Melbourne collection; "Barcelona", "Spain", "Catalunya" route to Barcelona; "Monza", "Italy", "2024" route to Monza.

---

## How it works

### Architecture

```
Question
   │
   ▼
smart_retrieve()          ← driver name normalisation, intent detection,
   │                         pair-specific metadata filtering, context expansion
   │
   ▼
ChromaDB (cosine search)  ← pre-built HNSW index, 384-dim embeddings (all-MiniLM-L6-v2)
   │
   ▼
Rerank by significance    ← chunks with larger time deltas between drivers ranked higher
   │
   ▼
format_prompt()           ← structured context block passed to LLM
   │
   ▼
Claude API                ← answers only from retrieved context
   │
   ▼
Answer
```

### Chunk types

Each session is broken into specialised chunk types stored in ChromaDB:

| Chunk type | What it contains |
|---|---|
| `head_to_head_event` | Two drivers compared at the same track section — section time, entry/apex/exit speeds, braking point, throttle pickup, DRS |
| `telemetry_corner_event` | One driver at one corner/complex — apex speeds per corner, braking distance, throttle pickup |
| `telemetry_straight_event` | One driver on one straight — top speed, DRS usage, braking point into the next corner |
| `driver_lap_summary` | Fastest lap time, sector splits, speed trap readings |
| `driver_sector` | Sector-by-sector breakdown with telemetry zone detail |
| `head_to_head` | Overall lap and sector comparison between two drivers |
| `session_pace_evolution` | Lap-by-lap time progression across Q1/Q2/Q3 |
| `race_control_event` | Flags, lap deletions, red flags, safety car messages |
| `session_weather` | Air/track temperature, humidity, rainfall during the session |
| `circuit_overview` | Track layout, length, corners, DRS zones |

### Track sections

The lap is divided into non-overlapping sections that tile the full track distance. Section boundaries are shared across all drivers, and times through each section are computed by linear interpolation at the exact boundary distance — not by slicing telemetry samples. This means:

- Section times for any driver pair sum to within ~1ms of their actual lap time gap
- Every metre of the lap belongs to exactly one section
- Straight sections include a synthetic lead straight (start/finish → T1) and tail straight (last corner → finish line)

### Retrieval features

- **Driver name normalisation** — "Norris", "Lando", "NOR" all resolve to the same driver code before embedding
- **Pair-specific filtering** — for two-driver questions, the metadata filter restricts search to chunks where exactly those two drivers are compared, preventing cross-pair contamination
- **Intent detection** — single-driver questions ("Why was Bottas slow?") detect whether the question is about pole, Q3 cutoff, or general pace, then infer a comparison driver from session standings
- **Guaranteed slot injection** — if an inferred comparison driver is missing from the top-N results, one targeted chunk is injected
- **Significance reranking** — chunks where drivers differ most in section time are promoted, so the most diagnostic sections surface first
- **Head-to-head priority fetching** — for driver pair questions, the candidate pool is filled with `head_to_head_event` chunks before any other type, preventing per-driver telemetry chunks from crowding out corner sections that rank lower in cosine space
- **Official turn labels** — section IDs are mapped to official FIA corner numbers per circuit; unnamed curvy sections are labelled relative to the nearest official turns (e.g. "Sweeper between T7-T8")

---

## Evaluation

All metrics are tracked in [METRICS.md](METRICS.md). Runners live in `tests/`.

> **Note:** the retrieval and faithfulness numbers below were measured on the previous data build. The MRR labels are regenerated from the current store with the same rule the reranker uses (top-3 time deltas), so MRR is partly circular. An independent evaluation with traced runs is the next step.

### Retrieval — MRR@6

Evaluated on an 18-question labeled set covering head-to-head comparisons, single-driver pole questions, and DNQ/Q-miss questions across Barcelona and Melbourne 2026. Expected chunks are the top-3 highest |time_delta_s| sections for each driver pair, derived directly from the stored data.

| System | MRR@6 |
|---|---|
| Baseline (raw cosine, no h2h fetch, no delta weight) | 0.377 |
| Full system | **0.935** |

The baseline completely missed every mid-grid pair (BOR vs LAW, PIA vs NOR) and every single-driver pole question. The h2h-first fetch guarantees those pairs enter the candidate pool; the delta reranking ensures the most diagnostic sections surface first.

### Answer quality — RAGAS Faithfulness

Evaluated on the same 18 questions using RAGAS with Claude Haiku as the evaluator.

| Metric | Score |
|---|---|
| RAGAS faithfulness | **0.872** |
| Per-claim audit (manual) | 0.941 (312 claims, 18 unsupported) |

The gap between the two numbers is explained by RAGAS being stricter on causal inferences. Manual audit of all 18 unsupported claims found **zero fabricated facts** — no invented lap times, positions, or driver history. All failures were either correct causal reasoning not literally stated in the retrieved text, or arithmetic derivations (e.g. 211m − 203m = 8m earlier throttle). Run `python tests/run_faithfulness_audit.py --fails` to see each flagged claim with the evaluator's reasoning.

---

## Tracing and evaluation with Phoenix

[Arize Phoenix](https://github.com/Arize-ai/phoenix) runs locally and records every question as a trace: the retrieved chunks with rank and distance, the prompt and the Claude call (tokens, latency). Tracing is off unless `PHOENIX_COLLECTOR_ENDPOINT` is set, so the deployed site never stores visitor questions, and the API key is never put on a span.

```bash
pip install -r requirements_evals.txt
python -m venv .phoenix-venv && .phoenix-venv/bin/pip install arize-phoenix
PHOENIX_WORKING_DIR=.phoenix .phoenix-venv/bin/phoenix serve        # UI on http://localhost:6006

PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 python query.py "Where did Hamilton lose time to Russell?"
python -m evals.annotate_traces                                      # attach the checks below to each trace
python -m evals.run_experiment --name my-change                      # run the 40-question set as an experiment
```

**What "good" means** is defined in code, in `evals/code_checks.py`, so the same input always gets the same verdict:

| Check | Fails when |
|---|---|
| `uncertainty_compliance` | a chunk says it cannot tell who braked later (a gap in the car data) and the answer states it anyway |
| `number_grounding` | a number with a unit (km/h, s, m) is not in the retrieved chunks or the lap comparison, or a difference / sum of numbers that are |
| `total_consistency` | a claimed total is not the net of the sections discussed, or one driver's gains |
| `retrieval_coverage`, `answer_coverage` | fewer than two of the three biggest-gap sections were retrieved / discussed |
| `inference_rate` | not pass/fail: the share of sentences using causal wording (reported as a trend) |

The experiment adds three more: the expected drivers were retrieved, unanswerable questions (race results, tyre degradation, fuel load) are declined, and answerable ones are not refused. Questions are in `evals/questions.json`. Each experiment is stored next to earlier ones in Phoenix, so a change to the prompt, chunk format or data is compared question by question.

---

## Adding a new session

```bash
# 1. Edit reingest.py — add your session to the SESSIONS list:
{"year": 2025, "circuit": "Silverstone", "short": "Silverstone"}

# 2. Run re-ingestion (OpenF1 car data + MultiViewer official corners, no login needed;
#    use --source fastf1 for the old FastF1 path, which needs an F1 TV subscription)
python reingest.py --chroma-path /tmp/new_store --only Silverstone

# 3. Check the store, then copy it over RAG_data_layers/chroma_store and commit
git add RAG_data_layers/chroma_store/
git commit -m "Add Silverstone 2025 qualifying"
```

Any circuit with an OpenF1 qualifying session (2023+) and a MultiViewer circuit entry works. Official corner data is cached under `RAG_data_layers/circuit_data/` so ingestion does not depend on the live service.

The `short` name must match the OpenF1 API `circuit_short_name` field. To find it:
```bash
curl "https://api.openf1.org/v1/sessions?year=2025&session_name=Qualifying"
```

---

## Project structure

```
f1-analytics-/
├── query.py                          # CLI entry point
├── reingest.py                       # Rebuild ChromaDB collections from scratch
├── requirements_core.txt             # Minimal dependencies
├── .env.example                      # API key template
│
├── server.py, web/                   # FastAPI app + website
├── RAG_data_layers/
│   ├── circuit_data/                 # Cached official corner data
│   ├── chroma_store/                 # Pre-built vector database (committed)
│   └── F1 Project Thoughts/
│       ├── rag_retrieval.py          # Retrieval logic (smart_retrieve, reranking)
│       ├── load_qualifying_openf1.py # OpenF1 session loader (default)
│       ├── load_circuit_official.py  # Official corners + outline (MultiViewer)
│       ├── load_qualifying.py        # FastF1 session loader (alternative)
│       ├── load_openf1.py            # OpenF1 weather + race control loader
│       └── load_track_data.py        # Track geometry loader
│
├── events_and_chunking/
│   ├── pipeline.py                   # Top-level: runs all chunkers for a session
│   ├── detect_events_official.py     # Sections from official corners + driver brake/throttle
│   ├── detect_events.py              # Geometry-only detection (fallback)
│   ├── circuit_labels.py             # Official FIA corner name mappings per circuit
│   ├── chunk_telemetry.py            # Head-to-head + per-driver event chunks
│   ├── chunk_driver.py               # Lap summary, sector, telemetry zone chunks
│   ├── chunk_session.py              # Weather + race control chunks
│   ├── chunk_circuit.py              # Circuit overview chunks
│   └── chunk_types.py                # Chunk dataclass and ID helpers
│
└── tests/                            # retrieval, server, loader, detector and braking-text tests
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | LLM API calls (Claude) |
| `chromadb` | Local vector store |
| `fastembed` | Embedding model (all-MiniLM-L6-v2, ONNX, no PyTorch) |
| `fastf1` | Optional alternative data source |
| `pandas` / `numpy` | Data processing |
| `requests` | OpenF1 API calls |

The embedding model (~40MB ONNX) downloads automatically on first run and is cached locally. No GPU required.

---

## Data sources

- **[OpenF1](https://openf1.org)** — free REST API: laps, car data (~3.7 Hz), positions, results, weather, race control (2023+)
- **[MultiViewer circuit API](https://multiviewer.app)** — official corner numbers and track outline per year (community service, cached locally)
- **[FastF1](https://github.com/theOehrly/Fast-F1)** — optional alternative source (needs an F1 TV subscription)
- **[bacinger circuit data](https://github.com/bacinger/f1-circuits)** — GPS circuit outlines and metadata
- **[TUMFTM track geometry](https://github.com/TUMFTM/racetrack-database)** — high-resolution x/y coordinates with track width for 8 circuits
