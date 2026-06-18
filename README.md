# F1 Qualifying RAG

A retrieval-augmented generation system for analysing F1 qualifying sessions. Ask natural language questions about lap times, sector performance, braking points, and head-to-head driver comparisons — answers are grounded entirely in telemetry data from FastF1, with no hallucinated facts.

**Sessions included (pre-built, no re-ingestion needed):**
- Barcelona 2026 Qualifying — 703 chunks
- Monza 2024 Qualifying — 933 chunks

---

## Design Decisions & Contributions

This project started with raw telemetry data and no framework for what "context" means for an F1 question. The core intellectual work was defining that — what information an LLM actually needs to reason about a qualifying lap, and how to structure it so retrieval surfaces the right things.

**Defining the unit of analysis: the track section**
The first non-obvious decision was that a lap should be broken into spatial sections — not by time, not by sector, but by what happens on track. Corners were identified not as individual turns but as sequences (T1–T6 at Barcelona is one continuous complex with shared entry, multiple apices, and a single exit) because that's how drivers and engineers actually think about them. Straights are the recovery zones between them. This framing — derived from domain knowledge of how lap time is built — determined the entire chunk structure.

**Using braking and throttle as the structural signals**
Rather than hardcoding section boundaries by circuit, the system detects them from curvature geometry and validates them against braking and acceleration data in the telemetry. The LLM receives braking point (distance and entry speed), throttle pickup distance, and per-apex minimum speeds for every section. This gives it the information a race engineer would use to identify where time was gained or lost — not just that one driver was faster, but whether they carried more speed into the apex, braked later, or got on the throttle earlier on exit.

**Curating what data matters**
Every field in every chunk was a deliberate choice about what an LLM needs to reason about lap time: entry speed tells you how much energy a driver carried from the previous section, apex speed tells you how much they scrubbed off, exit speed tells you how well they converted that into acceleration. DRS status, gear range, and braking distance are included because they explain the *why* behind the numbers. Fields that don't change the answer were left out.

**Catching the overlap bug through domain knowledge**
During development, the section breakdown showed a 0.4s advantage for one driver on a single straight — but their overall lap gap was only 0.064s. That number was immediately suspicious: no driver gains four tenths on one straight against someone they only beat by six hundredths total. Investigation confirmed that the raw geometry detection was producing overlapping 80-metre zones at every corner-to-straight boundary, so each boundary region was being counted twice. The fix required normalising every boundary to a shared midpoint, interpolating times at the exact boundary distance rather than snapping to the nearest telemetry sample, and adding synthetic sections for the uncovered start and end of the lap. After the fix, section times sum to within 1ms of the actual lap time gap for every driver pair — the numbers close.

**Retrieval shaped by intent, not just keywords**
A question like "Why was Bottas slow?" names one driver. The system detects that the question is about pace relative to someone, infers a comparison driver from session standings (teammate, pole-sitter, or Q3 boundary depending on question type), and expands the retrieval filter to include that driver before the vector search runs. For two-driver questions, the filter restricts results to chunks where exactly those two drivers are compared head-to-head — preventing chunks about unrelated driver pairs from filling the context window.

**Time delta as a ranking signal**
Chunks where two drivers differ most in section time are promoted in ranking above chunks where they are nearly equal. A chunk showing a 0.3s difference through a corner complex is more likely to explain lap time than one showing 0.01s — so it surfaces first. This was the insight that moved the system from retrieving *related* content to retrieving *diagnostic* content.

**The result**
The project moved from a state where the vector database had no principled structure and retrieval returned whatever was semantically closest, to one where every retrieved chunk contains quantitatively precise, engineer-readable information, section sums match actual lap times, and the system can answer questions like "where exactly did Russell beat his teammate" with specific distances, speeds, and time deltas — derived entirely from stored telemetry with no external lookups.

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

Session is auto-detected from the question text. Mentions of "Barcelona", "Spain", "2026" route to the Barcelona collection; "Monza", "Italy", "2024" route to Monza.

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

---

## Adding a new session

```bash
# 1. Edit reingest.py — add your session to the SESSIONS list:
{"year": 2025, "circuit": "Silverstone", "short": "silverstone"}

# 2. Run re-ingestion (downloads FastF1 data, builds chunks, embeds, stores)
python reingest.py

# 3. Commit the updated chroma store
git add RAG_data_layers/chroma_store/
git commit -m "Add Silverstone 2025 qualifying"
```

Circuits with high-resolution track geometry (required for section detection):
Monza, Silverstone, Spa, Suzuka, Barcelona, Zandvoort, Austin, Melbourne

---

## Project structure

```
f1-analytics-/
├── query.py                          # CLI entry point
├── reingest.py                       # Rebuild ChromaDB collections from scratch
├── requirements_core.txt             # Minimal dependencies
├── .env.example                      # API key template
│
├── RAG_data_layers/
│   ├── chroma_store/                 # Pre-built vector database (committed)
│   └── F1 Project Thoughts/
│       ├── rag_retrieval.py          # Retrieval logic (smart_retrieve, reranking)
│       ├── load_qualifying.py        # FastF1 session loader
│       ├── load_openf1.py            # OpenF1 weather + race control loader
│       └── load_track_data.py        # Track geometry loader
│
├── events_and_chunking/
│   ├── pipeline.py                   # Top-level: runs all chunkers for a session
│   ├── detect_events.py              # Geometry-based track section detection
│   ├── chunk_telemetry.py            # Head-to-head + per-driver event chunks
│   ├── chunk_driver.py               # Lap summary, sector, telemetry zone chunks
│   ├── chunk_session.py              # Weather + race control chunks
│   ├── chunk_circuit.py              # Circuit overview chunks
│   └── chunk_types.py                # Chunk dataclass and ID helpers
│
└── tests/
    └── test_retrieval.py             # 24 retrieval test cases (24/24 passing)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | LLM API calls (Claude) |
| `chromadb` | Local vector store |
| `fastembed` | Embedding model (all-MiniLM-L6-v2, ONNX, no PyTorch) |
| `fastf1` | F1 telemetry and lap data |
| `pandas` / `numpy` | Data processing |
| `requests` | OpenF1 API calls |

The embedding model (~40MB ONNX) downloads automatically on first run and is cached locally. No GPU required.

---

## Data sources

- **[FastF1](https://github.com/theOehrly/Fast-F1)** — official F1 timing feed, car telemetry at ~18Hz
- **[OpenF1](https://openf1.org)** — free REST API for weather, race control, stint data (2023+)
- **[bacinger circuit data](https://github.com/bacinger/f1-circuits)** — GPS circuit outlines and metadata
- **[TUMFTM track geometry](https://github.com/TUMFTM/racetrack-database)** — high-resolution x/y coordinates with track width for 8 circuits
