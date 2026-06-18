# F1 Qualifying RAG

A retrieval-augmented generation system for analysing F1 qualifying sessions. Ask natural language questions about lap times, sector performance, braking points, and head-to-head driver comparisons — answers are grounded entirely in telemetry data from FastF1, with no hallucinated facts.

**Sessions included (pre-built, no re-ingestion needed):**
- Barcelona 2026 Qualifying — 703 chunks
- Monza 2024 Qualifying — 933 chunks

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
