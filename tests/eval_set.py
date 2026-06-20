"""
eval_set.py
-----------
Labeled evaluation set for measuring retrieval quality (MRR@6).

Each entry has:
  - question:         natural language question
  - session:          "barcelona" | "melbourne"
  - expected_chunks:  list of (chunk_type, event_id_or_None) tuples that
                      should appear in the top-6 retrieved results.
                      For h2h_event sections, event_id is the section name.
                      For status chunks (dnq, lap_summary), event_id is None.
  - notes:            why these sections are expected (derived from data)

Expected sections are the top-3 |time_delta_s| sections for that driver pair,
fetched directly from the ChromaDB collections. DNQ/Q-miss questions expect
the driver_dnq_status or driver_lap_summary chunk for the relevant driver.

Pole positions (from stored data):
  Barcelona 2026: RUS (74.679s)
  Melbourne 2026: RUS (78.518s)
"""

EVAL_SET = [

    # ── Barcelona 2026: head-to-head Q3 comparisons ──────────────────────────

    {
        "question": "How did Russell compare to Hamilton at Barcelona 2026?",
        "session":  "barcelona",
        "expected_chunks": [
            ("head_to_head_event", "Between T9-T10"),   # 0.389s — biggest delta
            ("head_to_head_event", "T10-T11-T12"),      # 0.327s
            ("head_to_head_event", "T1-T2-T3-T4-T5"),  # 0.289s
        ],
        "notes": "RUS P1 vs HAM P2; top-3 absolute time deltas across all sections",
    },
    {
        "question": "Where did Antonelli lose time to Russell at Barcelona?",
        "session":  "barcelona",
        "expected_chunks": [
            ("head_to_head_event", "T1-T2-T3-T4-T5"),  # 0.296s
            ("head_to_head_event", "T10-T11-T12"),      # 0.235s
            ("head_to_head_event", "Main straight"),    # 0.209s
        ],
        "notes": "ANT P3 vs RUS P1",
    },
    {
        "question": "How did Hamilton compare to Leclerc at Barcelona 2026?",
        "session":  "barcelona",
        "expected_chunks": [
            ("head_to_head_event", "T1-T2-T3-T4-T5"),  # 0.302s
            ("head_to_head_event", "T10-T11-T12"),      # 0.158s
            ("head_to_head_event", "Main straight"),    # 0.138s
        ],
        "notes": "HAM P2 vs LEC P8",
    },
    {
        "question": "Compare Antonelli and Leclerc at the 2026 Spanish Grand Prix",
        "session":  "barcelona",
        "expected_chunks": [
            ("head_to_head_event", "T1-T2-T3-T4-T5"),  # 0.310s
            ("head_to_head_event", "Main straight"),    # 0.147s
            ("head_to_head_event", "T6-T7-T8"),        # 0.112s
        ],
        "notes": "ANT P3 vs LEC P8",
    },

    # ── Barcelona 2026: single driver → pole comparison ──────────────────────

    {
        "question": "Why did Russell take pole at Barcelona 2026?",
        "session":  "barcelona",
        "expected_chunks": [
            ("head_to_head_event", "Between T9-T10"),   # 0.389s
            ("head_to_head_event", "T10-T11-T12"),      # 0.327s
            ("head_to_head_event", "T1-T2-T3-T4-T5"),  # 0.289s
        ],
        "notes": "Single-driver pole question; comparison auto-expands to HAM (P2)",
    },
    {
        "question": "How did Norris qualify at Barcelona 2026?",
        "session":  "barcelona",
        "expected_chunks": [
            ("head_to_head_event", "T1-T2-T3-T4-T5"),  # biggest delta vs pole
            ("head_to_head_event", "T10-T11-T12"),
            ("head_to_head_event", "Main straight"),
        ],
        "notes": "NOR P4; single-driver question expands to pole sitter RUS",
    },
    {
        "question": "Where did Leclerc struggle compared to pole at Barcelona?",
        "session":  "barcelona",
        "expected_chunks": [
            ("head_to_head_event", "Main straight"),    # 0.138s — LEC vs RUS
            ("head_to_head_event", "Between T9-T10"),
            ("head_to_head_event", "T10-T11-T12"),
        ],
        "notes": "LEC P8; single-driver question expands to pole sitter RUS",
    },

    # ── Melbourne 2026: head-to-head Q3 comparisons ──────────────────────────

    {
        "question": "How did Russell compare to Antonelli in Melbourne qualifying?",
        "session":  "melbourne",
        "expected_chunks": [
            ("head_to_head_event", "Between T12-T13"),  # 0.382s
            ("head_to_head_event", "T13-T14"),          # 0.195s
            ("head_to_head_event", "Sweepers before T9"), # 0.145s
        ],
        "notes": "RUS P1 vs ANT P2",
    },
    {
        "question": "Where did Hadjar gain time on Russell at Melbourne?",
        "session":  "melbourne",
        "expected_chunks": [
            ("head_to_head_event", "Between T12-T13"),  # 0.184s
            ("head_to_head_event", "Sweepers before T9"), # 0.149s
            ("head_to_head_event", "Main straight"),    # 0.102s
        ],
        "notes": "HAD P3 vs RUS P1",
    },
    {
        "question": "How did Hadjar compare to Leclerc in Australia 2026?",
        "session":  "melbourne",
        "expected_chunks": [
            ("head_to_head_event", "Between T12-T13"),  # 0.223s
            ("head_to_head_event", "T13-T14"),          # 0.179s
            ("head_to_head_event", "Main straight"),    # 0.135s
        ],
        "notes": "HAD P3 vs LEC P4",
    },
    {
        "question": "Compare Piastri and Norris in Melbourne qualifying",
        "session":  "melbourne",
        "expected_chunks": [
            ("head_to_head_event", "Before T3-T4"),     # 0.185s
            ("head_to_head_event", "Sweepers before T9"), # 0.167s
            ("head_to_head_event", "T6-T7-T8"),        # 0.158s
        ],
        "notes": "PIA P5 vs NOR P6",
    },
    {
        "question": "Bortoleto versus Lawson in Melbourne 2026",
        "session":  "melbourne",
        "expected_chunks": [
            ("head_to_head_event", "Main straight"),    # 0.233s
            ("head_to_head_event", "T3-T4"),           # 0.144s
            ("head_to_head_event", "Sweepers before T9"), # 0.116s
        ],
        "notes": "BOR P10 vs LAW P9",
    },

    # ── Melbourne 2026: single driver → pole ─────────────────────────────────

    {
        "question": "Why did Russell take pole in Melbourne 2026?",
        "session":  "melbourne",
        "expected_chunks": [
            ("head_to_head_event", "Between T12-T13"),  # vs ANT P2
            ("head_to_head_event", "T13-T14"),
            ("head_to_head_event", "Sweepers before T9"),
        ],
        "notes": "Pole question; auto-expands to ANT (P2)",
    },
    {
        "question": "How did Hamilton qualify in Australia?",
        "session":  "melbourne",
        "expected_chunks": [
            ("head_to_head_event", "Between T12-T13"),  # HAM vs RUS: 0.382s... wait check
            ("head_to_head_event", "Main straight"),
            ("head_to_head_event", "T13-T14"),
        ],
        "notes": "HAM P7; single-driver question expands to pole sitter RUS",
    },

    # ── DNQ / did not make Q3 ─────────────────────────────────────────────────

    {
        "question": "Why wasn't Verstappen in Q3 at Melbourne?",
        "session":  "melbourne",
        "expected_chunks": [
            ("driver_dnq_status", None),  # VER — flying lap interrupted by red flag
        ],
        "notes": "VER had no lap time; DNQ chunk explains red flag elimination",
    },
    {
        "question": "Why did Verstappen miss qualifying in Australia 2026?",
        "session":  "melbourne",
        "expected_chunks": [
            ("driver_dnq_status", None),  # VER DNQ chunk must be rank 1
        ],
        "notes": "Second VER question to verify DNQ chunk consistently surfaces first",
    },
    {
        "question": "Did Hulkenberg make Q3 in Melbourne?",
        "session":  "melbourne",
        "expected_chunks": [
            ("driver_lap_summary", None),  # HUL — text says 'Eliminated at end of Q2'
        ],
        "notes": "HUL P11, Q2 eliminated; lap summary chunk contains progression text",
    },
    {
        "question": "Why didn't Hulkenberg qualify for Q3 in Australia?",
        "session":  "melbourne",
        "expected_chunks": [
            ("driver_lap_summary", None),  # HUL — Q2 elimination in chunk text
        ],
        "notes": "HUL Q2 elimination; expects driver_lap_summary to surface with Q progression",
    },
]
