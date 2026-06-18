"""
tests/test_retrieval.py
-----------------------
Retrieval evaluation for the F1 RAG system.

Tests that vector search surfaces the correct drivers and context for a range
of qualifying question types. Every test case prints its question and the full
retrieved chunks so you can manually evaluate quality alongside the pass/fail.

Covers two sessions:
  - f1_qualifying_2024_monza     (Monza 2024 — baseline)
  - f1_qualifying_2026_barcelona (Barcelona 2026 — most recent, includes red flag in Q3)

Usage:
    python tests/test_retrieval.py           # prints all questions + retrieval
    pytest tests/test_retrieval.py -v -s     # same output, pytest reporting
"""

import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "RAG_data_layers/F1 Project Thoughts"))

from rag_retrieval import retrieve_chunks, smart_retrieve, make_embed_fn

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_PATH              = ROOT / "RAG_data_layers/chroma_store"
COLLECTION_MONZA_2024    = "f1_qualifying_2024_monza"
COLLECTION_BARCELONA_2026 = "f1_qualifying_2026_barcelona"
N_RESULTS                = 5
TEAMMATE_GAP_THRESHOLD   = 0.3


# ── Session context ───────────────────────────────────────────────────────────

def load_session_context(collection) -> dict:
    """
    Derive standings, teammates, and Q3 cutoff positions directly from
    the driver_lap_summary chunks already stored in the Chroma collection.
    No hardcoded values — everything is read from the data.
    """
    results = collection.get(where={"chunk_type": "driver_lap_summary"})

    drivers = {}
    for meta in results["metadatas"]:
        drivers[meta["driver"]] = {
            "position":      meta["qualifying_position"],
            "lap_time_s":    meta.get("lap_time_seconds") or 0.0,
            "team":          meta.get("team", "Unknown"),
            "gap_to_pole_s": meta.get("gap_to_pole_seconds") or 0.0,
        }

    standings = sorted(drivers.items(), key=lambda x: x[1]["position"])

    teams = {}
    for driver, info in drivers.items():
        teams.setdefault(info["team"], []).append(driver)

    teammate_of = {}
    for team_drivers in teams.values():
        if len(team_drivers) == 2:
            a, b = team_drivers
            teammate_of[a] = b
            teammate_of[b] = a

    return {
        "drivers":      drivers,
        "standings":    standings,
        "teammate_of":  teammate_of,
        "pole":         standings[0][0],
        "p2":           standings[1][0],
        "q3_last":      standings[9][0]  if len(standings) >= 10 else None,
        "q3_miss":      standings[10][0] if len(standings) >= 11 else None,
    }


# ── Query routing helpers ─────────────────────────────────────────────────────

def resolve_comparison_driver(question: str, driver: str, ctx: dict) -> tuple[str, str]:
    """
    For single-driver questions (no second driver named), determine who to
    compare against and why. Returns (comparison_driver_code, reason_string).
    """
    q        = question.lower()
    info     = ctx["drivers"].get(driver, {})
    pos      = info.get("position", 99)
    teammate = ctx["teammate_of"].get(driver)
    pole     = ctx["pole"]
    p2       = ctx["p2"]

    if any(kw in q for kw in ["pole", "p1 ", "first place", "fastest lap", "quickest"]):
        if driver == pole:
            return p2,   f"{driver} IS on pole → compare to P2 ({p2})"
        return pole,     f"{driver} not on pole → compare to pole sitter ({pole})"

    if any(kw in q for kw in ["q2", "q3", "make it", "didn't make", "miss", "cut", "through", "eliminated", "qualify for"]):
        q3_last = ctx["q3_last"]
        q3_miss = ctx["q3_miss"]
        if pos <= 10:
            return q3_miss, f"{driver} made Q3 (P{pos}) → compare to first to miss ({q3_miss})"
        return q3_last,     f"{driver} missed Q3 (P{pos}) → compare to last to make it ({q3_last})"

    if any(kw in q for kw in ["slow", "off", "struggle", "behind", "gap", "lose", "lost", "weak", "worst", "pace"]):
        if teammate:
            tm_info  = ctx["drivers"].get(teammate, {})
            tm_pos   = tm_info.get("position", 99)
            gap      = abs(info.get("lap_time_s", 999) - tm_info.get("lap_time_s", 999))
            if pos > tm_pos and gap > TEAMMATE_GAP_THRESHOLD:
                return teammate, (
                    f"{driver} (P{pos}) is {gap:.3f}s slower than teammate "
                    f"{teammate} (P{tm_pos}) — gap > {TEAMMATE_GAP_THRESHOLD}s → compare to teammate"
                )
        return pole, (
            f"{driver} not significantly slower than teammate or is faster → compare to pole ({pole})"
        )

    return pole, f"No clear intent detected → default to pole comparison ({pole})"


# ── Chunk inspection helpers ──────────────────────────────────────────────────

def drivers_in_chunks(chunks: list[dict]) -> set[str]:
    found = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        for key in ("driver", "driver_a", "driver_b"):
            val = meta.get(key)
            if val:
                found.add(val)
    return found


def sessions_in_chunks(chunks: list[dict]) -> set[tuple]:
    sessions = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        c = meta.get("circuit")
        y = meta.get("year")
        if c and y:
            sessions.add((c, y))
    return sessions


# ── Test case definitions ─────────────────────────────────────────────────────

def build_monza_2024_cases(ctx: dict) -> list[dict]:
    pole    = ctx["pole"]
    p2      = ctx["p2"]
    q3_last = ctx["q3_last"]
    q3_miss = ctx["q3_miss"]

    bot_cmp, bot_reason = resolve_comparison_driver("Why was Bottas so slow", "BOT", ctx)
    per_cmp, per_reason = resolve_comparison_driver("Perez off the pace slow struggle", "PER", ctx)

    return [
        {
            "id": "monza24_h2h_001", "category": "explicit_h2h",
            "question": "Why was Norris faster than Piastri at Monza 2024 qualifying?",
            "expected": {"NOR", "PIA"},
            "desc": "Named driver pair + named session → both must appear in top-5",
        },
        {
            "id": "monza24_h2h_002", "category": "explicit_h2h",
            "question": "How did Russell compare to Hamilton at the 2024 Italian Grand Prix qualifying?",
            "expected": {"RUS", "HAM"},
            "desc": "Named pair + alternate session name → both must appear in top-5",
        },
        {
            "id": "monza24_h2h_003", "category": "explicit_h2h",
            "question": "What was the sector breakdown between Leclerc and Sainz at Monza 2024 qualifying?",
            "expected": {"LEC", "SAI"},
            "desc": "Ferrari teammates, sector focus + named session → both must appear",
        },
        {
            "id": "monza24_no_session_001", "category": "no_session",
            "question": "What was the gap between Norris and Piastri in their last qualifying session together?",
            "expected": {"NOR", "PIA"}, "same_session": True,
            "desc": "No session named → should resolve to Monza 2024",
        },
        {
            "id": "monza24_no_session_002", "category": "no_session",
            "question": "How did Leclerc and Sainz qualify against each other most recently?",
            "expected": {"LEC", "SAI"}, "same_session": True,
            "desc": "No session named → should resolve to Monza 2024",
        },
        {
            "id": "monza24_pole_001", "category": "single_driver_pole",
            "question": "How did Norris manage to secure pole position at Monza?",
            "driver": "NOR", "expected": {"NOR", p2},
            "desc": f"NOR is pole → compare to P2 ({p2})",
        },
        {
            "id": "monza24_pole_002", "category": "single_driver_pole",
            "question": "Why did Leclerc fail to get pole at the 2024 Italian GP?",
            "driver": "LEC", "expected": {"LEC", pole},
            "desc": f"LEC did not get pole → compare to pole sitter ({pole})",
        },
        {
            "id": "monza24_slow_001", "category": "single_driver_slow",
            "question": "Why was Bottas so slow in Monza qualifying?",
            "driver": "BOT", "expected": {"BOT", bot_cmp},
            "desc": f"BOT slowness → {bot_reason}",
        },
        {
            "id": "monza24_slow_002", "category": "single_driver_slow",
            "question": "Perez seemed well off the pace at Monza — what went wrong in qualifying?",
            "driver": "PER", "expected": {"PER", per_cmp},
            "desc": f"PER slowness → {per_reason}",
        },
        {
            "id": "monza24_cutoff_001", "category": "q3_cutoff",
            "question": "Did Hamilton make it through to Q3 at Monza 2024?",
            "driver": "HAM", "expected": {"HAM", q3_last, q3_miss},
            "desc": f"HAM Q3 check (made it) → must find HAM + boundary P10({q3_last})/P11({q3_miss})",
        },
        {
            "id": "monza24_cutoff_002", "category": "q3_cutoff",
            "question": "Did Alonso qualify for Q3 at the 2024 Italian Grand Prix?",
            "driver": "ALO", "expected": {"ALO", q3_last},
            "desc": f"ALO Q3 check (missed it) → must find ALO + last qualifier ({q3_last})",
        },
        {
            "id": "monza24_cutoff_003", "category": "q3_boundary",
            "question": "Who was the last driver to make Q3 and who just missed out at Monza 2024 qualifying?",
            "expected": {q3_last, q3_miss},
            "desc": f"Boundary question → expect P10 ({q3_last}) and P11 ({q3_miss}) in top-5",
        },
    ]


def build_barcelona_2026_cases(ctx: dict) -> list[dict]:
    """
    Barcelona 2026 qualifying test cases.
    Grid highlights: RUS pole (Mercedes), HAM P2 (Ferrari — moved from Mercedes),
    ANT P3 (Mercedes rookie), LEC P10 last in Q3 (Ferrari teammate of HAM),
    LIN P11 first Q3 miss (Racing Bulls). Q3 had a red flag and restart.
    """
    pole    = ctx["pole"]    # RUS
    p2      = ctx["p2"]      # HAM
    q3_last = ctx["q3_last"] # LEC (P10)
    q3_miss = ctx["q3_miss"] # LIN (P11)

    ant_cmp, ant_reason = resolve_comparison_driver("Antonelli pace struggle slow", "ANT", ctx)
    lec_cmp, lec_reason = resolve_comparison_driver("Leclerc off the pace slow", "LEC", ctx)

    return [
        # ── Explicit H2H ──────────────────────────────────────────────────────
        {
            "id": "bcn26_h2h_001", "category": "explicit_h2h",
            "question": "How did Russell compare to Hamilton in Barcelona 2026 qualifying?",
            "expected": {"RUS", "HAM"},
            "desc": "Old teammates now on different teams (Mercedes vs Ferrari) — both must appear",
        },
        {
            "id": "bcn26_h2h_002", "category": "explicit_h2h",
            "question": "What was the gap between Norris and Piastri at the 2026 Spanish Grand Prix qualifying?",
            "expected": {"NOR", "PIA"},
            "desc": "McLaren teammates, named session → both must appear",
        },
        {
            "id": "bcn26_h2h_003", "category": "explicit_h2h",
            "question": "How did Hamilton qualify against Leclerc at Barcelona 2026? They are Ferrari teammates.",
            "expected": {"HAM", "LEC"},
            "desc": "Ferrari teammates HAM/LEC (new 2026 pairing) — both must appear",
        },
        # ── No session named ──────────────────────────────────────────────────
        {
            "id": "bcn26_no_session_001", "category": "no_session",
            "question": "How did Russell and Hamilton compare in qualifying most recently?",
            "expected": {"RUS", "HAM"}, "same_session": True,
            "desc": "No session — should pull from Barcelona 2026, ex-teammates on rival teams",
        },
        # ── Pole intent ───────────────────────────────────────────────────────
        {
            "id": "bcn26_pole_001", "category": "single_driver_pole",
            "question": "How did Russell take pole in Barcelona 2026 qualifying?",
            "driver": "RUS", "expected": {"RUS", p2},
            "desc": f"RUS is pole → compare to P2 ({p2})",
        },
        {
            "id": "bcn26_pole_002", "category": "single_driver_pole",
            "question": "Why couldn't Hamilton get pole at the 2026 Spanish GP?",
            "driver": "HAM", "expected": {"HAM", pole},
            "desc": f"HAM P2, not pole → compare to pole sitter ({pole})",
        },
        # ── Slowness intent ───────────────────────────────────────────────────
        {
            "id": "bcn26_slow_001", "category": "single_driver_slow",
            "question": "Why was Antonelli off the pace compared to his teammate at Barcelona?",
            "driver": "ANT", "expected": {"ANT", ant_cmp},
            "desc": f"ANT slowness → {ant_reason}",
        },
        {
            "id": "bcn26_slow_002", "category": "single_driver_slow",
            "question": "Leclerc struggled in Barcelona 2026 qualifying — what went wrong?",
            "driver": "LEC", "expected": {"LEC", lec_cmp},
            "desc": f"LEC slowness → {lec_reason}",
        },
        # ── Q3 cutoff ─────────────────────────────────────────────────────────
        {
            "id": "bcn26_cutoff_001", "category": "q3_cutoff",
            "question": "Did Leclerc make it through to Q3 at the 2026 Barcelona qualifying?",
            "driver": "LEC", "expected": {"LEC", q3_last, q3_miss},
            "desc": f"LEC made Q3 (P10) → must find LEC + boundary P10({q3_last})/P11({q3_miss})",
        },
        {
            "id": "bcn26_cutoff_002", "category": "q3_cutoff",
            "question": "Did Lindblad qualify for Q3 at Barcelona 2026?",
            "driver": "LIN", "expected": {"LIN", q3_last},
            "desc": f"LIN missed Q3 (P11) → must find LIN + last qualifier ({q3_last})",
        },
        {
            "id": "bcn26_cutoff_003", "category": "q3_boundary",
            "question": "Who just made Q3 and who just missed out at the 2026 Spanish Grand Prix?",
            "expected": {q3_last, q3_miss},
            "desc": f"Boundary question → expect P10 ({q3_last}) and P11 ({q3_miss})",
        },
        # ── Red flag (session-specific event) ─────────────────────────────────
        {
            "id": "bcn26_event_001", "category": "explicit_h2h",
            "question": "What happened during the red flag in Q3 at Barcelona 2026 and how did it affect Russell's pole lap?",
            "expected": {"RUS"},
            "desc": "Q3 was red-flagged and restarted — race_control_event chunks should surface",
        },
    ]


# ── Pass/fail evaluation ──────────────────────────────────────────────────────

def evaluate(test: dict, chunks: list[dict]) -> tuple[bool, str]:
    found    = drivers_in_chunks(chunks)
    expected = test.get("expected", set())

    if test["category"] == "no_session":
        sessions = sessions_in_chunks(chunks)
        if len(sessions) > 1:
            return False, f"Chunks span multiple sessions: {sessions}"
        missing = expected - found
        if missing:
            return False, f"Missing drivers {missing} (found: {found})"
        return True, f"Single session ✓ | drivers found: {found}"

    if test["category"] in ("q3_cutoff", "q3_boundary"):
        driver   = test.get("driver")
        boundary = expected - ({driver} if driver else set())
        found_boundary = boundary & found
        if driver and driver not in found:
            return False, f"Primary driver {driver} not found (found: {found})"
        if not found_boundary:
            return False, f"No boundary driver from {boundary} found (found: {found})"
        return True, f"Found: {found & expected}"

    missing = expected - found
    if missing:
        return False, f"Missing {missing} (found: {found})"
    return True, f"All expected found: {found & expected}"


# ── Pretty printer ────────────────────────────────────────────────────────────

def _chunk_row(i: int, chunk: dict) -> str:
    meta  = chunk["metadata"]
    ctype = meta.get("chunk_type", "?")
    drv   = (meta.get("driver")
             or f'{meta.get("driver_a","?")} vs {meta.get("driver_b","?")}')
    event = f'{meta.get("circuit","?")} {meta.get("year","?")}'
    dist  = chunk["distance"]
    text  = chunk["text"][:110].replace("\n", " ")
    return (
        f"    {i}. [{ctype}] {drv} | {event} | dist={dist:.4f}\n"
        f"       \"{text}...\""
    )


# ── Standalone runner ─────────────────────────────────────────────────────────

def run_session(label: str, collection_name: str, build_cases_fn, embed_fn, client) -> tuple[int, int]:
    collection = client.get_collection(collection_name)
    print(f"\n{'='*72}")
    print(f"  SESSION: {label}  ({collection.count()} chunks)")
    print(f"{'='*72}")

    ctx = load_session_context(collection)
    print(f"Pole: {ctx['pole']}  |  P2: {ctx['p2']}")
    print(f"Q3 last (P10): {ctx['q3_last']}  |  Q3 miss (P11): {ctx['q3_miss']}")
    teammate_str = "  ".join(f"{a}↔{b}" for a, b in
                             {frozenset([k, v]) for k, v in ctx["teammate_of"].items()})
    print(f"Teammates: {teammate_str}\n")

    cases   = build_cases_fn(ctx)
    results = []

    for test in cases:
        chunks        = smart_retrieve(collection, test["question"], embed_fn, n_results=N_RESULTS, session_ctx=ctx)
        passed, reason = evaluate(test, chunks)
        results.append((test, chunks, passed, reason))

    n_pass = sum(1 for _, _, p, _ in results if p)
    n_fail = len(results) - n_pass

    print(f"  RESULT: {n_pass}/{len(results)} passed  |  {n_fail} failed\n")

    for test, chunks, passed, reason in results:
        tag = "✓ PASS" if passed else "✗ FAIL"
        print(f"{tag}  [{test['id']}]  ({test['category']})")
        print(f"  Question : {test['question']}")
        print(f"  Expected : {test.get('expected', '—')}")
        print(f"  Rule     : {test['desc']}")
        print(f"  Verdict  : {reason}")
        print(f"  Top {N_RESULTS} retrieved chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(_chunk_row(i, chunk))
        print()

    return n_pass, n_fail


def run_all() -> bool:
    print("Loading embedding model...")
    embed_fn = make_embed_fn()
    client   = chromadb.PersistentClient(path=str(CHROMA_PATH))

    total_pass, total_fail = 0, 0

    p, f = run_session(
        "Monza 2024", COLLECTION_MONZA_2024, build_monza_2024_cases, embed_fn, client
    )
    total_pass += p; total_fail += f

    p, f = run_session(
        "Barcelona 2026", COLLECTION_BARCELONA_2026, build_barcelona_2026_cases, embed_fn, client
    )
    total_pass += p; total_fail += f

    print("=" * 72)
    print(f"  OVERALL: {total_pass}/{total_pass+total_fail} passed  |  {total_fail} failed")
    print("=" * 72)
    return total_fail == 0


# ── pytest-compatible individual test functions ───────────────────────────────

_cache: dict = {}

def _fixtures(collection_name: str = COLLECTION_MONZA_2024):
    key = collection_name
    if key not in _cache:
        if "embed_fn" not in _cache:
            _cache["embed_fn"] = make_embed_fn()
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _cache[key] = {
            "collection": client.get_collection(collection_name),
        }
        _cache[key]["ctx"] = load_session_context(_cache[key]["collection"])
    return _cache["embed_fn"], _cache[key]["collection"], _cache[key]["ctx"]


def _run(question: str, expected: set[str], category: str = "",
         driver: str = "", collection_name: str = COLLECTION_MONZA_2024) -> tuple[bool, str, list]:
    embed_fn, collection, ctx = _fixtures(collection_name)
    chunks = smart_retrieve(collection, question, embed_fn, n_results=N_RESULTS, session_ctx=ctx)
    test   = {"expected": expected, "category": category, "driver": driver}
    passed, reason = evaluate(test, chunks)
    return passed, reason, chunks


# ── Monza 2024 tests ──────────────────────────────────────────────────────────

def test_monza24_h2h_norris_piastri():
    passed, reason, _ = _run(
        "Why was Norris faster than Piastri at Monza 2024 qualifying?",
        {"NOR", "PIA"}, "explicit_h2h",
    )
    assert passed, reason


def test_monza24_h2h_russell_hamilton():
    passed, reason, _ = _run(
        "How did Russell compare to Hamilton at the 2024 Italian Grand Prix qualifying?",
        {"RUS", "HAM"}, "explicit_h2h",
    )
    assert passed, reason


def test_monza24_h2h_leclerc_sainz_sectors():
    passed, reason, _ = _run(
        "What was the sector breakdown between Leclerc and Sainz at Monza 2024 qualifying?",
        {"LEC", "SAI"}, "explicit_h2h",
    )
    assert passed, reason


def test_monza24_no_session_norris_piastri():
    passed, reason, _ = _run(
        "What was the gap between Norris and Piastri in their last qualifying session together?",
        {"NOR", "PIA"}, "no_session",
    )
    assert passed, reason


def test_monza24_no_session_leclerc_sainz():
    passed, reason, _ = _run(
        "How did Leclerc and Sainz qualify against each other most recently?",
        {"LEC", "SAI"}, "no_session",
    )
    assert passed, reason


def test_monza24_pole_norris_is_pole():
    _, _, ctx = _fixtures()
    passed, reason, _ = _run(
        "How did Norris manage to secure pole position at Monza?",
        {"NOR", ctx["p2"]}, "single_driver_pole", "NOR",
    )
    assert passed, reason


def test_monza24_pole_leclerc_not_pole():
    _, _, ctx = _fixtures()
    passed, reason, _ = _run(
        "Why did Leclerc fail to get pole at the 2024 Italian GP?",
        {"LEC", ctx["pole"]}, "single_driver_pole", "LEC",
    )
    assert passed, reason


def test_monza24_slow_bottas():
    _, _, ctx = _fixtures()
    cmp, _ = resolve_comparison_driver("Why was Bottas so slow", "BOT", ctx)
    passed, reason, _ = _run(
        "Why was Bottas so slow in Monza qualifying?",
        {"BOT", cmp}, "single_driver_slow", "BOT",
    )
    assert passed, reason


def test_monza24_slow_perez():
    _, _, ctx = _fixtures()
    cmp, _ = resolve_comparison_driver("Perez off the pace slow struggle", "PER", ctx)
    passed, reason, _ = _run(
        "Perez seemed well off the pace at Monza — what went wrong in qualifying?",
        {"PER", cmp}, "single_driver_slow", "PER",
    )
    assert passed, reason


def test_monza24_q3_hamilton_made_it():
    _, _, ctx = _fixtures()
    passed, reason, _ = _run(
        "Did Hamilton make it through to Q3 at Monza 2024?",
        {"HAM", ctx["q3_last"], ctx["q3_miss"]}, "q3_cutoff", "HAM",
    )
    assert passed, reason


def test_monza24_q3_alonso_missed():
    _, _, ctx = _fixtures()
    passed, reason, _ = _run(
        "Did Alonso qualify for Q3 at the 2024 Italian Grand Prix?",
        {"ALO", ctx["q3_last"]}, "q3_cutoff", "ALO",
    )
    assert passed, reason


def test_monza24_q3_boundary():
    _, _, ctx = _fixtures()
    passed, reason, _ = _run(
        "Who was the last driver to make Q3 and who just missed out at Monza 2024 qualifying?",
        {ctx["q3_last"], ctx["q3_miss"]}, "q3_boundary",
    )
    assert passed, reason


# ── Barcelona 2026 tests ──────────────────────────────────────────────────────

def test_bcn26_h2h_russell_hamilton():
    passed, reason, _ = _run(
        "How did Russell compare to Hamilton in Barcelona 2026 qualifying?",
        {"RUS", "HAM"}, "explicit_h2h",
        collection_name=COLLECTION_BARCELONA_2026,
    )
    assert passed, reason


def test_bcn26_h2h_norris_piastri():
    passed, reason, _ = _run(
        "What was the gap between Norris and Piastri at the 2026 Spanish Grand Prix qualifying?",
        {"NOR", "PIA"}, "explicit_h2h",
        collection_name=COLLECTION_BARCELONA_2026,
    )
    assert passed, reason


def test_bcn26_h2h_hamilton_leclerc_ferrari_teammates():
    passed, reason, _ = _run(
        "How did Hamilton qualify against Leclerc at Barcelona 2026? They are Ferrari teammates.",
        {"HAM", "LEC"}, "explicit_h2h",
        collection_name=COLLECTION_BARCELONA_2026,
    )
    assert passed, reason


def test_bcn26_no_session_russell_hamilton():
    passed, reason, _ = _run(
        "How did Russell and Hamilton compare in qualifying most recently?",
        {"RUS", "HAM"}, "no_session",
        collection_name=COLLECTION_BARCELONA_2026,
    )
    assert passed, reason


def test_bcn26_pole_russell_is_pole():
    _, _, ctx = _fixtures(COLLECTION_BARCELONA_2026)
    passed, reason, _ = _run(
        "How did Russell take pole in Barcelona 2026 qualifying?",
        {"RUS", ctx["p2"]}, "single_driver_pole", "RUS",
        collection_name=COLLECTION_BARCELONA_2026,
    )
    assert passed, reason


def test_bcn26_pole_hamilton_not_pole():
    _, _, ctx = _fixtures(COLLECTION_BARCELONA_2026)
    passed, reason, _ = _run(
        "Why couldn't Hamilton get pole at the 2026 Spanish GP?",
        {"HAM", ctx["pole"]}, "single_driver_pole", "HAM",
        collection_name=COLLECTION_BARCELONA_2026,
    )
    assert passed, reason


def test_bcn26_slow_antonelli():
    _, _, ctx = _fixtures(COLLECTION_BARCELONA_2026)
    cmp, _ = resolve_comparison_driver("Antonelli pace struggle slow", "ANT", ctx)
    passed, reason, _ = _run(
        "Why was Antonelli off the pace compared to his teammate at Barcelona?",
        {"ANT", cmp}, "single_driver_slow", "ANT",
        collection_name=COLLECTION_BARCELONA_2026,
    )
    assert passed, reason


def test_bcn26_slow_leclerc():
    _, _, ctx = _fixtures(COLLECTION_BARCELONA_2026)
    cmp, _ = resolve_comparison_driver("Leclerc off the pace slow", "LEC", ctx)
    passed, reason, _ = _run(
        "Leclerc struggled in Barcelona 2026 qualifying — what went wrong?",
        {"LEC", cmp}, "single_driver_slow", "LEC",
        collection_name=COLLECTION_BARCELONA_2026,
    )
    assert passed, reason


def test_bcn26_q3_leclerc_made_it():
    _, _, ctx = _fixtures(COLLECTION_BARCELONA_2026)
    passed, reason, _ = _run(
        "Did Leclerc make it through to Q3 at the 2026 Barcelona qualifying?",
        {"LEC", ctx["q3_last"], ctx["q3_miss"]}, "q3_cutoff", "LEC",
        collection_name=COLLECTION_BARCELONA_2026,
    )
    assert passed, reason


def test_bcn26_q3_lindblad_missed():
    _, _, ctx = _fixtures(COLLECTION_BARCELONA_2026)
    passed, reason, _ = _run(
        "Did Lindblad qualify for Q3 at Barcelona 2026?",
        {"LIN", ctx["q3_last"]}, "q3_cutoff", "LIN",
        collection_name=COLLECTION_BARCELONA_2026,
    )
    assert passed, reason


def test_bcn26_q3_boundary():
    _, _, ctx = _fixtures(COLLECTION_BARCELONA_2026)
    passed, reason, _ = _run(
        "Who just made Q3 and who just missed out at the 2026 Spanish Grand Prix?",
        {ctx["q3_last"], ctx["q3_miss"]}, "q3_boundary",
        collection_name=COLLECTION_BARCELONA_2026,
    )
    assert passed, reason


def test_bcn26_red_flag_event():
    passed, reason, _ = _run(
        "What happened during the red flag in Q3 at Barcelona 2026 and how did it affect Russell's pole lap?",
        {"RUS"}, "explicit_h2h",
        collection_name=COLLECTION_BARCELONA_2026,
    )
    assert passed, reason


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
