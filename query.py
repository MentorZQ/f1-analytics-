"""
query.py
--------
CLI for the F1 qualifying RAG system.

Usage:
    python query.py "Why did Russell take pole at Barcelona 2026?"
    python query.py --session monza "How did Norris beat Piastri?"
    python query.py --chunks 8 "Where did Antonelli lose time to Russell?"
    python query.py  # interactive mode — prompts for questions until Ctrl+C

Requires:
    ANTHROPIC_API_KEY set in environment or .env file.

Sessions available:
    barcelona  →  f1_qualifying_2026_barcelona
    monza      →  f1_qualifying_2024_monza
"""

import argparse
import contextvars
import os
import re
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import chromadb

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "RAG_data_layers/F1 Project Thoughts"))

from rag_retrieval import (smart_retrieve, make_embed_fn, format_prompt,
                           resolve_context_drivers, question_drivers, named_teams)

import tracing

# The pair sections are rebuilt from a saved snapshot; if the snapshot code cannot be imported, the stored chunks serve.
try:
    from events_and_chunking.session_snapshot import clean_metadata, compare_drivers, load_snapshot
    from events_and_chunking import session_queries as sq
except Exception:                                    # pragma: no cover - a missing optional dependency must not break answering
    compare_drivers = load_snapshot = clean_metadata = sq = None


# ── Session registry ──────────────────────────────────────────────────────────

SESSIONS = {
    "barcelona": "f1_qualifying_2026_barcelona",
    "melbourne": "f1_qualifying_2026_melbourne",
    "monza":     "f1_qualifying_2024_monza",
}

# Words used to auto-detect the session from the question. Matched as whole words, so "oz" does not
# fire inside another word; short nicknames are listed explicitly.
_SESSION_KEYWORDS = {
    "barcelona": ["barcelona", "barca", "barc", "barna", "spain", "spanish", "catalunya", "catalan", "montmelo"],
    "melbourne": ["melbourne", "australia", "australian", "aussie", "aus", "oz", "albert park", "down under"],
    "monza":     ["monza", "italy", "italian", "2024"],
}


def detect_session(question: str) -> str:
    """Return the most likely session key based on keywords in the question."""
    q = question.lower()
    scores = {name: sum(re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", q) is not None for kw in kws)
              for name, kws in _SESSION_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    # If nothing matched, default to the most recent session
    return best if scores[best] > 0 else "barcelona"


# ── Session context ───────────────────────────────────────────────────────────

def parse_speed_trap(summary: str) -> float | None:
    """The speed-trap reading on the driver's best lap, from the lap summary text."""
    m = re.search(r"Speed trap \(speed trap\):\s*([\d.]+)\s*km/h", summary or "")
    return float(m.group(1)) if m else None


def parse_stage(summary: str) -> str | None:
    """Last qualifying session the driver reached ('Q1' / 'Q2' / 'Q3'), from 'Qualifying progression: ...'."""
    m = re.search(r"Qualifying progression:\s*([^\n]+)", summary or "")
    if not m:
        return None
    line = m.group(1)
    if "Qualified for Q3" in line:
        return "Q3"
    out = re.search(r"Eliminated at end of Q([12])", line)
    return f"Q{out.group(1)}" if out else None


def parse_session_times(summary: str) -> dict[int, float]:
    """Best time in each qualifying session ({1: seconds, 2: seconds, 3: seconds}; a session with no time is absent)."""
    m = re.search(r"Best time in each session:\s*([^\n]+)", summary or "")
    out = {}
    if m:
        for n, minutes, seconds in re.findall(r"Q([123]) (\d+):(\d+\.\d+)", m.group(1)):
            out[int(n)] = int(minutes) * 60 + float(seconds)
    return out


def load_session_context(collection) -> dict:
    """Build session context from driver_lap_summary chunks stored in the collection."""
    results = collection.get(
        where={"chunk_type": "driver_lap_summary"},
        include=["metadatas", "documents"],
    )
    drivers = {}
    for m, doc in zip(results["metadatas"], results["documents"]):
        if "driver" not in m:
            continue
        drivers[m["driver"]] = {
            "lap_time_seconds":    m.get("lap_time_seconds"),
            "qualifying_position": m.get("qualifying_position"),
            "team":                m.get("team", ""),
            "speed_trap_kmh":      parse_speed_trap(doc),
            "stage":               parse_stage(doc),
            "session_times":       parse_session_times(doc),
        }

    standings = sorted(drivers.items(), key=lambda x: x[1]["qualifying_position"] or 99)

    teams: dict[str, list[str]] = {}
    for drv, info in drivers.items():
        t = info.get("team", "")
        if t:
            teams.setdefault(t, []).append(drv)

    teammate_of: dict[str, str] = {}
    for drvs in teams.values():
        for d in drvs:
            mates = [x for x in drvs if x != d]
            if mates:
                teammate_of[d] = mates[0]

    return {
        "drivers":      drivers,
        "standings":    [d for d, _ in standings],
        "teammate_of":  teammate_of,
        "pole":         standings[0][0] if standings else None,
        "p2":           standings[1][0] if len(standings) > 1 else None,
    }


# ── Headline lap comparison ───────────────────────────────────────────────────

def _lap_text(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:06.3f}"


def _like_for_like(a: str, b: str, ia: dict, ib: dict) -> list[str]:
    """
    A best lap can come from different sessions: a driver out in Q1 has only a Q1 lap, and a driver's best is not always
    his last session (Lindblad's Q3 lap was slower than his Q2 lap). When the two best laps come from different sessions the
    gap between them is not like-for-like, so give the gap in every session both set a time in. If one driver went out
    earlier, the last session both ran is the like-for-like one.
    """
    ta, tb = ia.get("session_times") or {}, ib.get("session_times") or {}
    if not ta or not tb:
        return []
    best_a, best_b = min(ta, key=ta.get), min(tb, key=tb.get)
    shared = sorted(set(ta) & set(tb))
    if best_a == best_b or not shared:
        return []

    def per_session(n: int) -> str:
        faster = a if ta[n] <= tb[n] else b
        return f"Q{n}: {a} {_lap_text(ta[n])}, {b} {_lap_text(tb[n])} ({faster} {abs(ta[n] - tb[n]):.3f}s quicker)"

    lines = [f"Their best laps are from different sessions ({a}'s from Q{best_a}, {b}'s from Q{best_b}), "
             f"so the gap between them is not like-for-like.",
             "Session by session (each session in which both set a time): " + "; ".join(per_session(n) for n in shared) + "."]
    stage_a, stage_b = ia.get("stage"), ib.get("stage")
    if stage_a and stage_b and stage_a != stage_b:
        earlier, out = (a, stage_a) if stage_a < stage_b else (b, stage_b)
        n = int(out[1])
        lines.append(f"{earlier} went out in {out}, so {out} is the last session both ran and the like-for-like gap is {per_session(n)}.")
    return lines


def lap_comparison_chunk(deltas: dict | None, session_ctx: dict) -> dict | None:
    """
    The one fact every two-driver answer needs: the two best laps and the gap between them. Section chunks
    explain where time was won, but never state the total, so answers to "why didn't X get pole?" never
    said by how much.
    """
    if not deltas:
        return None
    a, b = deltas["driver_a"], deltas["driver_b"]
    info = session_ctx.get("drivers", {})
    ia, ib = info.get(a), info.get(b)
    if not ia or not ib or not ia.get("lap_time_seconds") or not ib.get("lap_time_seconds"):
        return None
    ta, tb = ia["lap_time_seconds"], ib["lap_time_seconds"]
    pa, pb = ia.get("qualifying_position"), ib.get("qualifying_position")
    quicker, slower = (a, b) if ta <= tb else (b, a)
    lines = [f"Lap comparison: {a} vs {b}",
             f"{a}: best lap {_lap_text(ta)}, classified P{pa}",
             f"{b}: best lap {_lap_text(tb)}, classified P{pb}",
             f"Over the lap {quicker} was {abs(ta - tb):.3f}s quicker than {slower}."]
    if pa and pb and (pa < pb) != (ta < tb):
        lines.append("Note: the classification order differs from the best-lap order for these two drivers.")
    lines += _like_for_like(a, b, ia, ib)
    return {"chunk_id": f"lap_comparison:{a}-{b}", "text": "\n".join(lines), "distance": 0.0, "rank": 0,
            "metadata": {"chunk_type": "lap_comparison", "driver_a": a, "driver_b": b}}


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(prompt: str, model: str = "claude-sonnet-4-6", api_key: str | None = None) -> str:
    """Send the prompt to the Anthropic API and return the response text.

    api_key overrides ANTHROPIC_API_KEY for this call only; it is never stored.
    """
    try:
        import anthropic
    except ImportError:
        return (
            "[anthropic package not installed — run: pip install anthropic]\n\n"
            "Prompt that would have been sent:\n\n" + prompt
        )

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "[ANTHROPIC_API_KEY not set — add it to .env or export it]\n\n"
            "Prompt that would have been sent:\n\n" + prompt
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Split the format_prompt output into system + user parts
    # format_prompt returns: "You are an F1... Answer:\n\nQuestion: ...\n\nRetrieved context:\n..."
    # We send system instructions separately for better results
    system = (
        "You are an expert F1 qualifying analyst. "
        "Answer questions using ONLY the retrieved telemetry and session data provided. "
        "Be precise with numbers — quote lap times, speeds, and deltas directly from the context. "
        "If the context does not contain enough information to answer fully, say so explicitly "
        "rather than guessing. "
        "Some chunks say 'cannot tell', give a range, or note a gap in the car data: "
        "state that uncertainty in one short clause and never turn it into a firm claim about who braked or lifted earlier. "
        "Then keep explaining: build the answer from what the data does show (time deltas, entry, apex, "
        "minimum and exit speeds, throttle application, top speed) and say what those measurements imply. "
        "A missing detail is never a reason to skip the explanation of a section. "
        "Do not add up sections or compute new totals unless the context gives the total; quote the figures as given. "
        "Do not attribute differences to car or engine characteristics (power unit, aero, downforce): the data cannot show them."
    )

    extra = {}
    # Haiku 5.5 thinks by default, and the thinking eats the 1024-token budget (measured: answers cut to ~380 characters),
    # so it answers without thinking. F1_DISABLE_THINKING=1 does the same for any other model.
    if model.startswith("claude-haiku-5") or os.environ.get("F1_DISABLE_THINKING") == "1":
        extra["thinking"] = {"type": "disabled"}
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        **extra,
    )
    usage = getattr(message, "usage", None)
    _LLM_USAGE.set({
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "model": model,
    })
    return "".join(block.text for block in message.content if block.type == "text")


# Token usage of the most recent call_llm in this context. call_llm returns plain text (tests and
# callers rely on that), so usage is passed beside it; None when the call was mocked or failed.
_LLM_USAGE: contextvars.ContextVar = contextvars.ContextVar("llm_usage", default=None)


# ── Shared resources ──────────────────────────────────────────────────────────
# Built once per process: loading the embedding model takes seconds, which a
# long-running server should not repeat on every request.

class SessionError(Exception):
    """The requested session is unknown or its collection is missing."""


@lru_cache(maxsize=1)
def get_embed_fn():
    return make_embed_fn()


@lru_cache(maxsize=1)
def _get_chroma_client():
    chroma_host = os.environ.get("CHROMA_HOST")
    if chroma_host:
        return chromadb.HttpClient(host=chroma_host, port=int(os.environ.get("CHROMA_PORT", "8000")))
    return chromadb.PersistentClient(
        path=os.environ.get("CHROMA_PATH") or str(ROOT / "RAG_data_layers/chroma_store")
    )


def _get_collection(session_key: str):
    collection_name = SESSIONS.get(session_key)
    if collection_name is None:
        raise SessionError(f"Unknown session '{session_key}'. Available: {list(SESSIONS)}")
    try:
        return _get_chroma_client().get_collection(collection_name)
    except Exception as exc:
        raise SessionError(
            f"Collection '{collection_name}' not found in the chroma store. "
            f"Run: python reingest.py"
        ) from exc


@lru_cache(maxsize=len(SESSIONS))
def _get_session_context(session_key: str) -> dict:
    return load_session_context(_get_collection(session_key))


# ── Pair sections generated on demand ─────────────────────────────────────────

SNAPSHOT_DIR = ROOT / "RAG_data_layers" / "session_snapshots"
PAIR_SECTIONS = 4        # the pair's sections with the largest time gaps that go into the context


@lru_cache(maxsize=len(SESSIONS))
def _get_snapshot(session_key: str):
    if load_snapshot is None or session_key not in SESSIONS:
        return None
    path = SNAPSHOT_DIR / f"{SESSIONS[session_key]}.npz"
    return load_snapshot(path) if path.exists() else None


@lru_cache(maxsize=512)
def pair_chunks(session_key: str, a: str, b: str) -> tuple[dict, ...]:
    """Every head-to-head section chunk for the pair, in the same form retrieval returns them."""
    snapshot = _get_snapshot(session_key)
    if snapshot is None:
        return ()
    out = []
    for c in compare_drivers(snapshot, a, b):
        out.append({"chunk_id": c.chunk_id, "text": c.text, "distance": 0.0, "rank": 0,
                    "metadata": clean_metadata({"chunk_type": c.chunk_type, "source": c.source, **c.metadata})})
    return tuple(out)


def top_pair_sections(session_key: str, wanted: set[str], k: int = PAIR_SECTIONS, pin: tuple[str, ...] = ()) -> list[dict]:
    """
    The k sections where the two wanted drivers differ most in time, or [] if the pair cannot be built.
    Sections in `pin` (the corners the question names) come first whatever their gap, then the rest by gap.
    """
    if len(wanted) != 2:
        return []
    a, b = sorted(wanted)
    sections = pair_chunks(session_key, a, b)
    pinned = [c for event_id in pin for c in sections if c["metadata"].get("event_id") == event_id][:k]
    rest = sorted((c for c in sections if c not in pinned), key=lambda c: -abs(c["metadata"].get("time_delta_s") or 0))
    return [dict(c) for c in (pinned + rest)[:k]]          # copies: the cached chunks are shared between requests


def pinned_sections(question: str, session_key: str) -> tuple[str, ...]:
    """Section ids of the corners the question names ("T9", "turn 3", "the first / final corner")."""
    if sq is None:
        return ()
    sections = _section_times(session_key)
    ids = []
    for number in sq.corner_numbers_asked(question, sections):
        ids += [s["id"] for s in sq.sections_for_corner(sections, number)[:1]]
    return tuple(dict.fromkeys(ids))


# ── Whole-session views (grid, teams, sections across the field) ───────────────

EXTRA_VIEWS = 3          # at most this many whole-session blocks go into one prompt
MIN_RETRIEVED = 2        # chunks left for ordinary retrieval whatever else is added


@lru_cache(maxsize=len(SESSIONS))
def _section_times(session_key: str):
    snapshot = _get_snapshot(session_key)
    return sq.section_times(snapshot) if snapshot is not None and sq is not None else []


def split_budget(n_chunks: int, n_extras_available: int, pair_available: bool) -> tuple[int, int, int]:
    """
    How many context slots go to (whole-session blocks, the pair's sections, ordinary retrieval).
    Whole-session blocks come first, then the pair's biggest-gap sections, and ordinary retrieval always keeps
    at least MIN_RETRIEVED slots. The three always add up to n_chunks.
    """
    extras = min(n_extras_available, max(n_chunks - MIN_RETRIEVED, 0))
    pair = min(PAIR_SECTIONS, max(n_chunks - extras - MIN_RETRIEVED, 0)) if pair_available else 0
    return extras, pair, n_chunks - extras - pair


def extra_context(question: str, session_key: str, session_ctx: dict) -> list[dict]:
    """
    Whole-session blocks for questions that are not about one pair: the grid ("why were drivers so close"),
    a team ("the ferraris"), the team advantage by section ("where did each team have an advantage"), and
    one section across the field ("who was fastest through T9"). Computed from the session, not stored.
    """
    if sq is None:
        return []
    snapshot = _get_snapshot(session_key)
    teams = named_teams(question, session_ctx)
    asked = sq.plan(question, teams, no_driver_named=not question_drivers(question, session_ctx))
    label = "{circuit} {year}".format(**snapshot.session_info) if snapshot is not None else ""
    out: list[dict] = []
    if asked["team_advantage"] and snapshot is not None:
        text = sq.team_advantage_text(snapshot, session_ctx)
        if text:
            out.append(sq.make_chunk("team_advantage", text, "all", session_key))
    for team in asked["teams"][:2]:
        text = sq.team_text(session_ctx, team)
        if text:
            out.append(sq.make_chunk("team_summary", text, team, session_key))
    if asked["section_best"] and snapshot is not None:
        sections = _section_times(session_key)
        for number in sq.corner_numbers_asked(question, sections)[:2]:
            for section in sq.sections_for_corner(sections, number)[:1]:
                text = sq.section_ranking_text(section, session_ctx)
                if text:
                    out.append(sq.make_chunk("section_ranking", text, section["id"], session_key))
    if asked["corner_facts"] and snapshot is not None:
        text = sq.corner_facts_text(_section_times(session_key))
        if text:
            out.append(sq.make_chunk("circuit_facts", text, "corners", session_key))
    if asked["grid"]:
        text = sq.grid_text(session_ctx, label)
        if text:
            out.append(sq.make_chunk("grid", text, "all", session_key))
    return out[:EXTRA_VIEWS]


# ── Section-by-section lap comparison ─────────────────────────────────────────

_DISTANCE_WINDOW = re.compile(r"Distance window:\s*([\d.]+)m\s*[–-]\s*([\d.]+)m")


def section_deltas(collection, chunks: list[dict], wanted: set[str] | None = None, session_key: str | None = None) -> dict | None:
    """Compare a whole lap for the driver pair most represented in `chunks`.

    `wanted` is the set of drivers the question is about (named or inferred). When given, only a
    comparison between two of them is returned: a head-to-head chunk that merely happened to be
    retrieved (a teammate pair surfaced by a "max speed" question) must not become the table the
    reader sees. A wanted pair with no stored comparison gives None.

    Fetches every section for the pair rather than only the retrieved ones, so
    the result covers the full lap and lap_gap_s sums to the real lap gap.

    gain_s is positive where driver_a was faster. Stored time_delta_s is the
    opposite (driver_a's section time minus driver_b's).

    Returns None when no retrieved chunk compares two drivers.
    """
    pairs = Counter(
        frozenset((c["metadata"]["driver_a"], c["metadata"]["driver_b"]))
        for c in chunks
        if c["metadata"].get("chunk_type") == "head_to_head_event"
    )
    if wanted is not None:
        pairs = Counter({pair: n for pair, n in pairs.items() if pair <= wanted})
        if not pairs and len(wanted) == 2:
            pairs = Counter({frozenset(wanted): 1})        # nothing retrieved for them: look the pair up directly
    if not pairs:
        return None
    first, second = sorted(pairs.most_common(1)[0][0])

    generated = pair_chunks(session_key, first, second) if session_key else ()
    if generated:
        stored = {"documents": [c["text"] for c in generated], "metadatas": [c["metadata"] for c in generated]}
    else:
        stored = collection.get(
            where={"$and": [
                {"chunk_type": "head_to_head_event"},
                {"$or": [
                    {"$and": [{"driver_a": first}, {"driver_b": second}]},
                    {"$and": [{"driver_a": second}, {"driver_b": first}]},
                ]},
            ]},
            include=["documents", "metadatas"],
        )

    if not stored["metadatas"]:
        return None                                        # this pair has no stored comparison

    # Present the higher-qualified driver first, whichever way the pair is stored.
    meta0 = stored["metadatas"][0]
    driver_a, driver_b = meta0["driver_a"], meta0["driver_b"]
    if (meta0.get("pos_a") or 99) > (meta0.get("pos_b") or 99):
        driver_a, driver_b = driver_b, driver_a

    retrieved = {
        c["metadata"].get("event_id")
        for c in chunks
        if c["metadata"].get("chunk_type") == "head_to_head_event"
        and {c["metadata"].get("driver_a"), c["metadata"].get("driver_b")} == {driver_a, driver_b}
    }

    sections: dict[str, dict] = {}
    for document, meta in zip(stored["documents"], stored["metadatas"]):
        delta = meta.get("time_delta_s")
        if delta is None or meta["event_id"] in sections:
            continue
        window = _DISTANCE_WINDOW.search(document)
        sections[meta["event_id"]] = {
            "section":   meta["event_id"],
            "type":      meta.get("event_type"),
            "gain_s":    round(-delta if meta["driver_a"] == driver_a else delta, 3),
            "start_m":   float(window.group(1)) if window else None,
            "end_m":     float(window.group(2)) if window else None,
            "retrieved": meta["event_id"] in retrieved,
        }

    ordered = sorted(
        sections.values(),
        key=lambda s: (s["start_m"] is None, s["start_m"] or 0),
    )
    return {
        "driver_a":  driver_a,
        "driver_b":  driver_b,
        "lap_gap_s": round(sum(s["gain_s"] for s in ordered), 3),
        "sections":  ordered,
    }


# ── Main answer functions ─────────────────────────────────────────────────────

def answer_with_sources(
    question: str,
    session_key: str | None = None,
    n_chunks: int = 6,
    model: str = "claude-sonnet-4-6",
    api_key: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    Full RAG pipeline, returning the answer and the evidence behind it.

    Args:
        question:    Natural language question about F1 qualifying.
        session_key: A key of SESSIONS. Auto-detected from question if None.
        n_chunks:    Number of chunks to retrieve and pass to the LLM.
        model:       Anthropic model ID.
        api_key:     Overrides ANTHROPIC_API_KEY for this call only.
        verbose:     Print retrieved chunks before the answer.

    Returns:
        {"answer", "session", "sources", "deltas"}

    Raises:
        SessionError: the session is unknown or its collection is missing.
    """
    tracing.init_tracing()
    with tracing.span("f1.query", "CHAIN", input_value=question) as root:
        result = _answer_with_sources(question, session_key, n_chunks, model, api_key, verbose)
        tracing.set_attrs(
            root,
            **{"output.value": result["answer"], "f1.session": result["session"],
               "f1.model": model, "f1.n_chunks": n_chunks,
               "f1.sources": result["sources"], "f1.deltas": result["deltas"], "f1.metrics": result["metrics"]},
        )
        return result


def _answer_with_sources(question, session_key, n_chunks, model, api_key, verbose) -> dict:
    if session_key is None:
        session_key = detect_session(question)

    collection  = _get_collection(session_key)
    session_ctx = _get_session_context(session_key)

    t_start = time.perf_counter()
    wanted = resolve_context_drivers(question, question_drivers(question, session_ctx), session_ctx)
    available = extra_context(question, session_key, session_ctx)
    pair_all = top_pair_sections(session_key, wanted, pin=pinned_sections(question, session_key))
    n_extras, n_pair, _ = split_budget(n_chunks, len(available), bool(pair_all))
    extras, pair_top = available[:n_extras], pair_all[:n_pair]
    with tracing.span("f1.retrieve", "RETRIEVER", input_value=question) as sp:
        retrieved = smart_retrieve(
            collection, question, get_embed_fn(),
            n_results=n_chunks - len(extras) - len(pair_top),
            session_ctx=session_ctx,
            skip_pair_chunks=bool(pair_top),
        )
        chunks = extras + pair_top + retrieved
        for rank, chunk in enumerate(chunks, 1):
            chunk["rank"] = rank
        tracing.record_documents(sp, chunks)
        tracing.set_attrs(sp, **{"f1.session": session_key, "f1.pair_sections_generated": len(pair_top),
                                 "f1.session_views": [c["metadata"]["chunk_type"] for c in extras]})

    if verbose:
        print(f"\n[session: {session_key} | {len(chunks)} chunks retrieved]")
        for c in chunks:
            meta = c["metadata"]
            eid  = meta.get("event_id", "")
            drv  = meta.get("driver") or f"{meta.get('driver_a','')}+{meta.get('driver_b','')}"
            print(f"  {c['rank']}. [{meta.get('chunk_type')}] {drv} {eid} dist={round(c['distance'],4)}")
        print()

    t_retrieved = time.perf_counter()
    deltas = section_deltas(collection, chunks, wanted=wanted, session_key=session_key)
    headline = lap_comparison_chunk(deltas, session_ctx)
    t_prepared = time.perf_counter()
    prompt_chunks = [headline] + chunks if headline else chunks
    prompt   = format_prompt(question, prompt_chunks, max_chunks=len(prompt_chunks) if headline else n_chunks, session_ctx=session_ctx)
    _LLM_USAGE.set(None)
    t_llm = time.perf_counter()
    response = call_llm(prompt, model=model, api_key=api_key)
    t_answered = time.perf_counter()
    usage = _LLM_USAGE.get() or {}
    t_done = time.perf_counter()
    metrics = {
        "retrieve_s": round(t_retrieved - t_start, 3),
        "deltas_s": round(t_prepared - t_retrieved, 3),
        "llm_s": round(t_answered - t_llm, 3),
        "total_s": round(t_done - t_start, 3),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "prompt_chars": len(prompt),
        "model": model,
    }

    return {
        "answer":  response,
        "session": session_key,
        "sources": [
            {
                "chunk_type": c["metadata"].get("chunk_type"),
                "section":    c["metadata"].get("event_id"),
                "drivers":    [
                    c["metadata"][k] for k in ("driver", "driver_a", "driver_b")
                    if c["metadata"].get(k)
                ],
                "text":       c["text"],
            }
            for c in prompt_chunks
        ],
        "deltas":  deltas,
        "metrics": metrics,
    }


def answer(
    question: str,
    session_key: str | None = None,
    n_chunks: int = 6,
    model: str = "claude-sonnet-4-6",
    verbose: bool = False,
) -> str:
    """Answer-text-only form of answer_with_sources(), used by the CLI."""
    try:
        return answer_with_sources(
            question, session_key, n_chunks, model, verbose=verbose
        )["answer"]
    except SessionError as exc:
        return str(exc)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    """Minimal .env loader — sets env vars from a .env file if present."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Ask questions about F1 qualifying using RAG + Claude."
    )
    parser.add_argument("question", nargs="?", help="Question to answer (omit for interactive mode)")
    parser.add_argument("--session", choices=list(SESSIONS), default=None,
                        help="Force a specific session (default: auto-detect from question)")
    parser.add_argument("--chunks", type=int, default=6,
                        help="Number of chunks to retrieve (default: 6)")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Anthropic model ID (default: claude-sonnet-4-6)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print retrieved chunks before the answer")
    args = parser.parse_args()

    if args.question:
        print(answer(args.question, args.session, args.chunks, args.model, args.verbose))
        return

    # Interactive mode
    print("F1 RAG — ask questions about Melbourne 2026, Barcelona 2026, or Monza 2024 qualifying.")
    print("Type 'exit' or Ctrl+C to quit.\n")
    while True:
        try:
            q = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        print()
        print(answer(q, args.session, args.chunks, args.model, args.verbose))
        print()


if __name__ == "__main__":
    main()
