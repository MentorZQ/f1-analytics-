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
                           resolve_context_drivers, extract_driver_codes)

import tracing


# ── Session registry ──────────────────────────────────────────────────────────

SESSIONS = {
    "barcelona": "f1_qualifying_2026_barcelona",
    "melbourne": "f1_qualifying_2026_melbourne",
    "monza":     "f1_qualifying_2024_monza",
}

# Keywords used to auto-detect session from the question
_SESSION_KEYWORDS = {
    "barcelona": ["barcelona", "spain", "spanish", "catalunya", "catalan"],
    "melbourne": ["melbourne", "australia", "australian", "albert park"],
    "monza":     ["monza", "italy", "italian", "2024"],
}


def detect_session(question: str) -> str:
    """Return the most likely session key based on keywords in the question."""
    q = question.lower()
    scores = {name: sum(kw in q for kw in kws) for name, kws in _SESSION_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    # If nothing matched, default to the most recent session
    return best if scores[best] > 0 else "barcelona"


# ── Session context ───────────────────────────────────────────────────────────

def load_session_context(collection) -> dict:
    """Build session context from driver_lap_summary chunks stored in the collection."""
    results = collection.get(
        where={"chunk_type": "driver_lap_summary"},
        include=["metadatas"],
    )
    drivers = {}
    for m in results["metadatas"]:
        if "driver" not in m:
            continue
        drivers[m["driver"]] = {
            "lap_time_seconds":    m.get("lap_time_seconds"),
            "qualifying_position": m.get("qualifying_position"),
            "team":                m.get("team", ""),
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


# ── Section-by-section lap comparison ─────────────────────────────────────────

_DISTANCE_WINDOW = re.compile(r"Distance window:\s*([\d.]+)m\s*[–-]\s*([\d.]+)m")


def section_deltas(collection, chunks: list[dict], wanted: set[str] | None = None) -> dict | None:
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
    with tracing.span("f1.retrieve", "RETRIEVER", input_value=question) as sp:
        chunks = smart_retrieve(
            collection, question, get_embed_fn(),
            n_results=n_chunks,
            session_ctx=session_ctx,
        )
        tracing.record_documents(sp, chunks)
        tracing.set_attrs(sp, **{"f1.session": session_key})

    if verbose:
        print(f"\n[session: {session_key} | {len(chunks)} chunks retrieved]")
        for c in chunks:
            meta = c["metadata"]
            eid  = meta.get("event_id", "")
            drv  = meta.get("driver") or f"{meta.get('driver_a','')}+{meta.get('driver_b','')}"
            print(f"  {c['rank']}. [{meta.get('chunk_type')}] {drv} {eid} dist={round(c['distance'],4)}")
        print()

    t_retrieved = time.perf_counter()
    prompt   = format_prompt(question, chunks, max_chunks=n_chunks, session_ctx=session_ctx)
    _LLM_USAGE.set(None)
    t_llm = time.perf_counter()
    response = call_llm(prompt, model=model, api_key=api_key)
    t_answered = time.perf_counter()
    usage = _LLM_USAGE.get() or {}
    wanted = resolve_context_drivers(question, extract_driver_codes(question), session_ctx)
    deltas = section_deltas(collection, chunks, wanted=wanted)
    t_done = time.perf_counter()
    metrics = {
        "retrieve_s": round(t_retrieved - t_start, 3),
        "llm_s": round(t_answered - t_llm, 3),
        "deltas_s": round(t_done - t_answered, 3),
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
            for c in chunks
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
