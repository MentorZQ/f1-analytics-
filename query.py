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
import os
import re
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "RAG_data_layers/F1 Project Thoughts"))

from rag_retrieval import smart_retrieve, make_embed_fn, format_prompt


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
        "q3_last":      standings[9][0] if len(standings) > 9 else None,
        "q3_miss":      standings[10][0] if len(standings) > 10 else None,
    }


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    """Send the prompt to the Anthropic API and return the response text."""
    try:
        import anthropic
    except ImportError:
        return (
            "[anthropic package not installed — run: pip install anthropic]\n\n"
            "Prompt that would have been sent:\n\n" + prompt
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
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
        "rather than guessing."
    )

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── Main answer function ──────────────────────────────────────────────────────

def answer(
    question: str,
    session_key: str | None = None,
    n_chunks: int = 6,
    model: str = "claude-sonnet-4-6",
    verbose: bool = False,
) -> str:
    """
    Full RAG pipeline: retrieve → format → LLM → return answer string.

    Args:
        question:    Natural language question about F1 qualifying.
        session_key: "barcelona" or "monza". Auto-detected from question if None.
        n_chunks:    Number of chunks to retrieve and pass to the LLM.
        model:       Anthropic model ID.
        verbose:     Print retrieved chunks before the answer.
    """
    if session_key is None:
        session_key = detect_session(question)

    collection_name = SESSIONS.get(session_key)
    if collection_name is None:
        return f"Unknown session '{session_key}'. Available: {list(SESSIONS)}"

    client = chromadb.PersistentClient(path=str(ROOT / "RAG_data_layers/chroma_store"))

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return (
            f"Collection '{collection_name}' not found in the chroma store. "
            f"Run: python reingest.py"
        )

    embed_fn    = make_embed_fn()
    session_ctx = load_session_context(collection)

    chunks = smart_retrieve(
        collection, question, embed_fn,
        n_results=n_chunks,
        session_ctx=session_ctx,
    )

    if verbose:
        print(f"\n[session: {session_key} | {len(chunks)} chunks retrieved]")
        for c in chunks:
            meta = c["metadata"]
            eid  = meta.get("event_id", "")
            drv  = meta.get("driver") or f"{meta.get('driver_a','')}+{meta.get('driver_b','')}"
            print(f"  {c['rank']}. [{meta.get('chunk_type')}] {drv} {eid} dist={round(c['distance'],4)}")
        print()

    prompt    = format_prompt(question, chunks, max_chunks=n_chunks, session_ctx=session_ctx)
    response  = call_llm(prompt, model=model)
    return response


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
