"""Helpers for turning a user question into a Chroma retrieval prompt.

These utilities are intentionally lightweight so the notebook can reuse them
without depending on any one embedding provider.

Recommended embed_fn for local use (no API key, works anywhere):

    from rag_retrieval import make_embed_fn
    embed_fn = make_embed_fn()          # downloads all-MiniLM-L6-v2 on first call
    chunks = retrieve_chunks(collection, question, embed_fn)

Uses fastembed (ONNX Runtime) rather than sentence-transformers (PyTorch) —
same model, no 2 GB torch dependency, installs in seconds on any platform.
Swap in any other provider by passing a different embed_fn — the retrieval
and prompt-formatting code is provider-agnostic.
"""

from __future__ import annotations

import re
from typing import Callable, Any


# Maps common name forms → 3-letter code used in chunk text/metadata.
# Sorted by length at call time so "verstappen" matches before "max".
_DRIVER_NAMES: dict[str, str] = {
    "norris": "NOR", "lando": "NOR",
    "piastri": "PIA", "oscar": "PIA",
    "hamilton": "HAM", "lewis": "HAM",
    "russell": "RUS", "george": "RUS",
    "leclerc": "LEC", "charles": "LEC",
    "sainz": "SAI", "carlos": "SAI",
    "verstappen": "VER",
    "perez": "PER", "sergio": "PER", "checo": "PER",
    "alonso": "ALO", "fernando": "ALO",
    "stroll": "STR", "lance": "STR",
    "bottas": "BOT", "valtteri": "BOT",
    "zhou": "ZHO", "guanyu": "ZHO",
    "hulkenberg": "HUL", "hülkenberg": "HUL",
    "magnussen": "MAG", "kevin": "MAG",
    "gasly": "GAS", "pierre": "GAS",
    "ocon": "OCO", "esteban": "OCO",
    "albon": "ALB", "alex": "ALB",
    "colapinto": "COL", "franco": "COL",
    "tsunoda": "TSU", "yuki": "TSU",
    "ricciardo": "RIC", "daniel": "RIC",
    "antonelli": "ANT", "kimi": "ANT",
    "lindblad": "LIN", "arvid": "LIN",
    "bearman": "BEA", "ollie": "BEA",
    "hadjar": "HAD", "isack": "HAD",
    "lawson": "LAW", "liam": "LAW",
    "doohan": "DOO", "jack": "DOO",
    "max": "VER",
}


def normalize_query(question: str) -> str:
    """
    Expand driver full names to include their 3-letter chunk codes.

    Example:
        "Why was Norris faster than Piastri at Monza?"
        → "Why was Norris NOR faster than Piastri PIA at Monza?"

    This bridges the semantic gap between user queries (full names) and
    chunk text (3-letter abbreviations stored by the chunkers). Applied
    automatically inside retrieve_chunks; pass normalize=False to skip.
    """
    result = question
    for name, code in sorted(_DRIVER_NAMES.items(), key=lambda x: -len(x[0])):
        if not re.search(rf'\b{re.escape(name)}\b', result, re.IGNORECASE):
            continue
        if re.search(rf'\b{code}\b', result):
            continue  # code already present, skip
        result = re.sub(
            rf'\b{re.escape(name)}\b',
            lambda m: f"{m.group()} {code}",
            result,
            count=1,
            flags=re.IGNORECASE,
        )
    return result


def extract_driver_codes(question: str) -> set[str]:
    """Return the set of 3-letter driver codes mentioned in the question."""
    codes = set()
    for name, code in _DRIVER_NAMES.items():
        if re.search(rf'\b{re.escape(name)}\b', question, re.IGNORECASE):
            codes.add(code)
    return codes


def build_driver_where(codes: set[str]) -> dict | None:
    """
    Build a Chroma where-filter for the detected driver codes.

    Single driver → any chunk mentioning that driver.

    Two drivers (A, B) → chunks where:
      - driver_a=A AND driver_b=B  (direct pair, either order)
      - OR driver_a=B AND driver_b=A
      - OR driver=A  (individual lap summary / sector / telemetry for A)
      - OR driver=B  (individual lap summary / sector / telemetry for B)

    This prevents "Norris vs Piastri" from surfacing PIA↔RUS chunks just
    because PIA is in the filter — cross-pair h2h chunks are excluded.
    """
    if not codes:
        return None

    codes = list(codes)

    if len(codes) == 1:
        c = codes[0]
        return {"$or": [{"driver_a": c}, {"driver_b": c}, {"driver": c}]}

    # Two or more drivers: restrict h2h chunks to the exact pair(s), but
    # allow individual-driver chunks for any mentioned driver.
    conditions = []
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            conditions.append({"$and": [{"driver_a": a}, {"driver_b": b}]})
            conditions.append({"$and": [{"driver_a": b}, {"driver_b": a}]})
    for c in codes:
        conditions.append({"driver": c})

    return {"$or": conditions}


_POLE_KW    = {"pole", "fastest lap", "quickest", "p1 ", "first place"}
_Q3_KW      = {"q3", "q2", "make it", "miss", "cut", "through", "eliminated",
               "qualify for", "missed out", "make q3"}
_SLOW_KW    = {"slow", "off the pace", "off ", "struggle", "behind", "gap",
               "went wrong", "pace", "lost", "weak", "worst"}


def detect_query_intent(question: str) -> str:
    """
    Classify a question into one of four intent buckets used for context
    expansion. Returns: "pole" | "q3_cutoff" | "slowness" | "general"
    """
    q = question.lower()
    if any(kw in q for kw in _POLE_KW):
        return "pole"
    if any(kw in q for kw in _Q3_KW):
        return "q3_cutoff"
    if any(kw in q for kw in _SLOW_KW):
        return "slowness"
    return "general"


def resolve_context_drivers(
    question: str,
    named_codes: set[str],
    session_ctx: dict,
    teammate_gap_threshold: float = 0.3,
) -> set[str]:
    """
    Expand named_codes with positionally-inferred comparison partners.

    session_ctx shape (from load_session_context):
        drivers:     {code: {position, lap_time_s, team, gap_to_pole_s}}
        teammate_of: {code: code}
        pole, p2, q3_last, q3_miss: driver codes

    Rules by intent:
      - 2+ named drivers → no expansion needed, return as-is
      - pole intent      → add pole sitter (or P2 if driver IS pole)
      - q3_cutoff intent → add P10 (q3_last) and P11 (q3_miss)
      - slowness intent  → add teammate if driver is slower by > threshold, else pole
      - 0 named + q3    → return {q3_last, q3_miss} (boundary question)
    """
    if len(named_codes) >= 2:
        return named_codes

    intent       = detect_query_intent(question)
    drivers_info = session_ctx.get("drivers", {})
    pole         = session_ctx.get("pole")
    p2           = session_ctx.get("p2")
    q3_last      = session_ctx.get("q3_last")
    q3_miss      = session_ctx.get("q3_miss")

    if not named_codes:
        if intent == "q3_cutoff":
            return {c for c in (q3_last, q3_miss) if c}
        return named_codes

    driver   = next(iter(named_codes))
    info     = drivers_info.get(driver, {})
    pos      = info.get("position", 99)
    teammate = session_ctx.get("teammate_of", {}).get(driver)

    if intent == "pole":
        cmp = p2 if driver == pole else pole
        return {driver, cmp} if cmp else {driver}

    if intent == "q3_cutoff":
        result = {driver}
        if q3_last:
            result.add(q3_last)
        if q3_miss:
            result.add(q3_miss)
        return result

    if intent == "slowness" and teammate:
        tm_info = drivers_info.get(teammate, {})
        tm_pos  = tm_info.get("position", 99)
        gap     = abs(info.get("lap_time_s", 999) - tm_info.get("lap_time_s", 999))
        if pos > tm_pos and gap > teammate_gap_threshold:
            return {driver, teammate}

    return {driver, pole} if pole else {driver}


def _event_significance(chunk: dict) -> float:
    """
    For head_to_head_event chunks, return a significance score in seconds.

    Prefers time_delta_s (actual lap-time contribution of the section gap)
    when available — added to chunks from the updated chunker. Falls back
    to a speed-delta proxy for older chunks without the field.

    Non-h2h-event chunks return 0 (no adjustment).
    """
    meta = chunk.get("metadata", {})
    if meta.get("chunk_type") != "head_to_head_event":
        return 0.0

    time_delta = meta.get("time_delta_s")
    if time_delta is not None:
        return abs(time_delta)  # already in seconds

    # Speed-delta fallback: scale km/h to approximate seconds.
    # Rough heuristic: 20 km/h speed difference ≈ 0.1s at a typical corner.
    entry = abs(meta.get("entry_speed_delta") or 0)
    exit_ = abs(meta.get("exit_speed_delta") or 0)
    return (entry + exit_) * 0.005  # ~0.005s per km/h delta


def rerank_by_significance(
    chunks: list[dict],
    delta_weight: float = 3.0,
) -> list[dict]:
    """
    Re-sort chunks so that high-delta head_to_head_event chunks rise above
    semantically-similar but low-delta ones.

    adjusted_score = cosine_distance − delta_weight × significance_s

    delta_weight=3.0 means a 0.1s section gap causes a 0.3 downward
    adjustment in cosine distance — strongly promoting high time-delta
    sections regardless of their cosine rank. Designed to work with
    over-fetching (3× n_results from ChromaDB) so high-delta sections
    that fall outside the raw cosine top-N can still surface.
    """
    def adjusted(chunk: dict) -> float:
        return chunk["distance"] - delta_weight * _event_significance(chunk)

    return sorted(chunks, key=adjusted)


def _chunks_driver_set(chunks: list[dict]) -> set[str]:
    """Return all driver codes referenced in chunk metadata."""
    found: set[str] = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        for key in ("driver", "driver_a", "driver_b"):
            val = meta.get(key)
            if val:
                found.add(val)
    return found


def smart_retrieve(
    collection: Any,
    question: str,
    embed_fn: Callable[[str], list[float]],
    n_results: int = 5,
    session_ctx: dict | None = None,
    normalize: bool = True,
) -> list[dict]:
    """
    retrieve_chunks with context-aware driver filter expansion and guaranteed
    representation for inferred comparison drivers.

    When session_ctx is provided, single-driver and no-driver queries are
    automatically expanded based on intent:

      "How did Norris take pole?"     → filter: NOR + PIA (P2)
      "Why was Bottas so slow?"       → filter: BOT + teammate (or pole)
      "Who just missed Q3?"           → filter: q3_last + q3_miss
      "Did Hamilton make Q3?"         → filter: HAM + q3_last + q3_miss
      "Hamilton vs Leclerc pace?"     → filter: HAM + LEC (2 named, no expansion)

    After primary retrieval, any inferred driver missing from the top-N results
    gets one guaranteed chunk injected so the comparison context is always present.

    Without session_ctx, falls back to auto_filter behaviour of retrieve_chunks.
    """
    named_codes = extract_driver_codes(question)
    if session_ctx is not None:
        all_codes = resolve_context_drivers(question, named_codes, session_ctx)
    else:
        all_codes = named_codes

    where = build_driver_where(all_codes) if all_codes else None

    # When comparing two drivers, prioritise head_to_head_event chunks so all
    # section comparisons for that pair enter the candidate pool before the
    # general fetch consumes slots with per-driver telemetry chunks.
    if len(all_codes) >= 2:
        codes_list = sorted(all_codes)
        pair_conditions = []
        for i, a in enumerate(codes_list):
            for b in codes_list[i + 1:]:
                pair_conditions.append({"$and": [{"driver_a": a}, {"driver_b": b}]})
                pair_conditions.append({"$and": [{"driver_a": b}, {"driver_b": a}]})
        pair_filter = pair_conditions[0] if len(pair_conditions) == 1 else {"$or": pair_conditions}
        h2h_where = {"$and": [{"chunk_type": "head_to_head_event"}, pair_filter]}

        chunks   = retrieve_chunks(collection, question, embed_fn,
                                   n_results=n_results * 3, where=h2h_where,
                                   normalize=normalize, auto_filter=False)
        seen_ids = {c["chunk_id"] for c in chunks}

        # Fill remaining slots with other chunk types (lap summaries, sectors, etc.)
        for c in retrieve_chunks(collection, question, embed_fn,
                                 n_results=n_results, where=where,
                                 normalize=normalize, auto_filter=False):
            if c["chunk_id"] not in seen_ids:
                chunks.append(c)
                seen_ids.add(c["chunk_id"])
    else:
        chunks   = retrieve_chunks(collection, question, embed_fn,
                                   n_results=n_results * 3, where=where,
                                   normalize=normalize, auto_filter=False)
        seen_ids = {c["chunk_id"] for c in chunks}

    # Guarantee at least one chunk for each context-inferred (non-named) driver.
    inferred = all_codes - named_codes
    if inferred:
        found = _chunks_driver_set(chunks)
        for driver in inferred - found:
            driver_where = {"$or": [
                {"driver_a": driver}, {"driver_b": driver}, {"driver": driver},
            ]}
            for chunk in retrieve_chunks(collection, question, embed_fn,
                                         n_results=3, where=driver_where,
                                         normalize=normalize, auto_filter=False):
                if chunk["chunk_id"] not in seen_ids:
                    chunks.append(chunk)
                    seen_ids.add(chunk["chunk_id"])
                    break

    return rerank_by_significance(chunks)[:n_results]


def retrieve_chunks(
    collection: Any,
    question: str,
    embed_fn: Callable[[str], list[float]],
    n_results: int = 5,
    where: dict | None = None,
    normalize: bool = True,
    auto_filter: bool = True,
) -> list[dict]:
    """Embed a question and fetch the best matching chunks from Chroma.

    normalize=True  — expands driver names to 3-letter codes before embedding.
    auto_filter=True — when no explicit where is given, restricts search to
                       chunks that mention any driver named in the question,
                       preventing unrelated driver pairs from dominating results.
    """
    query = normalize_query(question) if normalize else question
    if where is None and auto_filter:
        codes = extract_driver_codes(question)
        where = build_driver_where(codes)
    result = collection.query(
        query_embeddings=[embed_fn(query)],
        n_results=n_results,
        where=where,
    )

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    chunks = []
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        chunks.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "text": document,
                "metadata": metadata or {},
                "distance": float(distance),
            }
        )

    return chunks


def format_prompt(question: str, retrieved_chunks: list[dict], max_chunks: int = 5) -> str:
    """Format retrieved chunks into a prompt the LLM can answer from."""
    context_blocks = []

    for chunk in retrieved_chunks[:max_chunks]:
        metadata = chunk.get("metadata", {})
        header_bits = [
            metadata.get("chunk_type"),
            metadata.get("driver"),
            metadata.get("event_id"),
            metadata.get("sector"),
            metadata.get("source"),
        ]
        header = " | ".join(str(bit) for bit in header_bits if bit not in (None, ""))
        context_blocks.append(f"[{chunk['rank']}] {header}\n{chunk['text'].strip()}")

    context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."

    return (
        "You are an F1 qualifying analyst. Answer the question using only the retrieved context. "
        "If the context does not fully support the answer, say what is missing.\n\n"
        f"Question: {question}\n\n"
        "Retrieved context:\n"
        f"{context}\n\n"
        "Answer:"
    )


def build_question_prompt(
    collection: Any,
    question: str,
    embed_fn: Callable[[str], list[float]],
    n_results: int = 5,
    where: dict | None = None,
    max_chunks_in_prompt: int = 5,
    normalize: bool = True,
) -> dict:
    """Convenience wrapper that returns both retrieved chunks and a prompt."""
    retrieved_chunks = retrieve_chunks(
        collection=collection,
        question=question,
        embed_fn=embed_fn,
        n_results=n_results,
        where=where,
        normalize=normalize,
    )
    prompt = format_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
        max_chunks=max_chunks_in_prompt,
    )
    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
        "where": where,
        "n_results": n_results,
    }


def make_embed_fn(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Return an embed_fn compatible with retrieve_chunks() and build_question_prompt().

    Uses fastembed (ONNX Runtime) with all-MiniLM-L6-v2 by default:
      - 384-dim embeddings, ~40 MB download on first run
      - No API key required — works offline after the initial download
      - No PyTorch dependency — ONNX Runtime only (~50 MB vs ~2 GB for torch)
      - Semantically meaningful: synonyms and paraphrases match correctly

    To swap in a different provider, just return a different callable that
    accepts a string and returns a list[float].

    Args:
        model_name: any fastembed-supported model name

    Returns:
        embed_fn(text: str) -> list[float]
    """
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name)

    def embed_fn(text: str) -> list[float]:
        return next(model.embed([text])).tolist()

    return embed_fn
