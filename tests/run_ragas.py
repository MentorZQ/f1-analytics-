"""
run_ragas.py
------------
Measures RAGAS faithfulness on the eval set.

Faithfulness = fraction of claims in the LLM answer that are supported
by the retrieved context. 1.0 = no hallucination. Evaluated by a second
Claude call that decomposes the answer into statements and verifies each.

Answers + contexts are cached to tests/ragas_cache.json so generation
only runs once. Pass --regen to force a fresh run.

Usage:
    python tests/run_ragas.py            # use cache if available
    python tests/run_ragas.py --regen    # regenerate all answers
"""

import sys
import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "RAG_data_layers/F1 Project Thoughts"))
sys.path.insert(0, str(ROOT / "tests"))

CACHE_PATH = ROOT / "tests" / "ragas_cache.json"

# ── Load env + imports ────────────────────────────────────────────────────────

def _load_dotenv():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

_load_dotenv()

# Stub out the removed vertexai module that RAGAS 0.4.3 still imports at load time
import types as _types
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = _types.ModuleType("langchain_community.chat_models.vertexai")
    class _ChatVertexAI: pass
    _stub.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

import chromadb
from rag_retrieval import smart_retrieve, make_embed_fn, format_prompt
from eval_set import EVAL_SET

COLLECTIONS = {
    "barcelona": "f1_qualifying_2026_barcelona",
    "melbourne": "f1_qualifying_2026_melbourne",
}

client   = chromadb.PersistentClient(path=str(ROOT / "RAG_data_layers/chroma_store"))
embed_fn = make_embed_fn()

def get_session_ctx(col):
    r = col.get(where={"chunk_type": "driver_lap_summary"}, include=["metadatas"])
    drivers, teams = {}, {}
    for m in r["metadatas"]:
        if "driver" not in m:
            continue
        d = m["driver"]
        drivers[d] = {"qualifying_position": m.get("qualifying_position"),
                      "lap_time_s": m.get("lap_time_seconds"), "team": m.get("team", "")}
        t = m.get("team", "")
        if t:
            teams.setdefault(t, []).append(d)
    standings = sorted(drivers.items(), key=lambda x: x[1]["qualifying_position"] or 99)
    teammate_of = {}
    for drvs in teams.values():
        for d in drvs:
            mates = [x for x in drvs if x != d]
            if mates:
                teammate_of[d] = mates[0]
    return {"drivers": drivers, "standings": [d for d, _ in standings],
            "teammate_of": teammate_of,
            "pole": standings[0][0] if standings else None,
            "p2":   standings[1][0] if len(standings) > 1 else None}

cols = {k: client.get_collection(v) for k, v in COLLECTIONS.items()}
ctxs = {k: get_session_ctx(v) for k, v in cols.items()}

# ── Answer generation ─────────────────────────────────────────────────────────

def generate_answers(n_chunks: int = 6) -> list[dict]:
    """Run all eval questions through the RAG pipeline. Returns list of records."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    ac = anthropic.Anthropic(api_key=api_key)
    system = (
        "You are an expert F1 qualifying analyst. "
        "Answer questions using ONLY the retrieved telemetry and session data provided. "
        "Be precise with numbers — quote lap times, speeds, and deltas directly from the context. "
        "If the context does not contain enough information to answer fully, say so explicitly "
        "rather than guessing."
    )

    records = []
    for i, item in enumerate(EVAL_SET, 1):
        q   = item["question"]
        col = cols[item["session"]]
        ctx = ctxs[item["session"]]

        chunks = smart_retrieve(col, q, embed_fn, n_results=n_chunks, session_ctx=ctx)
        prompt = format_prompt(q, chunks, max_chunks=n_chunks, session_ctx=ctx)

        msg = ac.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                                 system=system, messages=[{"role": "user", "content": prompt}])
        answer = msg.content[0].text
        contexts = [c["text"] for c in chunks]

        records.append({"question": q, "answer": answer, "contexts": contexts})
        print(f"  [{i:>2}/{len(EVAL_SET)}] generated: {q[:60]}")

    return records

# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def run_ragas(records: list[dict]) -> dict:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import Faithfulness
    from ragas.llms import LangchainLLMWrapper
    from langchain_anthropic import ChatAnthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    llm = LangchainLLMWrapper(ChatAnthropic(
        model="claude-haiku-4-5-20251001",  # cheapest Claude for eval calls
        api_key=api_key,
    ))

    faithfulness = Faithfulness(llm=llm)

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
        )
        for r in records
    ]
    dataset = EvaluationDataset(samples=samples)
    result  = evaluate(dataset, metrics=[faithfulness])
    return result

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen", action="store_true",
                        help="Regenerate answers even if cache exists")
    args = parser.parse_args()

    if CACHE_PATH.exists() and not args.regen:
        print(f"Loading cached answers from {CACHE_PATH.name} (use --regen to refresh)")
        records = json.loads(CACHE_PATH.read_text())
    else:
        print(f"Generating answers for {len(EVAL_SET)} questions...")
        records = generate_answers()
        CACHE_PATH.write_text(json.dumps(records, indent=2))
        print(f"Cached to {CACHE_PATH.name}")

    print(f"\nRunning RAGAS faithfulness on {len(records)} answers...")
    print("(uses claude-haiku for eval calls to conserve cost)\n")
    result = run_ragas(records)
    print(result)

if __name__ == "__main__":
    main()
