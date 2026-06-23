"""
run_faithfulness_audit.py
-------------------------
Shows WHERE faithfulness breaks down by asking Claude to reason through
each answer claim-by-claim against the retrieved context.

Uses cached answers from tests/ragas_cache.json — run run_ragas.py first.
Makes one Claude Haiku call per question (18 total).

Usage:
    python tests/run_faithfulness_audit.py           # all questions
    python tests/run_faithfulness_audit.py --fails   # only questions with unsupported claims
"""

import sys, os, json, argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / "tests" / "ragas_cache.json"

def _load_dotenv():
    env = ROOT / ".env"
    if not env.exists(): return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ: os.environ[k] = v

_load_dotenv()

EVAL_PROMPT = """\
You are evaluating whether an AI answer is faithful to its retrieved context.

QUESTION:
{question}

RETRIEVED CONTEXT (what the AI was allowed to use):
{context}

AI ANSWER:
{answer}

Task: List every specific factual claim made in the answer. For each claim, state:
- SUPPORTED if the claim can be directly verified from the retrieved context above
- UNSUPPORTED if the claim asserts something not present in the context (even if it might be true from general knowledge)

Format each claim exactly like this:
CLAIM: <the specific claim>
VERDICT: SUPPORTED | UNSUPPORTED
REASON: <one sentence explaining why, quoting the context if supported>

Be strict. If the context says "P3" and the answer says "third place", that is SUPPORTED.
If the context gives a lap time in seconds and the answer converts it to minutes:seconds correctly, that is SUPPORTED.
If the answer mentions a fact (driver nationality, team history, race outcome) not present in the context, that is UNSUPPORTED.
"""

def audit_answer(client, record: dict) -> dict:
    question = record["question"]
    answer   = record["answer"]
    contexts = record["contexts"]
    context_text = "\n\n---\n\n".join(contexts)

    prompt = EVAL_PROMPT.format(
        question=question,
        context=context_text,
        answer=answer,
    )

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    reasoning = msg.content[0].text

    # Parse verdicts
    lines = reasoning.split("\n")
    claims = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("CLAIM:"):
            claim = line[6:].strip()
            verdict = "UNKNOWN"
            reason  = ""
            if i+1 < len(lines) and lines[i+1].strip().startswith("VERDICT:"):
                verdict = lines[i+1].strip()[8:].strip()
            if i+2 < len(lines) and lines[i+2].strip().startswith("REASON:"):
                reason = lines[i+2].strip()[7:].strip()
            claims.append({"claim": claim, "verdict": verdict, "reason": reason})
            i += 3
        else:
            i += 1

    supported   = sum(1 for c in claims if "SUPPORTED" in c["verdict"] and "UNSUPPORTED" not in c["verdict"])
    unsupported = sum(1 for c in claims if "UNSUPPORTED" in c["verdict"])
    total       = len(claims)
    score       = supported / total if total > 0 else 1.0

    return {
        "question":   question,
        "score":      score,
        "supported":  supported,
        "unsupported": unsupported,
        "total":      total,
        "claims":     claims,
        "raw":        reasoning,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fails", action="store_true",
                        help="Only show questions with at least one unsupported claim")
    args = parser.parse_args()

    if not CACHE_PATH.exists():
        print("No cached answers found. Run: python tests/run_ragas.py first.")
        sys.exit(1)

    records = json.loads(CACHE_PATH.read_text())

    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set"); sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    results   = []
    all_scores = []

    for i, record in enumerate(records, 1):
        print(f"[{i:>2}/{len(records)}] auditing: {record['question'][:60]}")
        result = audit_answer(client, record)
        results.append(result)
        all_scores.append(result["score"])

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FAITHFULNESS AUDIT REPORT")
    print("=" * 80)

    for r in results:
        if args.fails and r["unsupported"] == 0:
            continue

        status = "✓" if r["unsupported"] == 0 else "✗"
        print(f"\n{status} [{r['score']:.2f}] {r['question']}")
        print(f"  {r['supported']}/{r['total']} claims supported")

        for c in r["claims"]:
            if "UNSUPPORTED" in c["verdict"]:
                print(f"  ✗ UNSUPPORTED: {c['claim']}")
                print(f"    → {c['reason']}")
            elif args.fails is False:
                print(f"  ✓ {c['claim'][:80]}")

    mean_score = sum(all_scores) / len(all_scores)
    total_claims     = sum(r["total"]      for r in results)
    total_unsupported = sum(r["unsupported"] for r in results)

    print("\n" + "=" * 80)
    print(f"Mean faithfulness : {mean_score:.3f}")
    print(f"Total claims      : {total_claims}")
    print(f"Unsupported       : {total_unsupported} ({total_unsupported/total_claims*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
