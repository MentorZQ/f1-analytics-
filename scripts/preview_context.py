"""
Show exactly what the model would be given for a question, without calling any model (no API call, no cost).

    python scripts/preview_context.py "how did the ferraris do in barca"
    python scripts/preview_context.py "what happened to leclerc in Q3 at barcelona" --session barcelona

Prints the sources (the blocks of evidence) and the full prompt. Use it to check retrieval, the grid and the notes; only the
wording of the final answer needs a real model call.
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.pop("ANTHROPIC_API_KEY", None)             # make a real call impossible, even by mistake


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--session")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:                      # Chroma can rewrite its files when it opens a store
        os.environ["CHROMA_PATH"] = shutil.copytree(ROOT / "RAG_data_layers" / "chroma_store", Path(tmp) / "store").as_posix()
        import query

        seen = {}

        def no_model(prompt, **kwargs):
            seen["prompt"], seen["kwargs"] = prompt, kwargs
            return "[no model was called]"

        query.call_llm = no_model
        result = query.answer_with_sources(args.question, session_key=args.session, model="preview")
        print(f"session: {result['session']}   sources: {len(result['sources'])}   extra call settings: {seen['kwargs'] or 'none'}\n")
        for i, s in enumerate(result["sources"], 1):
            print(f"--- source {i}: {s['chunk_type']} {s.get('section') or ''} {s['drivers']}")
            print(s["text"][:700] + ("..." if len(s["text"]) > 700 else ""))
        print("\n=== prompt sent to the model (not sent) ===")
        print(seen["prompt"][:1500] + ("..." if len(seen["prompt"]) > 1500 else ""))


if __name__ == "__main__":
    main()
