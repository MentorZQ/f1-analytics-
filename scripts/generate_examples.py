"""
generate_examples.py
--------------------
Fills in web/examples.json with a saved answer for each example question, so the
site can serve them instantly without an API call. Re-run after changing the
questions, the prompt or the data.

Each run makes one paid Anthropic API call per example.

Run from the project root (needs ANTHROPIC_API_KEY in the environment or .env):
    python scripts/generate_examples.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import query as query_module  # noqa: E402

EXAMPLES_PATH = ROOT / "web/examples.json"
STORE_PATH = ROOT / "RAG_data_layers/chroma_store"


def main() -> None:
    query_module._load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to .env or export it.")

    examples = json.loads(EXAMPLES_PATH.read_text())

    with tempfile.TemporaryDirectory() as tmp:
        # Newer chromadb versions rewrite the store on open. Reading a copy keeps
        # the committed store files unchanged.
        os.environ["CHROMA_PATH"] = shutil.copytree(STORE_PATH, Path(tmp) / "store").as_posix()

        generated = []
        for example in examples:
            print(f"[{example['session']}] {example['question']}")
            result = query_module.answer_with_sources(
                example["question"],
                session_key=example["session"],
                model=os.environ.get("F1_MODEL", "claude-sonnet-4-6"),
            )
            if not result["answer"].strip():
                sys.exit("Got an empty answer. web/examples.json was not changed.")
            generated.append({"question": example["question"], **result})

    # Written only after every call succeeds, so a failure can't leave a partial file.
    EXAMPLES_PATH.write_text(json.dumps(generated, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {len(generated)} answers to {EXAMPLES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
