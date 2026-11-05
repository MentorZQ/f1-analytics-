"""
Ask a model that has seen none of the checks, answers or code to write a batch of visitor questions.

It is given only what a visitor could know: what the site is for, which sessions it covers, who drove for which team, and
what kinds of data exist. It is not given the evaluation code, the checks, earlier questions or any answers, and it does not
see the data values, so it cannot write questions to fit what the system does well. The output is committed BEFORE the first run.

    python scripts/write_fresh_questions.py            # writes evals/questions_heldout_fresh.json
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RAG_data_layers" / "F1 Project Thoughts"))

import anthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from events_and_chunking import session_snapshot as snap  # noqa: E402
from rag_retrieval import DRIVER_SURNAMES  # noqa: E402

load_dotenv(ROOT / ".env")
MODEL = "claude-opus-5-5"
OUT = ROOT / "evals" / "questions_heldout_fresh.json"
SESSIONS = {"monza": ("f1_qualifying_2024_monza", "the 2024 Italian Grand Prix at Monza"),
            "barcelona": ("f1_qualifying_2026_barcelona", "the 2026 Spanish Grand Prix at Barcelona-Catalunya"),
            "melbourne": ("f1_qualifying_2026_melbourne", "the 2026 Australian Grand Prix at Albert Park, Melbourne")}


def rosters() -> str:
    parts = []
    for key, (name, label) in SESSIONS.items():
        s = snap.load_snapshot(ROOT / "RAG_data_layers" / "session_snapshots" / f"{name}.npz", align=False)
        rows = [f"{r['Driver']} ({DRIVER_SURNAMES.get(r['Driver'], r['Driver']).title()}, {r['Team']})" for _, r in s.fastest.iterrows()]
        parts.append(f"{label} [{key}]: " + "; ".join(sorted(rows)))
    return "\n".join(parts)


PROMPT = f"""You are helping test a public website that answers questions about Formula 1 qualifying sessions. Visitors type a question
in plain language and get an answer written from the site's data. Write {{n}} realistic questions that different visitors might type.

WHAT THE SITE COVERS
Three qualifying sessions only:
{rosters()}

The data for each session: every driver's best lap and their best time in each of Q1, Q2 and Q3, final classification and how far
each driver got; sector times; speed-trap readings; for each driver's lap, telemetry split into corner and straight sections
(entry, apex and exit speeds, braking points, throttle, top speed, gear); head-to-head comparisons between any two drivers
section by section; the fresh or used tyre set on the lap; weather; and race-control messages (flags, deleted laps, penalties
and incidents during the session).

WHAT IT DOES NOT HAVE
Race results, race laps, pit stops, tyre strategy over a race, championship standings, sprint sessions, practice sessions, other
circuits or seasons, driver contracts, car specifications, or anything after qualifying ended.

HOW TO WRITE THEM
- Write in the voice of real visitors: casual, sometimes abbreviated or misspelled, first names or nicknames or surnames or
  three-letter codes, circuit nicknames ("Barca", "Albert Park", "Aussie GP"), sometimes without naming the session at all.
- Mix the kinds. Roughly: 14 simple lookups (a time, a position, who was fastest at something); 12 comparisons of two drivers or
  two teammates; 8 "why" or "how" questions about a driver's performance; 6 questions about a particular corner or straight;
  5 about teams or the whole grid; 5 about the session itself (weather, flags, incidents, who made Q3); 5 questions the data
  cannot answer (race, strategy, championship, other seasons, predictions); 3 with a false or unsupported premise that the
  site's data should not go along with (for example asking why something happened that you have good reason to think did not);
  2 that are vague or ambiguous.
- Do not write questions with instructions aimed at the AI. Do not include any answers.
- You have not seen the data values, so never state a lap time, gap or position as a fact inside a question.

OUTPUT
Return only a JSON array. Each element: {{"question": str, "session": one of "monza", "barcelona", "melbourne" or null (the session
the visitor means, null if they do not say and it is not clear), "kind": a short label, "drivers": [three-letter codes the question
is about, from the lists above, empty if none], "answerable": true if the data described above could answer it, false if not,
"note": one sentence on what a correct answer must contain or why it is unanswerable}}."""


def main() -> None:
    n = 60
    client = anthropic.Anthropic()
    msg = client.messages.create(model=MODEL, max_tokens=20000, messages=[{"role": "user", "content": PROMPT.replace("{n}", str(n))}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    items = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
    out = []
    for i, it in enumerate(items, 1):
        out.append({"id": f"f{i:02d}", "question": it["question"].strip(), "session": None,
                    "expected_session": it.get("session"), "kind": "fresh_" + re.sub(r"[^a-z0-9]+", "_", str(it.get("kind", "other")).lower()).strip("_"),
                    "expected_drivers": [d for d in it.get("drivers", []) if re.fullmatch(r"[A-Z]{3}", str(d))],
                    "answerable": bool(it.get("answerable", True)), "false_premise": False, "must_include": [],
                    "author": f"written by {MODEL} (no access to checks, code or data values)", "note": it.get("note", "")})
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {len(out)} questions to {OUT}; {sum(not q['answerable'] for q in out)} marked unanswerable; prompt tokens {msg.usage.input_tokens}, output {msg.usage.output_tokens}")


if __name__ == "__main__":
    main()
