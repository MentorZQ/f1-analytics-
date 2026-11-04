"""
Generate the data-driven held-out questions, with the expected facts computed from the stored data.

    python -m evals.make_heldout                     # write evals/questions_heldout_data.json
    python -m evals.make_heldout --check             # recompute from the current data and compare with the file

Nothing here asks a model: every fact (a lap time, a position, a gap, who was quicker through a corner) is read from
the store's lap summaries and the session snapshots, so the ground truth does not depend on anything being tuned.
A fixed seed makes the choice of drivers, pairs, corners and wording reproducible. The questions are written to the
file and committed BEFORE the first run; they are for measuring, not for tuning against.

Question families (per session): best lap, grid position, gap to pole, speed-trap leader, fastest sector, gap between
two drivers, who was quicker through a corner, gap between teammates, and how far a driver got in qualifying.

History: the first set (seed 20261103) asked "which driver was last into Q3?" and took P10 as the answer; the second set
(seed 20261104) read each driver's recorded stage instead. Both were wrong in a way the data hid: the P10 car at Barcelona
(Leclerc) crashed in Q3 and set no time, so the loader recorded him as out in Q2 and only 9 cars appeared to reach Q3. The
loader now counts a top-ten car as having entered Q3, and "last into Q3" is the slowest Q2 time among the ten cars that
advanced (the classification is decided by Q3 laps). Two families were added: that one, and a pair where one driver went
out in Q1 and the other reached Q3, whose like-for-like gap is the Q1 gap.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RAG_data_layers" / "F1 Project Thoughts"))

import chromadb  # noqa: E402

from evals.code_checks import SURNAMES  # noqa: E402
from events_and_chunking import session_snapshot as snap  # noqa: E402
from rag_retrieval import _DRIVER_NAMES  # noqa: E402

OUT = Path(__file__).resolve().parent / "questions_heldout_data.json"
SEED = 20261107        # v3 20261105; v4 20261106 (like-for-like corners); v5 expects the like-for-like gap for pair and teammate questions

SESSIONS = {
    "monza": ("f1_qualifying_2024_monza", ["Monza", "the Italian GP", "the Italian Grand Prix"]),
    "barcelona": ("f1_qualifying_2026_barcelona", ["Barcelona", "the Spanish GP", "the Spanish Grand Prix"]),
    "melbourne": ("f1_qualifying_2026_melbourne", ["Melbourne", "the Australian GP", "Albert Park"]),
}
# The drivers most of the development questions were about; the held-out set leans on the others.
COMMON = {"RUS", "HAM", "ANT", "NOR", "PIA", "LEC", "VER", "HAD"}
WORDS = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
         "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
         "twentieth", "twenty-first", "twenty-second"]


# ── formatting helpers ──

def lap_alternatives(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:06.3f}|{seconds:.3f}"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def position_alternatives(n: int) -> str:
    return "|".join(f"re:\\b{w}\\b" for w in (f"P{n}", ordinal(n), WORDS[n]))


def first_names(code: str) -> list[str]:
    surname = SURNAMES.get(code, "")
    return sorted({name for name, c in _DRIVER_NAMES.items() if c == code and name != surname and " " not in name
                   and name not in {"max", "jack", "alex", "hulk", "checo", "gabi", "ollie"}})


def display(code: str, rng: random.Random) -> str:
    """How a fan might write the driver: surname or first name."""
    surname = SURNAMES.get(code, code.lower()).replace("hulkenberg", "Hülkenberg").title()
    firsts = first_names(code)
    if firsts and rng.random() < 0.35:
        return rng.choice(firsts).title()
    return surname


def name_alternatives(code: str) -> str:
    """Any way the answer might name the driver: surname, first name or code, as whole words."""
    names = [SURNAMES.get(code, ""), *first_names(code), code]
    if code == "HUL":
        names.append("hülkenberg")
    return "|".join(f"re:\\b{re.escape(n)}\\b" for n in dict.fromkeys(n for n in names if n))


def _any(*patterns: str) -> str:
    """Alternatives for must_include: each is its own regex (the evaluator splits on '|', so no '|' inside a pattern)."""
    return "|".join(f"re:{p}" for p in patterns)


def _out_in(stage: str) -> str:
    q = stage.lower()
    return _any(rf"eliminated[^.]{{0,25}}\b{q}\b", rf"knocked out[^.]{{0,25}}\b{q}\b", rf"dropped out[^.]{{0,25}}\b{q}\b",
                rf"\bout in {q}\b", rf"\bend of {q}\b", rf"\b{q} elimination\b", rf"\b{q} exit\b")


STAGE_ALTERNATIVES = {
    "Q3": _any(r"qualified for q3", r"reached q3", r"advanced to q3", r"progressed to q3", r"through to q3", r"made it (?:in)?to q3",
               r"made q3", r"\bin q3\b"),
    "Q2": _out_in("Q2"),
    "Q1": _out_in("Q1"),
}


def gap_text(value: float) -> str:
    return f"{abs(value):.3f}"


# ── reading the data ──

def load_session(client, collection: str, snapshot_path: Path) -> dict:
    col = client.get_collection(collection)
    drivers = {}
    summaries = col.get(where={"chunk_type": "driver_lap_summary"}, include=["documents", "metadatas"])
    for doc, meta in zip(summaries["documents"], summaries["metadatas"]):
        trap = re.search(r"Speed trap \(speed trap\):\s*([\d.]+)\s*km/h", doc)
        progression = re.search(r"Qualifying progression:\s*([^\n]+)", doc)
        line = progression.group(1) if progression else ""
        stage = "Q3" if "Qualified for Q3" in line else (f"Q{m.group(1)}" if (m := re.search(r"Eliminated at end of Q([12])", line)) else None)
        times_line = re.search(r"Best time in each session:\s*([^\n]+)", doc)
        times = {int(n): int(mm) * 60 + float(ss) for n, mm, ss in re.findall(r"Q([123]) (\d+):(\d+\.\d+)", times_line.group(1))} if times_line else {}
        drivers[meta["driver"]] = {"time": meta["lap_time_seconds"], "position": int(meta["qualifying_position"]),
                                   "team": meta.get("team", ""), "trap": float(trap.group(1)) if trap else None, "stage": stage,
                                   "times": times}
    fastest_sector = {}
    for doc in col.get(where={"chunk_type": "driver_sector"}, include=["documents"])["documents"]:
        m = re.search(r"Fastest sector (\d) in session:\s*([\d.]+)s \((\w{3})\)", doc)
        if m:
            fastest_sector[int(m.group(1))] = (float(m.group(2)), m.group(3))
    return {"drivers": drivers, "fastest_sector": fastest_sector, "snapshot": snap.load_snapshot(snapshot_path)}


def corner_sections(snapshot) -> list[tuple[int, str]]:
    """(corner number, id of the section holding it) for every numbered corner."""
    out = []
    for event in snapshot.events:
        for n in event.get("corner_numbers") or []:
            out.append((int(n), event.get("display_id", event["id"])))
    return sorted(set(out))


def section_gap(snapshot, a: str, b: str, section_id: str, session: int | None = None):
    """
    (quicker driver, gap in seconds) through a section, from the stored head-to-head computation; None if unavailable.
    `session` is the like-for-like session when the two best laps come from different sessions (see snap.focus_session).
    """
    for chunk in snap.compare_drivers(snapshot, a, b, session=session):
        if chunk.metadata.get("event_id") == section_id and chunk.metadata.get("time_delta_s") is not None:
            delta = chunk.metadata["time_delta_s"]                       # driver_a's time minus driver_b's
            first, second = chunk.metadata["driver_a"], chunk.metadata["driver_b"]
            return (second if delta > 0 else first), abs(delta)
    return None


# ── the questions ──

def build(session_key: str, data: dict, rng: random.Random, next_id) -> list[dict]:
    _, circuit_words = SESSIONS[session_key]
    drivers = data["drivers"]
    codes = sorted(drivers)
    by_pos = sorted(codes, key=lambda c: drivers[c]["position"])
    pole = by_pos[0]
    questions = []

    def circuit() -> str:
        return rng.choice(circuit_words)

    def pick(n: int, exclude=()) -> list[str]:
        pool = [c for c in codes if c not in exclude]
        lean = [c for c in pool if c not in COMMON]
        chosen: list[str] = []
        while len(chosen) < n and pool:
            source = lean if lean and rng.random() < 0.75 else pool
            c = rng.choice(source)
            if c not in chosen:
                chosen.append(c)
        return chosen

    def add(kind, text, drivers_involved, must, note):
        questions.append({"id": next_id(), "question": text, "session": None, "expected_session": session_key, "kind": kind,
                          "expected_drivers": list(drivers_involved), "answerable": True, "false_premise": False,
                          "must_include": must, "author": "generated", "note": note})

    for c in pick(3):
        template = rng.choice(["What was {d}'s best qualifying lap at {c}?", "How fast was {d} in qualifying at {c}?",
                               "What lap time did {d} set in qualifying at {c}?"])
        add("data_lap_time", template.format(d=display(c, rng), c=circuit()), [c], [lap_alternatives(drivers[c]["time"])],
            f"{c} best lap {drivers[c]['time']:.3f}s (driver lap summary)")
    for c in pick(2):
        template = rng.choice(["Where did {d} qualify at {c}?", "What was {d}'s grid position at {c}?"])
        add("data_position", template.format(d=display(c, rng), c=circuit()), [c], [position_alternatives(drivers[c]["position"])],
            f"{c} classified P{drivers[c]['position']}")
    for c in pick(2, exclude=(pole,)):
        gap = drivers[c]["time"] - drivers[pole]["time"]
        template = rng.choice(["How far off pole was {d} at {c}?", "What was {d}'s gap to pole in qualifying at {c}?"])
        add("data_gap_to_pole", template.format(d=display(c, rng), c=circuit()), [c], [gap_text(gap)],
            f"{c} {drivers[c]['time']:.3f}s vs pole {drivers[pole]['time']:.3f}s = {gap:.3f}s")
    trapped = [(drivers[c]["trap"], c) for c in codes if drivers[c]["trap"]]
    if trapped:
        speed = max(t for t, _ in trapped)
        leaders = sorted(c for t, c in trapped if t == speed)                    # a tie names every tied driver
        add("data_speed_trap", f"Who recorded the highest speed trap reading in qualifying at {circuit()}?", leaders,
            ["|".join(name_alternatives(c) for c in leaders), f"{speed:g}"],
            f"{' and '.join(leaders)} {speed:g} km/h, highest speed-trap reading of {len(trapped)} drivers")
    if data["fastest_sector"]:
        sector = rng.choice(sorted(data["fastest_sector"]))
        t, leader = data["fastest_sector"][sector]
        add("data_fastest_sector", f"Who set the quickest sector {sector} time in qualifying at {circuit()}?", [leader],
            [name_alternatives(leader), f"{t:.3f}"], f"fastest sector {sector}: {leader} {t:.3f}s")
    pairs = []
    while len(pairs) < 3:
        a, b = pick(2)
        if a != b and (a, b) not in pairs and (b, a) not in pairs:
            pairs.append((a, b))
    def gap_between(a: str, b: str) -> tuple[float, str]:
        """a minus b: in the last session both set a time when the best laps come from different sessions, else best laps."""
        focus = snap.focus_session(drivers[a].get("times") or {}, drivers[b].get("times") or {})
        if focus:
            return drivers[a]["times"][focus] - drivers[b]["times"][focus], f"Q{focus} laps"
        return drivers[a]["time"] - drivers[b]["time"], "best laps"

    for a, b in pairs:
        gap, basis = gap_between(a, b)
        quicker = a if gap < 0 else b
        add("data_pair_gap", f"How far apart were {display(a, rng)} and {display(b, rng)} in qualifying at {circuit()}?", [a, b],
            [gap_text(gap)], f"{a} {drivers[a]['time']:.3f}s, {b} {drivers[b]['time']:.3f}s; {quicker} quicker by {abs(gap):.3f}s on the {basis}")
    corners = corner_sections(data["snapshot"])
    made = 0
    attempts = 0
    while made < 3 and attempts < 40 and corners:
        attempts += 1
        a, b = pick(2)
        number, section_id = rng.choice(corners)
        focus = snap.focus_session(drivers[a].get("times") or {}, drivers[b].get("times") or {})
        result = section_gap(data["snapshot"], a, b, section_id, focus)
        if result is None and focus:
            continue                                    # that session's laps were not stored for this pair
        if not result or result[1] < 0.020:
            continue
        winner, gap = result
        template = rng.choice(["Who was faster through turn {n} at {c}, {a} or {b}?", "{a} or {b}: who was quicker through T{n} at {c}?"])
        add("data_corner_winner", template.format(n=number, c=circuit(), a=display(a, rng), b=display(b, rng)), [a, b],
            [name_alternatives(winner), gap_text(gap)], f"through {section_id} (turn {number}): {winner} quicker by {gap:.3f}s")
        made += 1
    teams: dict[str, list[str]] = {}
    for c in codes:
        teams.setdefault(drivers[c]["team"], []).append(c)
    full_teams = sorted(t for t, cs in teams.items() if len(cs) == 2 and t)
    rng.shuffle(full_teams)
    for team in full_teams[:2]:
        a, b = teams[team]
        gap, basis = gap_between(a, b)
        add("data_teammate_gap", f"How far apart were the two {team} drivers in qualifying at {circuit()}?", [a, b], [gap_text(gap)],
            f"{a} {drivers[a]['time']:.3f}s, {b} {drivers[b]['time']:.3f}s; gap on the {basis}")
    staged = [c for c in codes if drivers[c].get("stage")]
    for c in pick(2, exclude=[x for x in codes if x not in staged]):
        stage = drivers[c]["stage"]
        template = rng.choice(["How far did {d} get in qualifying at {c}: out in Q1, out in Q2, or into Q3?",
                               "Did {d} make it into Q3 at {c}, or was {d} knocked out earlier?"])
        add("data_stage", template.format(d=display(c, rng), c=circuit()), [c], [STAGE_ALTERNATIVES[stage]],
            f"{c} progression: {stage if stage != 'Q3' else 'qualified for Q3'} (driver lap summary)")
    advanced = [c for c in codes if drivers[c].get("stage") == "Q3" and (drivers[c].get("times") or {}).get(2)]
    if len(advanced) >= 10:
        last = max(advanced, key=lambda c: drivers[c]["times"][2])
        template = rng.choice(["Who was the last driver to get through to Q3 at {c}?",
                               "Which driver only just made it into Q3 at {c}, getting through with the slowest Q2 time?"])
        add("data_last_into_q3", template.format(c=circuit()), [last], [name_alternatives(last)],
            f"{last} had the slowest Q2 time ({drivers[last]['times'][2]:.3f}s) of the {len(advanced)} cars that advanced to Q3")
    out_q1 = [c for c in codes if drivers[c].get("stage") == "Q1" and (drivers[c].get("times") or {}).get(1)]
    in_q3 = [c for c in codes if drivers[c].get("stage") == "Q3" and (drivers[c].get("times") or {}).get(1)]
    for _ in range(2):
        if not out_q1 or not in_q3:
            break
        low = rng.choice([c for c in out_q1 if c not in COMMON] or out_q1)
        high = rng.choice([c for c in in_q3 if c not in COMMON] or in_q3)
        gap = drivers[low]["times"][1] - drivers[high]["times"][1]
        template = rng.choice(["Compare {a} and {b} in qualifying at {c}.", "How did {a} stack up against {b} in qualifying at {c}?"])
        add("data_like_for_like", template.format(a=display(low, rng), b=display(high, rng), c=circuit()), [low, high],
            [gap_text(gap)], f"{low} out in Q1, {high} reached Q3: like-for-like is Q1, {low} {drivers[low]['times'][1]:.3f}s "
                             f"vs {high} {drivers[high]['times'][1]:.3f}s = {abs(gap):.3f}s")
    return questions


def generate(store: Path, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    counter = iter(range(1, 10_000))
    next_id = lambda: f"d{next(counter):02d}"                     # noqa: E731
    client = chromadb.PersistentClient(path=str(store))
    out: list[dict] = []
    for key, (collection, _) in SESSIONS.items():
        data = load_session(client, collection, ROOT / "RAG_data_layers" / "session_snapshots" / f"{collection}.npz")
        out += build(key, data, rng, next_id)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="recompute from the current data and compare with the committed file")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="heldout_store_")) / "store"          # Chroma can rewrite files when it opens a store
    shutil.copytree(ROOT / "RAG_data_layers" / "chroma_store", work)
    questions = generate(work, args.seed)
    if args.check:
        saved = json.loads(OUT.read_text())
        same = saved == json.loads(json.dumps(questions))
        print(f"{len(questions)} questions recomputed; identical to {OUT.name}: {same}")
        return 0 if same else 1
    OUT.write_text(json.dumps(questions, indent=1, ensure_ascii=False))
    print(f"Wrote {len(questions)} questions to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
