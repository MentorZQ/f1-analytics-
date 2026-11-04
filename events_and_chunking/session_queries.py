"""
session_queries.py
------------------
Questions that are not about one pair of drivers, answered by computing from the session instead of storing a
chunk for every possible question:

  grid_text              the whole grid: positions, teams, best laps, gaps, the tightest gaps
  team_text              one team: both cars, the gap between them, where the team's best car stands
  section_rankings       every section, every driver, in time order
  section_ranking_text   one section across the whole field
  team_advantage_text    for each section, which team's best car was quickest and by how much

Each returns plain text in the same style as the stored chunks, so the model (and the checks) treat it as evidence.
All figures come from the stored lap times and the session snapshot.
"""

from __future__ import annotations

import re

from events_and_chunking.chunk_telemetry import _time_at_distance

TIGHT_GAP_S = 0.100


def lap_text(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:06.3f}"


def _drivers_with_laps(ctx: dict) -> list[tuple[str, dict]]:
    rows = [(d, i) for d, i in ctx.get("drivers", {}).items() if i.get("lap_time_seconds") and i.get("qualifying_position")]
    return sorted(rows, key=lambda x: x[1]["qualifying_position"])


# ── the grid ──

def _best_session(info: dict):
    times = info.get("session_times") or {}
    return min(times, key=times.get) if times else None


def _gap_to_car_ahead(prev: dict | None, info: dict) -> str:
    """
    The gap to the classified car ahead, from the session that decided their order. Best laps can come from different
    sessions (a Q3 car's best may be a Q2 lap, an eliminated car's is from the session it went out in), and subtracting
    them mixes sessions, so with per-session times the gap is given only for two cars that went out in, or were
    classified from, the same session, and is labelled with it.
    """
    if prev is None:
        return ""
    pst, st = prev.get("session_times") or {}, info.get("session_times") or {}
    if not pst or not st:
        return f", {info['lap_time_seconds'] - prev['lap_time_seconds']:+.3f}s to the car ahead"
    stage = info.get("stage")
    if stage and stage == prev.get("stage"):
        n = int(stage[1])
        if n in pst and n in st:
            return f", {st[n] - pst[n]:+.3f}s to the car ahead (both Q{n} times)"
    return ", no like-for-like gap to the car ahead (they went out in different sessions or one set no time there)"


def grid_text(ctx: dict, session_label: str = "") -> str | None:
    rows = _drivers_with_laps(ctx)
    if not rows:
        return None
    pole_time = min(i["lap_time_seconds"] for _, i in rows)
    lines = [f"Qualifying grid{(' - ' + session_label) if session_label else ''} (best lap of each driver)"]
    prev_info = None
    for code, info in rows:
        t = info["lap_time_seconds"]
        ahead = _gap_to_car_ahead(prev_info, info)
        extra = ""
        if info.get("speed_trap_kmh"):
            extra += f"; speed trap {info['speed_trap_kmh']:g} km/h"
        if info.get("stage"):
            if info["stage"] == "Q3":
                extra += "; reached Q3" if 3 in (info.get("session_times") or {3: 0}) else "; reached Q3 but set no Q3 time"
            else:
                extra += f"; out in {info['stage']}"
        st = info.get("session_times") or {}
        if st:
            extra += "; " + ", ".join(f"Q{n} {lap_text(st[n])}" for n in sorted(st))
        lines.append(f"  P{info['qualifying_position']} {code} ({info.get('team', '')}) {lap_text(t)}  +{t - pole_time:.3f}s to pole{ahead}{extra}")
        prev_info = info
    times = sorted(i["lap_time_seconds"] for _, i in rows)
    by_time = sorted(rows, key=lambda x: x[1]["lap_time_seconds"])
    for n in (5, 10):
        if len(times) >= n:
            lines.append(f"Spread of the quickest {n} laps: {times[n - 1] - times[0]:.3f}s")
    tight = [(b[1]["lap_time_seconds"] - a[1]["lap_time_seconds"], a[0], b[0]) for a, b in zip(by_time, by_time[1:])]
    same_session = {c: _best_session(i) for c, i in rows}
    tight = sorted(g for g in tight if g[0] < TIGHT_GAP_S and (not same_session[g[1]] or same_session[g[1]] == same_session[g[2]]))
    if tight:
        lines.append("Cars separated by less than 0.100s in lap time: " + ", ".join(f"{a}/{b} {g:.3f}s" for g, a, b in tight))
    fastest = ctx.get("fastest_sector") or {}
    if fastest:
        lines.append(f"Fastest sector times in the session, over all {len(rows)} drivers: "
                     + ", ".join(f"sector {n} {t:.3f}s ({code})" for n, (t, code) in sorted(fastest.items())))
    trapped = sorted(((i["speed_trap_kmh"], c) for c, i in rows if i.get("speed_trap_kmh")), reverse=True)
    if trapped:
        top = trapped[:3]
        lines.append("Highest speed-trap readings: " + ", ".join(f"{c} {v:g} km/h" for v, c in top)
                     + f" (of {len(trapped)} drivers)")
    staged = {c: i.get("stage") for c, i in rows}
    if any(staged.values()):
        q3 = [c for c, st in staged.items() if st == "Q3"]
        q2 = [c for c, st in staged.items() if st == "Q2"]
        q1 = [c for c, st in staged.items() if st == "Q1"]
        q2_time = lambda c: (dict(rows)[c].get("session_times") or {}).get(2)             # noqa: E731
        parts = []
        if q3:
            via_q2 = sorted((q2_time(c), c) for c in q3 if q2_time(c))
            tail = f" (the slowest Q2 time among them: {via_q2[-1][1]} {lap_text(via_q2[-1][0])})" if via_q2 else ""
            parts.append(f"{len(q3)} reached Q3{tail}")
        if q2:
            slowest_in = sorted((q2_time(c), c) for c in q2 if q2_time(c))
            tail = f" (the quickest of them in Q2: {slowest_in[0][1]} {lap_text(slowest_in[0][0])})" if slowest_in else ""
            parts.append(f"{len(q2)} went out in Q2{tail}")
        if q1:
            parts.append(f"{len(q1)} went out in Q1")
        lines.append("Qualifying progression, from each driver's record: " + "; ".join(parts))
        lines.append("Who was last into Q3 is decided by the Q2 times (the slowest of the cars that advanced), not by the final "
                     "classification, which is set by Q3 laps. A driver classified in the top ten who set no Q3 time still reached Q3.")
        if q1 or q2:
            lines.append("Each gap to pole above compares the driver's best lap, which for a driver out in Q1 or Q2 is from that "
                         "earlier session, so it mixes sessions.")
    out_of_order = [c for (c, i), (_, j) in zip(rows, rows[1:]) if i["lap_time_seconds"] > j["lap_time_seconds"]]
    if out_of_order:
        lines.append("Classification order differs from best-lap order around: " + ", ".join(out_of_order))
    return "\n".join(lines)


# ── teams ──

def team_text(ctx: dict, team_name: str) -> str | None:
    rows = [(c, i) for c, i in _drivers_with_laps(ctx) if (i.get("team") or "").lower() == team_name.lower()]
    if not rows:
        return None
    pole_time = min(i["lap_time_seconds"] for _, i in _drivers_with_laps(ctx))
    lines = [f"Team summary: {team_name}"]
    for code, info in rows:
        lines.append(f"  {code}: P{info['qualifying_position']}, best lap {lap_text(info['lap_time_seconds'])}, "
                     f"+{info['lap_time_seconds'] - pole_time:.3f}s to pole")
    if len(rows) == 2:
        (a, ia), (b, ib) = rows
        lines.append(f"  Between teammates: {a if ia['lap_time_seconds'] <= ib['lap_time_seconds'] else b} was "
                     f"{abs(ia['lap_time_seconds'] - ib['lap_time_seconds']):.3f}s quicker.")
    best = min(rows, key=lambda r: r[1]["lap_time_seconds"])
    lines.append(f"  Best car: {best[0]} (P{best[1]['qualifying_position']}).")
    return "\n".join(lines)


def team_names(ctx: dict) -> list[str]:
    return sorted({i.get("team") for _, i in _drivers_with_laps(ctx) if i.get("team")})


# ── sections across the field ──

def section_times(snapshot) -> list[dict]:
    """Per section: its id, window and every driver's time through it, interpolated at the exact boundaries."""
    out = []
    for event in snapshot.events:
        times = {}
        for driver, tel in snapshot.telemetry.items():
            t0 = _time_at_distance(tel, event["dist_start"])
            t1 = _time_at_distance(tel, event["dist_end"])
            if t0 is not None and t1 is not None and t1 > t0:
                times[driver] = t1 - t0
        out.append({"id": event.get("display_id", event["id"]), "type": event["type"], "start": event["dist_start"],
                    "end": event["dist_end"], "corner_numbers": event.get("corner_numbers") or [], "times": times})
    return out


def sections_for_corner(sections: list[dict], number: int) -> list[dict]:
    return [s for s in sections if number in s["corner_numbers"]]


def section_ranking_text(section: dict, ctx: dict) -> str | None:
    if not section["times"]:
        return None
    ranked = sorted(section["times"].items(), key=lambda kv: kv[1])
    best = ranked[0][1]
    teams = {c: i.get("team", "") for c, i in ctx.get("drivers", {}).items()}
    lines = [f"Section ranking: {section['id']} ({section['start']:.0f}m - {section['end']:.0f}m), all drivers, quickest first"]
    for place, (code, t) in enumerate(ranked, 1):
        lines.append(f"  {place}. {code} ({teams.get(code, '')}) {t:.3f}s" + ("" if place == 1 else f"  +{t - best:.3f}s"))
    return "\n".join(lines)


def team_advantage_text(snapshot, ctx: dict) -> str | None:
    """For each section, which team's best car was quickest, and by how much over the next team."""
    teams = {c: i.get("team", "") for c, i in ctx.get("drivers", {}).items()}
    sections = section_times(snapshot)
    wins: dict[str, int] = {}
    lines = ["Team advantage by section (each team's quickest car; margin over the next-best team)"]
    for s in sections:
        best_by_team: dict[str, tuple[float, str]] = {}
        for code, t in s["times"].items():
            team = teams.get(code)
            if team and (team not in best_by_team or t < best_by_team[team][0]):
                best_by_team[team] = (t, code)
        if len(best_by_team) < 2:
            continue
        ranked = sorted(best_by_team.items(), key=lambda kv: kv[1][0])
        (lead, (t1, c1)), (second, (t2, _)) = ranked[0], ranked[1]
        wins[lead] = wins.get(lead, 0) + 1
        lines.append(f"  {s['id']} ({s['start']:.0f}m - {s['end']:.0f}m): {lead} ({c1}) {t1:.3f}s, "
                     f"{t2 - t1:.3f}s ahead of {second}")
    if len(lines) == 1:
        return None
    lines.append("Sections led by each team: " + ", ".join(f"{t} {n}" for t, n in sorted(wins.items(), key=lambda kv: -kv[1])))
    return "\n".join(lines)


def corner_facts_text(sections: list[dict]) -> str | None:
    """Which corner is first and which is last. Section chunks never say, so the model guessed (it called T13 the last corner of a 14-corner lap)."""
    numbers = sorted({n for s in sections for n in s["corner_numbers"]})
    if not numbers:
        return None
    first, last = numbers[0], numbers[-1]
    by_number = {n: next(s["id"] for s in sections if n in s["corner_numbers"]) for n in (first, last)}
    return (f"Circuit facts: {len(numbers)} numbered corners, T{first} to T{last}.\n"
            f"The first corner is T{first} (section {by_number[first]}). The final corner is T{last} (section {by_number[last]}), "
            f"after which the lap runs down the main straight to the line.")


def make_chunk(kind: str, text: str, label: str, session_key: str) -> dict:
    return {"chunk_id": f"{kind}:{session_key}:{label}", "text": text, "distance": 0.0, "rank": 0,
            "metadata": {"chunk_type": kind, "event_id": label if kind == "section_ranking" else None}}


# ── what the question asks for ──

_TEAM_ADVANTAGE = re.compile(
    r"\b(?:each|every|all|per|by)\s+team\b|\bwhich\s+team\b|\bbest\s+team\b|\bteams?\b[^.?]{0,40}\b(?:advantage|faster|quicker|best|fastest|strongest|stronger|edge)\b|"
    r"\b(?:advantage|faster|quicker|best|fastest|strongest|edge)\b[^.?]{0,40}\bteams?\b", re.I)
_FIELD = re.compile(
    r"\bclose together\b|\bclosely\b|\btight(?:ly)?\b|\bbunched\b|\bspread\b|\bcompress\w*\b|\bgaps?\b|\bwhole field\b|\bthe field\b|"
    r"\bmidfield\b|\bgrid\b|\btop\s*(?:3|three|5|five|10|ten)\b|\bhow many (?:drivers|cars)\b|\bclassification\b|\bpecking order\b|"
    r"\b(?:fastest|quickest|best|slowest)\s+sector\b|\bsector\s*[123]\b[^.?!]{0,25}\b(?:fastest|quickest|best)\b|"
    r"\bspeed[- ]?trap\b|\btop speed\b|\bhighest speed\b|\bstraight[- ]line speed\b|"
    r"\bq[123]\b|\bknocked out\b|\beliminated\b|\bmade it (?:into|to|through)\b|\bcut ?off\b|\b(?:first|last) (?:car|driver)\b|"
    r"\bdominan\w*\b|\bmargin\b|\bclose(?:ly)? (?:together|matched|fought)\b|\bwho (?:made|got) (?:into )?q[123]\b", re.I)
_SECTION_BEST = re.compile(r"\b(?:fastest|quickest|best|slowest|worst|faster|quicker|better)\b", re.I)
_CORNER_REF = re.compile(r"(?<![\w-])T(\d+)\b|\bturn\s+(\d+)\b", re.I)
_FIRST_LAST = [(re.compile(r"\b(?:first|opening)\s+(?:corner|turn)\b", re.I), "first"),
               (re.compile(r"\b(?:last|final)\s+(?:corner|turn)\b", re.I), "last")]


def corner_numbers_asked(question: str, sections: list[dict]) -> list[int]:
    nums = [int(a or b) for a, b in _CORNER_REF.findall(question)]
    all_numbers = sorted({n for s in sections for n in s["corner_numbers"]})
    for pattern, which in _FIRST_LAST:
        if pattern.search(question) and all_numbers:
            nums.append(all_numbers[0] if which == "first" else all_numbers[-1])
    return list(dict.fromkeys(nums))


def names_a_corner(question: str) -> bool:
    return bool(_CORNER_REF.search(question)) or any(p.search(question) for p, _ in _FIRST_LAST)


def plan(question: str, named_teams: list[str], no_driver_named: bool) -> dict:
    """Which whole-session views the question asks for. Pure text rules, no model."""
    return {
        "corner_facts": names_a_corner(question),
        "team_advantage": bool(_TEAM_ADVANTAGE.search(question)) and not named_teams,
        "grid": bool(_FIELD.search(question)) or bool(named_teams),
        "teams": named_teams,
        "section_best": bool(_SECTION_BEST.search(question)) and no_driver_named,
    }
