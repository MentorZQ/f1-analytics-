"""
lookup_router.py
----------------
Answers one-fact questions straight from the session data, with no model call.

"What was Sainz's best lap?", "where did Albon qualify?", "who got pole?", "did Gasly make Q3?" have one right answer in the
stored classification. Sending them to a language model costs money and time and is the only place a wrong answer can come
from, so they are answered here from the same figures the model would have been shown.

The router is built to be right or silent:
  - it answers only when exactly one intent matches the question (a question asking for two facts goes to the model);
  - the drivers it needs must be named unambiguously and be in the session's data;
  - anything else (a comparison, a "why", an intent it does not know) returns None and the normal pipeline runs.

`route(question, ctx, named)` is pure: ctx is the session context from query.load_session_context, named is the set of
driver codes the question is about. It returns a Routed answer or None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from events_and_chunking.session_queries import _best_session, lap_text


@dataclass
class Routed:
    answer: str
    route: str          # which intent answered, for traces and the evaluation
    facts: str          # the figures the answer rests on, shown as the source
    drivers: list[str]


# Words that make a question something other than one fact about one driver or the field.
_NOT_A_LOOKUP = re.compile(
    r"\bwhy\b|\bhow did\b|\bcompare|\bexplain\b|\bvs\.?\b|\bversus\b|\bthan\b|\bbetween\b|\bbeat\b|\brelative to\b"
    r"|\bimprove|\bslower\b|\bfaster\b|\bgain|\blos[et]\b|\bcorners?\b|\bturns?\b|(?<![\w-])T\d+\b|\bstraights?\b|\bbraking\b|\bthrottle\b"
    # asking where or how something happened is analysis, even when it mentions a best lap
    r"|\bwhere (?:was|were|is)\b|\bstrong|\bweak|\bstrength|\bstruggl|\blimit|\bwent wrong|\bheld\b|\bmade the difference"
    r"|\bhow\b(?!\s+(?:fast|far)\b)"
    # topics the router has no figures for: whatever intent words appear, these questions need the model
    r"|\btyres?\b|\btires?\b|\bcompound|\bsofts?\b|\bmediums?\b|\bhards?\b|\bused\b|\bnew set\b|\bpenalt|\bweather|\brain|\btemperature"
    r"|\bred flag|\bdrs\b|\bgears?\b|\bfuel\b|\bupgrade|\bslipstream|\btow\b|\bdamage|\bcrash|\bsteward|\bdeleted\b|\btrack limits\b",
    re.I)

# Two questions in one: answering only one of them would be a half answer, so the model takes it.
_MULTI_PART = re.compile(r"\band\b[^?]{0,60}\b(?:what|which|who|where|how|why|did|was|were|is)\b|\balso\b|\bas well\b|\bplus\b|;|\?[^?]*\?", re.I)

_QN = re.compile(r"\bq([123])\b", re.I)

_INTENTS: dict[str, re.Pattern] = {
    "lap_time": re.compile(
        r"\b(?:best|fastest|quickest)\s+(?:qualifying\s+)?(?:lap|time)\b|\blap\s*time\b|\bhow fast was\b|\bquali(?:fying)?\s+time\b"
        r"|\bwhat time did\b|\bq[123]\s+(?:lap\s*)?time\b", re.I),
    "position": re.compile(
        r"\bwhere did\b[^?]{0,40}\b(?:qualify|start|line[d]? up)\b|\b(?:grid|qualifying|starting)\s+(?:position|slot|spot)\b"
        r"|\bwhat (?:position|place)\b|\bstart(?:ed|ing)?\s+(?:from|position)\b", re.I),
    "gap_to_pole": re.compile(
        r"\bgap to pole\b|\bfar (?:off|behind|from|adrift)\b[^?]{0,15}\bpole\b|\b(?:off|behind|adrift of)\b[^?]{0,10}\bpole\b"
        r"|\bpole\b[^?]{0,20}\bgap\b", re.I),
    "stage": re.compile(
        r"\bmake (?:it )?(?:in)?to\s+q[23]\b|\bmake q[23]\b|\bget (?:in)?to q[23]\b|\bget through to q[23]\b|\breach(?:ed)? q[23]\b"
        r"|\bhow far did\b[^?]{0,30}\bget\b|\bknocked out\b|\beliminated\b|\bout in q[123]\b", re.I),
    "pole": re.compile(
        r"\bwho\b[^?]{0,40}\bpole\b|\bpole[- ]?sitter\b|\bwho\b[^?]{0,25}\b(?:fastest|quickest)\s+(?:qualifying\s+)?lap\b", re.I),
    "last_into_q3": re.compile(
        r"\blast\b[^?]{0,30}\b(?:into|to|through)\b[^?]{0,20}\bq3\b|\bslowest q2\b|\bonly just\b[^?]{0,30}\bq3\b|\bscraped\b[^?]{0,15}\bq3\b",
        re.I),
    "who_at_position": re.compile(r"\bwho\b[^?]{0,25}\b(?:p(\d{1,2})|(\d{1,2})(?:st|nd|rd|th))\b", re.I),
    "session_top": re.compile(
        r"\bwho\b[^?]{0,25}\b(?:topped|top of|led|fastest in|quickest in|set the fastest time in|set the quickest time in)\s+q([123])\b",
        re.I),
    "speed_trap": re.compile(r"\bspeed[- ]?trap\b", re.I),
    "fastest_sector": re.compile(
        r"\b(?:quickest|fastest)\s+(?:in\s+)?sector\s*([123])\b|\bsector\s*([123])\b[^?]{0,20}\b(?:quickest|fastest)\b", re.I),
}
_SLOWEST = re.compile(r"\b(?:slowest|lowest|worst|last)\b", re.I)


def matched_intents(question: str) -> list[str]:
    return [name for name, pattern in _INTENTS.items() if pattern.search(question)]


def _team(ctx: dict, code: str) -> str:
    info = (ctx.get("drivers") or {}).get(code) or (ctx.get("no_time") or {}).get(code) or {}
    return info.get("team", "")


def _label(ctx: dict, code: str) -> str:
    team = _team(ctx, code)
    return f"{code} ({team})" if team else code


def _session_tag(info: dict) -> str:
    n = _best_session(info)
    return f"Q{n}" if n else ""


def _facts_line(ctx: dict, code: str) -> str:
    info = (ctx.get("drivers") or {}).get(code)
    if not info:
        reason = ((ctx.get("no_time") or {}).get(code) or {}).get("reason", "no timed lap")
        return f"{code} ({_team(ctx, code)}): no timed lap ({reason})"
    parts = [f"{code} ({info.get('team', '')})", f"qualifying position P{info.get('qualifying_position')}"]
    if info.get("lap_time_seconds"):
        tag = _session_tag(info)
        parts.append(f"best lap {lap_text(info['lap_time_seconds'])}" + (f" ({tag})" if tag else ""))
    times = info.get("session_times") or {}
    if times:
        parts.append("session times " + ", ".join(f"Q{n} {lap_text(t)}" for n, t in sorted(times.items())))
    if info.get("stage"):
        parts.append(f"went out in / reached {info['stage']}")
    return ", ".join(parts)


# ── one builder per intent; each returns Routed or None ──

def _lap_time(question, ctx, named):
    if len(named) != 1:
        return None
    code = next(iter(named))
    info = (ctx.get("drivers") or {}).get(code)
    if not info or not info.get("lap_time_seconds"):
        return None
    qn = _QN.findall(question)
    if len(qn) > 1:
        return None
    if qn:
        t = (info.get("session_times") or {}).get(int(qn[0]))
        if t is None:
            return None                              # no time in that session: the model explains why (stopped on track, ...)
        return Routed(f"{_label(ctx, code)} set a Q{qn[0]} lap of {lap_text(t)}.", "lap_time", _facts_line(ctx, code), [code])
    tag = _session_tag(info)
    return Routed(f"{_label(ctx, code)} set a best lap of {lap_text(info['lap_time_seconds'])}" + (f" in {tag}." if tag else "."),
                  "lap_time", _facts_line(ctx, code), [code])


def _position(question, ctx, named):
    if len(named) != 1 or _QN.search(question):
        return None
    code = next(iter(named))
    info = (ctx.get("drivers") or {}).get(code)
    if not info or not info.get("qualifying_position"):
        return None
    return Routed(f"{_label(ctx, code)} qualified P{info['qualifying_position']}, before any grid penalties.",
                  "position", _facts_line(ctx, code), [code])


def _gap_to_pole(question, ctx, named):
    if len(named) != 1 or _QN.search(question):
        return None
    code = next(iter(named))
    drivers = ctx.get("drivers") or {}
    pole = ctx.get("pole")
    info, pole_info = drivers.get(code), drivers.get(pole) if pole else None
    if not info or not pole_info or not info.get("lap_time_seconds") or not pole_info.get("lap_time_seconds"):
        return None
    facts = _facts_line(ctx, code) + "\n" + _facts_line(ctx, pole)
    if code == pole:
        return Routed(f"{_label(ctx, code)} took pole, so there is no gap to it.", "gap_to_pole", facts, [code])
    gap = info["lap_time_seconds"] - pole_info["lap_time_seconds"]
    mine, theirs = _session_tag(info), _session_tag(pole_info)
    where = f" ({code}'s {mine} lap against {pole}'s {theirs} lap)" if mine and theirs and mine != theirs else ""
    return Routed(f"{_label(ctx, code)} was {gap:.3f}s off pole{where}: {lap_text(info['lap_time_seconds'])} against "
                  f"{pole}'s {lap_text(pole_info['lap_time_seconds'])}.", "gap_to_pole", facts, [code, pole])


def _stage(question, ctx, named):
    if len(named) != 1:
        return None
    code = next(iter(named))
    drivers, no_time = ctx.get("drivers") or {}, ctx.get("no_time") or {}
    target = _QN.findall(question)
    target_n = int(target[-1]) if target else None
    if code in no_time and code not in drivers:
        stage, pos, label = "Q1", None, _label(ctx, code)
    else:
        info = drivers.get(code)
        if not info or info.get("stage") not in ("Q1", "Q2", "Q3"):
            return None
        stage, pos, label = info["stage"], info.get("qualifying_position"), _label(ctx, code)
    n = int(stage[1])
    where = f", qualifying P{pos}" if pos else ", having set no timed lap"
    if target_n in (2, 3) and re.search(r"\bmake|\bget (?:in)?to|\bget through|\breach", question, re.I):
        if n >= target_n:
            extra = ""
            if target_n == 3 and 3 not in ((drivers.get(code) or {}).get("session_times") or {}):
                extra = " but set no time in Q3"
            return Routed(f"{label} made Q{target_n}{extra}{where}.", "stage", _facts_line(ctx, code), [code])
        return Routed(f"{label} did not make Q{target_n}; he went out in {stage}{where}.", "stage", _facts_line(ctx, code), [code])
    if n == 3:
        return Routed(f"{label} reached Q3{where}.", "stage", _facts_line(ctx, code), [code])
    return Routed(f"{label} went out in {stage}{where}.", "stage", _facts_line(ctx, code), [code])


def _pole(question, ctx, named):
    if named or _QN.search(question):
        return None
    drivers = ctx.get("drivers") or {}
    code = ctx.get("pole")
    info = drivers.get(code) if code else None
    if not info or not info.get("lap_time_seconds"):
        return None
    tag = _session_tag(info)
    return Routed(f"{_label(ctx, code)} took pole with a {lap_text(info['lap_time_seconds'])}" + (f" in {tag}." if tag else "."),
                  "pole", _facts_line(ctx, code), [code])


def _last_into_q3(question, ctx, named):
    if named:
        return None
    drivers = ctx.get("drivers") or {}
    in_q3 = {c: i for c, i in drivers.items() if i.get("stage") == "Q3"}
    if len(in_q3) < 2 or any(2 not in (i.get("session_times") or {}) for i in in_q3.values()):
        return None                                   # needs every Q3 driver's Q2 time, or the answer could be wrong
    ranked = sorted(in_q3, key=lambda c: in_q3[c]["session_times"][2], reverse=True)
    if in_q3[ranked[0]]["session_times"][2] == in_q3[ranked[1]]["session_times"][2]:
        return None
    code = ranked[0]
    t = in_q3[code]["session_times"][2]
    facts = f"Q2 times of the {len(in_q3)} drivers who reached Q3, slowest first: " + ", ".join(
        f"{c} {lap_text(in_q3[c]['session_times'][2])}" for c in ranked)
    return Routed(f"{_label(ctx, code)} was last into Q3, with a Q2 time of {lap_text(t)}.", "last_into_q3", facts, [code])


def _speed_trap(question, ctx, named):
    drivers = ctx.get("drivers") or {}
    if _QN.search(question):
        return None
    if len(named) == 1:
        code = next(iter(named))
        v = (drivers.get(code) or {}).get("speed_trap_kmh")
        if v is None:
            return None
        return Routed(f"{_label(ctx, code)}'s speed trap reading was {v:.1f} km/h.", "speed_trap", _facts_line(ctx, code)
                      + f", speed trap {v:.1f} km/h", [code])
    if named or _SLOWEST.search(question):
        return None
    readings = {c: i["speed_trap_kmh"] for c, i in drivers.items() if i.get("speed_trap_kmh") is not None}
    if len(readings) < 2:
        return None
    ranked = sorted(readings, key=readings.get, reverse=True)
    leaders = [c for c in ranked if readings[c] == readings[ranked[0]]]
    facts = "Speed trap readings, highest first: " + ", ".join(f"{c} {readings[c]:.1f} km/h" for c in ranked[:5])
    if len(leaders) > 1:
        names = " and ".join(_label(ctx, c) for c in leaders)
        return Routed(f"{names} tied for the highest speed trap reading, {readings[leaders[0]]:.1f} km/h.", "speed_trap", facts, leaders)
    code = leaders[0]
    return Routed(f"{_label(ctx, code)} had the highest speed trap reading, {readings[code]:.1f} km/h.", "speed_trap", facts, [code])


def _who_at_position(question, ctx, named):
    if named:
        return None
    m = _INTENTS["who_at_position"].search(question)
    n = int(next(g for g in m.groups() if g))
    standings = ctx.get("standings") or []
    if not 1 <= n <= len(standings):
        return None
    code = standings[n - 1]
    info = (ctx.get("drivers") or {}).get(code) or {}
    if info.get("qualifying_position") != n:
        return None                                   # the order in the data does not match the position asked for
    return Routed(f"{_label(ctx, code)} qualified P{n}, before any grid penalties.", "who_at_position", _facts_line(ctx, code), [code])


def _session_top(question, ctx, named):
    if named:
        return None
    n = int(_INTENTS["session_top"].search(question).group(1))
    times = {c: i["session_times"][n] for c, i in (ctx.get("drivers") or {}).items() if n in (i.get("session_times") or {})}
    if len(times) < 2:
        return None
    ranked = sorted(times, key=times.get)
    if times[ranked[0]] == times[ranked[1]]:
        return None
    code = ranked[0]
    facts = f"Q{n} times, quickest first: " + ", ".join(f"{c} {lap_text(times[c])}" for c in ranked[:5])
    return Routed(f"{_label(ctx, code)} was quickest in Q{n}, with {lap_text(times[code])}.", "session_top", facts, [code])


def _fastest_sector(question, ctx, named):
    if named:
        return None
    m = _INTENTS["fastest_sector"].search(question)
    n = int(next(g for g in m.groups() if g))
    best = (ctx.get("fastest_sector") or {}).get(n)
    if not best:
        return None
    seconds, code = best
    return Routed(f"{_label(ctx, code)} set the quickest sector {n} time, {seconds:.3f}s.", "fastest_sector",
                  f"Quickest sector {n} in the session: {seconds:.3f}s ({code}).", [code])


_BUILDERS = {
    "lap_time": _lap_time, "position": _position, "gap_to_pole": _gap_to_pole, "stage": _stage, "pole": _pole,
    "last_into_q3": _last_into_q3, "speed_trap": _speed_trap, "fastest_sector": _fastest_sector,
    "who_at_position": _who_at_position, "session_top": _session_top,
}

# How many drivers the question must name for each intent: (fewest, most). An intent that cannot apply to this question
# is ignored, so "who was the last to get through to Q3" is not also read as "did X get through to Q3".
_DRIVERS_NEEDED = {
    "lap_time": (1, 1), "position": (1, 1), "gap_to_pole": (1, 1), "stage": (1, 1),
    "pole": (0, 0), "last_into_q3": (0, 0), "fastest_sector": (0, 0), "who_at_position": (0, 0), "session_top": (0, 0),
    "speed_trap": (0, 1),
}


def route(question: str, ctx: dict, named: set[str]) -> Routed | None:
    """The answer to a one-fact question, or None when the question needs the model."""
    if not question or _NOT_A_LOOKUP.search(question) or _MULTI_PART.search(question):
        return None
    intents = [i for i in matched_intents(question) if _DRIVERS_NEEDED[i][0] <= len(named) <= _DRIVERS_NEEDED[i][1]]
    if len(intents) != 1:
        return None                                   # no intent we know, or a question asking for more than one fact
    entrants = ctx.get("entrants") or set()
    if any(code not in entrants for code in named):
        return None
    return _BUILDERS[intents[0]](question, ctx, named)
