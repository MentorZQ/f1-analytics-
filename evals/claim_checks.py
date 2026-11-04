"""
claim_checks.py
---------------
More deterministic checks on a RAG answer. code_checks.py asks "are the numbers and hedges right?";
these ask "is the answer about the right thing, and does it say the right thing about it?".

  reversed_winner          : the answer names the slower driver as the faster one in a section
  wrong_driver_or_session  : a driver, circuit or year appears that is not in the question or the evidence
  numbers_on_right_driver  : a speed is attached to a driver whose chunk gives it to the other driver
  completeness             : both named drivers, and a named corner, are actually addressed
  false_premise            : a question assumes something the evidence never shows (a crash, a penalty);
                             the answer must say so instead of explaining it
  overclaiming             : "clearly" / "definitely" about braking in a section the data could not resolve
  team_attribution         : a driver is named together with the wrong team ("Haas's Hadjar")
  team_ranking             : "the fourth quickest team" must match the ranking of the teams' cars in the grid
  grid_start_claim         : "started from P17" when the data has only the qualifying classification, not the final grid after penalties
  tyre_claims              : blaming tyre compound or age for a gap when the evidence shows the same tyres (or none)
  self_correction          : "wait, corrected:" - the model changed its mind in the answer text

Same contract as code_checks: pure functions of (answer, evidence), no network, same verdict every time.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from evals import code_checks as cc
from evals.code_checks import CheckResult

_RAG = Path(__file__).resolve().parent.parent / "RAG_data_layers" / "F1 Project Thoughts"
if str(_RAG) not in sys.path:
    sys.path.insert(0, str(_RAG))
from rag_retrieval import TEAM_ALIASES, extract_driver_codes  # noqa: E402

SESSION_WORDS = {
    "barcelona": ("barcelona", "spain", "spanish", "catalunya", "catalan"),
    "melbourne": ("melbourne", "australia", "australian", "albert park"),
    "monza": ("monza", "italy", "italian"),
}
SESSION_YEAR = {"barcelona": 2026, "melbourne": 2026, "monza": 2024}


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(word)}(?![\w])", text, re.IGNORECASE) is not None


def _source_text(sources: list[dict]) -> str:
    return "\n".join(s.get("text") or "" for s in sources)


def _drivers_in(text: str) -> set[str]:
    return extract_driver_codes(text)


# ── wrong driver or session ──────────────────────────────────────────────────

def wrong_driver_or_session(answer: str, sources: list[dict], question: str | None = None,
                            session: str | None = None) -> CheckResult:
    """A driver, circuit or year in the answer must be in the question or the evidence."""
    known_text = (question or "") + "\n" + _source_text(sources)
    allowed = _drivers_in(known_text) | {d for s in sources for d in (s.get("drivers") or [])}
    stray_drivers = sorted(_drivers_in(answer) - allowed)

    problems = []
    if stray_drivers:
        problems.append(f"driver(s) not in the question or evidence: {', '.join(stray_drivers)}")
    if session in SESSION_WORDS:
        asked = (question or "").lower()
        for other, words in SESSION_WORDS.items():
            if other == session:
                continue
            hit = next((w for w in words if _has_word(answer, w) and not _has_word(asked, w)
                        and not _has_word(_source_text(sources), w)), None)
            if hit:
                problems.append(f"mentions {hit!r}, which belongs to the {other} session, not {session}")
        years = {int(y) for y in re.findall(r"\b(20\d\d)\b", answer)}
        shown = {int(y) for y in re.findall(r"\b(20\d\d)\b", known_text)}
        wrong_years = sorted(y for y in years if y != SESSION_YEAR[session] and y not in shown)
        if wrong_years:
            problems.append(f"year(s) {wrong_years} do not match the {session} session ({SESSION_YEAR[session]})")
    if problems:
        return CheckResult("wrong_driver_or_session", "fail", 0.0, "; ".join(problems), problems)
    return CheckResult("wrong_driver_or_session", "pass", 1.0, "Every driver, circuit and year is in the question or the evidence.")


# ── reversed winner ──────────────────────────────────────────────────────────

def _name_alternatives(code: str) -> str:
    surname = cc.SURNAMES.get(code)
    parts = [rf"\b{code}\b"]                                    # codes in capitals only: "ham" is also a word
    if surname:
        parts.append(rf"(?i:\b{surname}\b)")
    return "(?:" + "|".join(parts) + ")"


def _driver_mentions(text: str, a: str, b: str) -> list[tuple[int, int, str]]:
    found = []
    for d in (a, b):
        found += [(m.start(), m.end(), d) for m in re.finditer(_name_alternatives(d), text)]
    return sorted(found)


def claimed_winner_for_time(unit: str, value: float, a: str, b: str) -> str | None:
    """
    Which of drivers a / b does this sentence or table row say was faster, judged only from the words
    attached to the time `value` ("HAM faster by 0.092s", "0.035s quicker", "+0.125s to RUS",
    "RUS gained 0.1s", "was 0.101s slower"). None when no such wording sits next to the time.
    """
    t = cc._norm(unit)
    verdicts = set()
    for m in re.finditer(r"(?<![\d.])(\d+\.\d+)\s*s\b", t):
        if abs(float(m.group(1)) - value) > cc.TOL["s"]:
            continue
        before, after = t[: m.start()], t[m.end():]

        def last_driver(text: str):
            """The driver the clause is about: the last one named, unless a pronoun or possessive muddies it."""
            ms = _driver_mentions(text, a, b)
            outside = _driver_mentions(re.sub(r"\([^()]*\)", lambda m: " " * len(m.group()), text), a, b)
            if outside:                                           # a driver named only inside a parenthetical is not the subject
                ms = outside
            if not ms:
                return None
            _, end, code = ms[-1]
            tail = re.sub(r"\([^()]*\)", " ", text[end:])
            if re.match(r"\s*['\u2019]s\b", text[end:]) or re.search(
                    r"\b(?:he|she|they|him|his|her|who|which|that(?! (?:section|corner|sequence|straight|lap|window|stretch|zone|point)))\b", tail, re.I):
                return None                                       # "Hamilton's ...", "... where he was 0.3s faster": cannot tell who
            return code

        other = {a: b, b: a}
        cue = re.search(r"\b(faster|quicker|slower)\s*(?:by\s*)?[~\u2248+\-]?\s*\(?\s*$", before)         # "HAM faster by 0.092s"
        if cue:
            subject = last_driver(before[: cue.start()])
            if subject:
                verdicts.add(subject if cue.group(1) != "slower" else other[subject])
                continue
        cue = re.match(r"\s*(faster|quicker|slower|longer)\b", after)                                      # "0.035s quicker", "took 0.46s longer"
        if cue:
            subject = last_driver(before)
            if subject:
                verdicts.add(subject if cue.group(1) not in {"slower", "longer"} else other[subject])
                continue
        to = re.match(r"\s*\)?\s*(?:to|for)\s+", after)                                                     # "+0.125s to RUS"
        if to:
            rest = after[to.end():]
            hit = [d for d in (a, b) if re.match(_name_alternatives(d), rest)]
            if len(hit) == 1:
                verdicts.add(hit[0])
                continue
        verb = re.search(r"(" + "|".join(_name_alternatives(d) for d in (a, b)) +
                         r")\s+(gained|gains|won|wins|took|takes|picked up|made up|lost|loses|dropped|conceded)\s+(?:about\s+)?[~\u2248+\-]?\s*$", before)
        if verb:                                                                                             # "RUS gained 0.1s"
            subject = next(d for d in (a, b) if re.fullmatch(_name_alternatives(d), verb.group(1)))
            verdicts.add(subject if verb.group(2) in {"gained", "gains", "won", "wins", "took", "takes", "picked up", "made up"} else other[subject])
    if not verdicts and t.lstrip().startswith("|"):                                                          # table: a cell that is just a driver
        cells = [c.strip() for c in t.split("|")]
        present = [d for d in (a, b) if any(re.fullmatch(_name_alternatives(d), c) for c in cells)]
        if len(present) == 1:
            verdicts.add(present[0])
    return next(iter(verdicts)) if len(verdicts) == 1 else None


def _units(answer: str):
    """(heading text, unit) for every sentence and table row, with the heading each sits under."""
    heading = ""
    for line in answer.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^#{1,6}\s", stripped) or (re.match(r"^\*\*[^*]+\*\*$", stripped) and cc._SECTION_REF.search(cc._norm(stripped))):
            heading = stripped
            continue
        if stripped.startswith("|"):
            yield heading, stripped
        else:
            for sent in cc._sentences(stripped):
                yield heading, sent


def reversed_winner(answer: str, sources: list[dict], deltas: dict | None = None) -> CheckResult:
    """If the answer says which driver was faster in a section, it must be the one the stored times say."""
    if not deltas or not deltas.get("sections") or not deltas.get("driver_a") or not deltas.get("driver_b"):
        return CheckResult("reversed_winner", "n/a", None, "No driver-pair comparison to check winners against.")
    a, b = deltas["driver_a"], deltas["driver_b"]
    truth = {s["section"]: (a if s["gain_s"] > 0 else b, abs(s["gain_s"]), s.get("start_m"))
             for s in deltas["sections"] if abs(s["gain_s"]) >= 0.0005}
    checked, wrong = 0, []
    for heading, unit in _units(answer):
        times = [v for v, u in cc.numbers_with_units(unit) if u == "s"]
        if not times:
            continue
        matched = [sec for sec, (_, gap, start) in truth.items()
                   if any(abs(t - gap) <= cc.TOL["s"] for t in times)
                   and (cc.section_mentioned(unit, sec, start) or (heading and cc.section_mentioned(heading, sec, start)))]
        if len(matched) != 1:
            continue                      # no section, or a sentence about several: the claim cannot be pinned to one
        sec = matched[0]
        winner, gap, _ = truth[sec]
        claimed = claimed_winner_for_time(unit, gap, a, b)
        if claimed is None:
            continue
        checked += 1
        if claimed != winner:
            wrong.append({"section": sec, "said": claimed, "stored": winner, "gap_s": round(gap, 3), "sentence": unit[:220]})
    if wrong:
        return CheckResult("reversed_winner", "fail", 0.0,
                           f"{len(wrong)} of {checked} winner claim(s) name the slower driver: "
                           + ", ".join(f"{w['section']} (said {w['said']}, stored {w['stored']})" for w in wrong), wrong)
    if not checked:
        return CheckResult("reversed_winner", "n/a", None, "No section winner could be read from the answer.")
    return CheckResult("reversed_winner", "pass", 1.0, f"All {checked} section winner claim(s) match the stored times.")


# ── numbers on the right driver ──────────────────────────────────────────────

_CODE_SPEED = re.compile(r"\b([A-Z]{3})\s+(\d+(?:\.\d+)?)\s*km/h")
_DELTA_AFTER = re.compile(r"\s*(?:more|less|faster|slower|higher|lower|quicker|advantage|gain|deficit|better|worse)\b", re.I)


def driver_speed_values(sources: list[dict]) -> dict[str, set[float]]:
    values: dict[str, set[float]] = {}
    for s in sources:
        for code, v in _CODE_SPEED.findall(s.get("text") or ""):
            values.setdefault(code, set()).add(round(float(v), 1))
        for code, rest in re.findall(r"^[ \t]*([A-Z]{3}):[ \t]*(.+)$", s.get("text") or "", re.M):
            for v, unit in cc.numbers_with_units(rest):      # "  ALO: first brakes at 5018m from 308.0 km/h": a line led by his code
                if unit == "km/h":
                    values.setdefault(code, set()).add(round(v, 1))
        owners = s.get("drivers") or []
        if len(owners) == 1:                    # a one-driver chunk (lap summary: speed traps, ...): every speed in it is his
            for v, unit in cc.numbers_with_units(s.get("text") or ""):
                if unit == "km/h":
                    values.setdefault(owners[0], set()).add(round(v, 1))
    return values


def numbers_on_right_driver(answer: str, sources: list[dict]) -> CheckResult:
    """'Hamilton … 152 km/h' must be a speed Hamilton's chunk gives, not the other driver's."""
    per_driver = driver_speed_values(sources)
    if len(per_driver) < 2:
        return CheckResult("numbers_on_right_driver", "n/a", None, "The evidence has no per-driver speeds to check against.")
    mention = re.compile("|".join(_name_alternatives(d) for d in sorted(per_driver)))
    owner = {}
    for d in per_driver:
        owner[d] = re.compile(_name_alternatives(d))
    checked, swapped = 0, []
    text = cc._norm(answer)
    for line in text.split("\n"):
        marks = [(m.start(), m.end(), next(d for d in per_driver if owner[d].fullmatch(m.group()))) for m in mention.finditer(line)]
        for k, (start, end, d) in enumerate(marks):
            stop = marks[k + 1][0] if k + 1 < len(marks) else min(len(line), end + 70)
            span = line[end:stop]
            m = re.search(r"(?<![\w.])\+?(\d+(?:\.\d+)?)\s*km/h", span)
            if not m:
                continue
            between = span[: m.start()]
            if re.search(r"\bvs\.?\b|versus|than|against|compared|\band\b|,|;|\(|[.!?]\s", between, re.I) or _DELTA_AFTER.match(span[m.end():]) \
                    or line[end + m.start(): end + m.start() + 1] == "+" \
                    or re.match(r"\s*(?:vs\.?|versus|/|and|,\s*(?:vs|and))\b", span[m.end():], re.I):
                continue                                       # ambiguous (first of an 'x vs y' pair), or a difference
            value = round(float(m.group(1)), 1)
            checked += 1
            mine = per_driver[d]
            theirs = set().union(*(v for k2, v in per_driver.items() if k2 != d))
            if not any(abs(value - x) <= cc.TOL["km/h"] for x in mine) and any(abs(value - x) <= cc.TOL["km/h"] for x in theirs):
                swapped.append({"driver": d, "value_kmh": value, "sentence": line.strip()[:200]})
    if swapped:
        return CheckResult("numbers_on_right_driver", "fail", 0.0,
                           f"{len(swapped)} speed(s) attached to a driver whose chunk gives it to someone else: "
                           + ", ".join(f"{s['driver']} {s['value_kmh']:g} km/h" for s in swapped), swapped)
    if not checked:
        return CheckResult("numbers_on_right_driver", "n/a", None, "No speed is directly attached to a named driver.")
    return CheckResult("numbers_on_right_driver", "pass", 1.0, f"{checked} driver-attached speed(s) all belong to that driver.")


# ── completeness ─────────────────────────────────────────────────────────────

_ORDINAL_CORNER = [(r"\b(?:first|opening)\s+(?:corner|turn)\b|\bturn\s+(?:1|one)\b", "first"),
                   (r"\b(?:last|final)\s+(?:corner|turn)\b", "last")]


def _corner_numbers(sources: list[dict], deltas: dict | None) -> list[int]:
    """Corner numbers of the whole lap. Only the full-lap comparison lists them all: the retrieved chunks do not."""
    ids = [s["section"] for s in (deltas or {}).get("sections", [])]
    nums = [int(n) for sid in ids for n in re.findall(r"(?<![\w-])T(\d+)", cc._norm(sid))]
    return sorted(set(nums))


def completeness(answer: str, sources: list[dict], deltas: dict | None = None, question: str | None = None) -> CheckResult:
    """Every driver the question names, and any corner it names, must be addressed in the answer."""
    if not question:
        return CheckResult("completeness", "n/a", None, "No question given.")
    asked = sorted(_drivers_in(question))
    missing_drivers = [d for d in asked if not cc._names_driver(answer, d)]
    wanted_corners = {int(n) for n in re.findall(r"(?<![\w-])T(\d+)\b", question)}
    wanted_corners |= {int(n) for n in re.findall(r"\bturn\s+(\d+)\b", question, re.I)}
    nums = _corner_numbers(sources, deltas)
    for pattern, which in _ORDINAL_CORNER:
        if re.search(pattern, question, re.I):
            if which == "first":
                wanted_corners.add(1)
            elif nums:
                wanted_corners.add(nums[-1])
    missing_corners = [n for n in sorted(wanted_corners)
                       if not re.search(rf"(?<![\w]){'T' + str(n)}(?!\d)|\bturn\s+{n}\b", cc._norm(answer), re.I)]
    if not asked and not wanted_corners:
        return CheckResult("completeness", "n/a", None, "The question names no driver or corner.")
    if missing_drivers or missing_corners:
        what = ([f"driver {', '.join(missing_drivers)}"] if missing_drivers else []) + \
               ([f"corner {', '.join('T' + str(n) for n in missing_corners)}"] if missing_corners else [])
        return CheckResult("completeness", "fail", 0.0, "The answer never addresses " + " and ".join(what) + ".",
                           {"drivers": missing_drivers, "corners": missing_corners})
    return CheckResult("completeness", "pass", 1.0, f"Addresses all named drivers ({len(asked)}) and corners ({len(wanted_corners)}).")


# ── false premise ────────────────────────────────────────────────────────────

PREMISES = {
    "a crash": r"\bcrash(?:ed|es|ing)?\b|\bspun\b|\bspin(?:ning)?\b|\bhit the (?:wall|barrier)",
    "a penalty": r"\bpenal(?:ty|ties|ise[sd]?|ize[sd]?)\b|\bdisqualif\w+|\bblack flag\b|\bgrid drop\b",
    "a mechanical failure": r"\bengine (?:failure|blew|blow|problem|trouble)\b|\bmechanical\b|\bgearbox\b|\bbrake failure\b|\bretire[d]?\b|\bdnf\b",
}
_CHALLENGE = re.compile(
    r"\bno (?:record|evidence|data|mention|indication|sign|information|crash|penalty|report)\b|"
    r"\b(?:does|do|did)\s*n[o']t (?:show|mention|indicate|contain|include|record|report|suggest|have)\b|"
    r"\bnothing (?:in|to suggest|indicates?)\b|\bthere (?:is|was|were) no\b|\bnot (?:a|any) (?:crash|penalty)\b|"
    r"\bno (?:such|evidence of)\b|\bcannot (?:confirm|find|see)\b|\bcan't (?:confirm|find|see)\b|"
    r"\bdid(?:n't| not) (?:crash|spin|retire|get a penalty|have a (?:crash|penalty))\b|\bthe premise\b|\bnot supported\b|"
    r"\b(?:isn't|is not|wasn't|was not) (?:in|supported|shown|recorded)\b", re.I)


def false_premise(answer: str, sources: list[dict], question: str | None = None) -> CheckResult:
    """If the question assumes a crash / penalty / failure that no retrieved chunk mentions, the answer must say so."""
    if not question:
        return CheckResult("false_premise", "n/a", None, "No question given.")
    ev = _source_text(sources)
    assumed = [name for name, pat in PREMISES.items()
               if re.search(pat, question, re.I) and not re.search(pat, ev, re.I)]
    if not assumed:
        return CheckResult("false_premise", "n/a", None, "The question assumes nothing the evidence lacks.")
    text = cc._norm(answer).replace("*", "")
    challenged = _CHALLENGE.search(text[:900])
    if challenged:
        return CheckResult("false_premise", "pass", 1.0,
                           f"The question assumes {', '.join(assumed)}, which the evidence never shows; the answer says so.")
    return CheckResult("false_premise", "fail", 0.0,
                       f"The question assumes {', '.join(assumed)}, which the evidence never shows, and the answer explains it anyway.",
                       assumed)


# ── overclaiming ─────────────────────────────────────────────────────────────

_ABSOLUTE = re.compile(r"\b(clearly|definitely|undoubtedly|obviously|certainly|conclusively|unquestionably|without (?:a )?doubt|"
                       r"proves?|proved|proving|undeniabl[ey]|beyond doubt)\b", re.I)


def overclaiming(answer: str, sources: list[dict], question: str | None = None) -> CheckResult:
    """No 'clearly' / 'definitely' / 'proves' about braking in a section whose chunk says it could not tell."""
    chunks = cc.uncertain_chunks(sources)
    if not chunks:
        return CheckResult("overclaiming", "n/a", None, "No retrieved chunk reports a data gap.")

    def is_braking_absolute(text: str) -> bool:
        return bool(_ABSOLUTE.search(text)) and bool(re.search(r"\bbrak\w*", text, re.I))

    hits = []
    for sec in cc.uncertain_sections(sources):
        for heading_refs, body in cc._parts(answer):
            if heading_refs is not None and sec not in heading_refs:
                continue
            if heading_refs is None and not cc._mentions(body, sec):
                continue
            last = heading_refs
            for sent in cc._sentences(body):
                refs = cc._SECTION_REF.findall(cc._norm(sent))
                if is_braking_absolute(sent) and not cc._HEDGE.search(sent) \
                        and cc._claim_belongs_to(sent, sec, default_refs=last, claim_test=is_braking_absolute):
                    hits.append({"section": sec, "sentence": sent.strip()[:220]})
                if refs:
                    last = refs
    if hits:
        return CheckResult("overclaiming", "fail", 0.0,
                           f"{len(hits)} absolute claim(s) about braking in a section the data could not resolve.", hits)
    return CheckResult("overclaiming", "pass", 1.0, "No absolute braking claim in a section with a data gap.")


_SELF_CORRECTION = re.compile(
    r"\bwait\b[,.:!\u2014-]|\bactually,? (?:no|that|the|it|he|she|i)\b|\bcorrection:|\bcorrected:|\blet me (?:re|double|check|correct)|"
    r"\.\.\.\s*(?:no|wait|actually|hmm)\b|\u2026\s*(?:no|wait|actually|hmm)\b|\bhmm\b|\bsorry,|\bi (?:mis|made a mistake)|\bscratch that\b|\bon second thought\b", re.I)


def self_correction(answer: str, sources: list[dict] | None = None) -> CheckResult:
    """The answer is a finished explanation: 'wait, corrected:' means the model changed its mind on the page."""
    hits = [m.group(0) for m in _SELF_CORRECTION.finditer(cc._norm(answer))]
    if hits:
        return CheckResult("self_correction", "fail", 0.0, f"The answer revises itself in the text: {', '.join(repr(h) for h in hits[:3])}", hits)
    return CheckResult("self_correction", "pass", 1.0, "No self-correction in the text.")


# ── team attribution ─────────────────────────────────────────────────────────

_CODE_TEAM = re.compile(r"\b([A-Z]{3})\s+(?:P\d+\s+)?\(([^)\n]+)\)")      # "RUS P1 (Mercedes)", "HAD (Red Bull Racing)"


def stored_teams(sources: list[dict]) -> dict[str, str]:
    teams: dict[str, str] = {}
    for s in sources:
        for code, team in _CODE_TEAM.findall(s.get("text") or ""):
            if any(key in team.lower() for key in TEAM_ALIASES):
                teams.setdefault(code, team.strip())
    return teams


def _team_key(team: str) -> str | None:
    low = team.lower()
    return next((key for key in TEAM_ALIASES if key in low), None)


def team_attribution(answer: str, sources: list[dict]) -> CheckResult:
    """'Haas's Hadjar' / 'Leclerc (McLaren)': the team named with a driver must be the driver's team in the evidence."""
    teams = stored_teams(sources)
    text = cc._norm(answer)
    checked, wrong = 0, []
    for code, team in teams.items():
        true_key = _team_key(team)
        if true_key is None:
            continue
        name = _name_alternatives(code)
        for key, alias in TEAM_ALIASES.items():
            patterns = [rf"(?:{alias})['\u2019]s\s+(?:\w+\s+)?{name}",                    # Haas's Hadjar
                        rf"{name}\s*\(\s*(?:{alias})",                                   # Hadjar (Haas)
                        rf"{name}\s*,?\s*(?:of|at|driving for|from)\s+(?:{alias})",       # Hadjar of Haas
                        rf"(?:{alias})\s+(?:driver|star|ace|rookie|man)\s+{name}"]        # Haas driver Hadjar
            for pattern in patterns:
                hit = re.search(pattern, text, re.IGNORECASE)
                if hit:
                    checked += 1
                    if key != true_key:
                        wrong.append({"driver": code, "said": key, "stored": team, "text": hit.group(0)[:60]})
    if wrong:
        return CheckResult("team_attribution", "fail", 0.0,
                           f"{len(wrong)} driver(s) attached to the wrong team: "
                           + ", ".join(f"{w['driver']} -> {w['said']} (stored: {w['stored']})" for w in wrong), wrong)
    if not checked:
        return CheckResult("team_attribution", "n/a", None, "No driver is named together with a team.")
    return CheckResult("team_attribution", "pass", 1.0, f"All {checked} driver-team pairing(s) match the evidence.")


# ── team ranking claims ──────────────────────────────────────────────────────

_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
             "eleventh": 11}
_ORD = "|".join(_ORDINALS) + r"|\d+(?:st|nd|rd|th)"
_RANK_CLAIM = re.compile(rf"\b({_ORD})[- ](?:quickest|fastest|best)\b[^.\n]{{0,25}}\b(?:team|constructor)", re.I)
_GRID_ROW = re.compile(r"^\s*P\d+\s+([A-Z]{3})\s+\(([^)\n]+)\)\s+(\d+):(\d+\.\d+)", re.M)


def team_ranking_from_grid(sources: list[dict]) -> dict[str, tuple[int, int]]:
    """{team key: (rank by its best car, rank by the average of its cars)} from the grid chunk, or {} without one."""
    times: dict[str, list[float]] = {}
    for src in sources:
        if src.get("chunk_type") != "grid":
            continue
        for _, team, minutes, seconds in _GRID_ROW.findall(src.get("text") or ""):
            key = _team_key(team)
            if key:
                times.setdefault(key, []).append(int(minutes) * 60 + float(seconds))
    if not times:
        return {}
    by_best = sorted(times, key=lambda k: min(times[k]))
    by_avg = sorted(times, key=lambda k: sum(times[k]) / len(times[k]))
    return {k: (by_best.index(k) + 1, by_avg.index(k) + 1) for k in times}


def team_ranking(answer: str, sources: list[dict]) -> CheckResult:
    """'Ferrari were the fourth quickest team': the rank must match the grid, by the best car or by the average of the cars."""
    ranks = team_ranking_from_grid(sources)
    if not ranks:
        return CheckResult("team_ranking", "n/a", None, "The evidence has no grid to rank the teams from.")
    checked, wrong = 0, []
    for sent in cc._sentences(cc._norm(answer)):
        claim = _RANK_CLAIM.search(sent)
        if not claim:
            continue
        word = claim.group(1).lower()
        n = _ORDINALS.get(word) or int(re.match(r"\d+", word).group())
        named = [k for k, alias in TEAM_ALIASES.items() if k in ranks and re.search(alias, sent, re.I)]
        if len(named) != 1:
            continue                                        # no team, or several: cannot tell which one the ordinal belongs to
        checked += 1
        if n not in ranks[named[0]]:
            wrong.append({"team": named[0], "said": n, "by_best_car": ranks[named[0]][0], "by_average": ranks[named[0]][1],
                          "sentence": sent.strip()[:200]})
    if wrong:
        return CheckResult("team_ranking", "fail", 0.0,
                           f"{len(wrong)} team ranking claim(s) do not match the grid: "
                           + ", ".join(f"{w['team']} said {w['said']}, grid gives {w['by_best_car']} (best car) / {w['by_average']} (average)" for w in wrong), wrong)
    if not checked:
        return CheckResult("team_ranking", "n/a", None, "The answer ranks no team.")
    return CheckResult("team_ranking", "pass", 1.0, f"{checked} team ranking claim(s) match the grid.")


# ── tyre claims ──────────────────────────────────────────────────────────────

_TYRE_LINE = re.compile(r"^Tyre:\s*([A-Za-z]+)(?:\s*\((\d+) laps old\))?(?:\s*\|\s*(Fresh|Used) set)?", re.M)
_TYRE_WORD = re.compile(r"\b(tyres?|tires?|compounds?|softs?|mediums?|hards?|fresh set|used set|tyre (?:age|wear|life)|rubber)\b", re.I)
_BLAME = re.compile(r"\b(because|due to|explain\w*|thanks to|down to|attribut\w*|result of|caused?|contribut\w*|helped|benefit\w*|"
                    r"advantage|accounts? for|reason|responsible)\b", re.I)
_TYRE_DENIAL = re.compile(r"\b(same|identical|equal|neither|both (?:on|used|ran|set)|no (?:tyre|tire|compound)|not (?:due|down|because|a tyre)|"
                          r"rules? out|does not (?:show|explain|support)|cannot (?:be )?attribut|unrelated|irrelevant)\b", re.I)


def tyres_in_evidence(sources: list[dict]) -> dict[str, tuple[str, str | None]]:
    """{driver: (compound, 'Fresh'/'Used'/None)} from one-driver lap summaries."""
    out = {}
    for src in sources:
        owners = src.get("drivers") or []
        m = _TYRE_LINE.search(src.get("text") or "")
        if m and len(owners) == 1:
            out[owners[0]] = (m.group(1).upper(), m.group(3))
    return out


def tyre_claims(answer: str, sources: list[dict]) -> CheckResult:
    """Fails when the answer blames tyre compound or age for a gap, but the evidence shows the same tyres (or says nothing)."""
    claims = [sent.strip() for sent in cc._sentences(cc._norm(answer))
              if _TYRE_WORD.search(sent) and _BLAME.search(sent) and not _TYRE_DENIAL.search(sent)
              and re.search(r"\b(gap|slower|faster|quicker|quickest|lost|gain\w*|pace|time|lap|difference)\b", sent, re.I)]
    if not claims:
        return CheckResult("tyre_claims", "n/a", None, "The answer does not blame the tyres for anything.")
    tyres = tyres_in_evidence(sources)
    if len(tyres) >= 2 and len(set(tyres.values())) == 1:
        compound, fresh = next(iter(tyres.values()))
        return CheckResult("tyre_claims", "fail", 0.0,
                           f"The answer blames the tyres, but every driver in the evidence ran the same ({compound}{', ' + fresh.lower() + ' set' if fresh else ''}).",
                           [{"sentence": c[:200]} for c in claims])
    if not tyres:
        return CheckResult("tyre_claims", "fail", 0.0, "The answer blames the tyres, but the evidence has no tyre information.",
                           [{"sentence": c[:200]} for c in claims])
    return CheckResult("tyre_claims", "pass", 1.0, "The evidence shows a tyre difference between the drivers.")


# ── starting-grid claims ─────────────────────────────────────────────────────

_STARTED = re.compile(r"\bstart(?:ed|s)\s+(?:the (?:race|grand prix|gp)\s+)?(?:from|in|on|at)\s+(?:P\d+|\d+(?:st|nd|rd|th)|pole|the front|the back)\b|"
                      r"\b(?:lined|lines) up\s+(?:in|on|at)\s+(?:P\d+|\d+(?:st|nd|rd|th))\b|\bstarting (?:grid )?(?:position|slot)\s+(?:was|of)\s+P?\d+", re.I)


def grid_start_claim(answer: str, sources: list[dict]) -> CheckResult:
    """The data is the qualifying classification. 'Started from P17' claims the race grid, which penalties can change."""
    hits = [m.group(0) for m in _STARTED.finditer(cc._norm(answer))]
    if hits:
        return CheckResult("grid_start_claim", "fail", 0.0,
                           "The answer says where a driver started the race, but the data is the qualifying classification (grid penalties are not in it): "
                           + ", ".join(repr(h) for h in hits[:3]), hits)
    return CheckResult("grid_start_claim", "pass", 1.0, "No claim about the race starting position.")


def run_all(answer: str, sources: list[dict], deltas: dict | None = None, question: str | None = None,
            session: str | None = None) -> list[CheckResult]:
    return [reversed_winner(answer, sources, deltas),
            self_correction(answer, sources),
            wrong_driver_or_session(answer, sources, question, session),
            numbers_on_right_driver(answer, sources),
            team_attribution(answer, sources),
            team_ranking(answer, sources),
            tyre_claims(answer, sources),
            grid_start_claim(answer, sources),
            completeness(answer, sources, deltas, question),
            false_premise(answer, sources, question),
            overclaiming(answer, sources, question)]
