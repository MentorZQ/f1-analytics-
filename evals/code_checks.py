"""
code_checks.py
--------------
Deterministic checks on a RAG answer. Each takes the answer text and the evidence the
model was given, and returns a CheckResult. No LLM and no network: the same input always
gives the same verdict, so they are cheap to run on every trace.

  uncertainty_compliance : did the answer claim something a chunk said it could not tell?
  number_grounding       : does every number in the answer come from the chunks?
  total_consistency      : does a claimed total equal the sum of the sections it covers?
  inference_rate         : how much of the answer is interpretation (reported, not pass/fail)
  coverage               : were the sections with the biggest time gaps retrieved / discussed?

`sources` is the list the API returns: [{"section", "text", "drivers", "chunk_type"}, ...].
`deltas` is the API's lap comparison: {"lap_gap_s", "sections": [{"section", "gain_s", "retrieved"}]}.
"""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field


_RAG = Path(__file__).resolve().parent.parent / "RAG_data_layers" / "F1 Project Thoughts"
if str(_RAG) not in sys.path:
    sys.path.insert(0, str(_RAG))
from rag_retrieval import DRIVER_SURNAMES  # noqa: E402


@dataclass
class CheckResult:
    name: str
    label: str                  # "pass" | "fail" | "info" (a measurement) | "n/a"
    score: float | None         # 1.0 pass, 0.0 fail, or a fraction; None when n/a
    explanation: str
    details: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "score": self.score,
                "explanation": self.explanation, "details": self.details}


def _norm(text: str) -> str:
    return text.replace("–", "-").replace("—", "-").replace("−", "-").replace("**", "")


# ── 1. Uncertainty compliance ────────────────────────────────────────────────

UNCERTAIN_MARKERS = ("cannot tell", "gap in the car data", "gap in car data")

# A firm statement about who braked / lifted earlier or later.
_FIRM_BRAKING = re.compile(
    r"brak\w*\s+\d+(?:\.\d+)?\s*m\s+(?:later|earlier)"
    r"|(?:later|earlier|more aggressive)\s+(?:braking|brake)"
    r"|brak\w*\s+(?:later|earlier)"
    r"|(?:later|earlier)\s+brak\w*\s+point",
    re.I,
)
_HEDGE = re.compile(
    r"cannot|can't|can not|unable|uncertain|unreliable|inconclusive|unclear|not clear|"
    r"no firm|gap in|has a gap|have a gap|does not (?:directly |clearly |fully )?(?:give|support|show|establish|settle|resolve)|"
    r"not (?:a )?clean|not sufficient|insufficient|not enough|overlap|at least|between \d+\s*m|unknown|not possible|couldn't|could not|caution|treat with care|approximate|tentative",
    re.I,
)


_INDIRECT_QUESTION = re.compile(
    r"\b(?:determine|say|tell|know|establish|decide|identify|clear|question|of|about|on|to|whether)\s+"
    r"(?:definitively\s+|conclusively\s+|confidently\s+)?(?:who|which driver)\s+brak", re.I)


def _sentences(text: str) -> list[str]:
    out = []
    for line in re.split(r"\n+", text):
        out.extend(s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(*#])", line) if s.strip())
    return out


def _mentions(sentence_or_block: str, section_id: str) -> bool:
    sid = re.escape(_norm(section_id))
    return re.search(rf"(?<![\w-]){sid}(?![\w-])", _norm(sentence_or_block)) is not None


def section_mentioned(answer: str, section_id: str, start_m: float | None = None) -> bool:
    """Is the section discussed? Accepts the exact id, its distance window, and the ways answers refer to straights."""
    if _mentions(answer, section_id):
        return True
    if start_m is not None and re.search(rf"(?<![\d.,]){int(round(start_m))}\s*m\b", _norm(answer).replace(",", "")):
        return True
    a, sid = _norm(answer).lower(), _norm(section_id).lower()
    if sid.startswith("main straight"):
        if "main straight" not in a and "start/finish" not in a and "finish straight" not in a and "start straight" not in a:
            return False
        if "(start" in sid:
            return bool(re.search(r"main straight[^\n]{0,40}(start|before t1|opening)|(start|opening)[^\n]{0,20}straight", a))
        return bool(re.search(r"main straight[^\n]{0,40}(finish|final|after)|(finish|final)[^\n]{0,20}straight", a))
    if sid.startswith("between "):
        pair = sid[len("between "):]
        return bool(re.search(rf"(?<![\w-]){re.escape(pair)}(?![\w-])[^\n]{{0,25}}straight|straight[^\n]{{0,25}}(?<![\w-]){re.escape(pair)}(?![\w-])", a)) or \
            bool(re.search(rf"between {re.escape(pair)}", a))
    return False


def _blocks(answer: str) -> list[str]:
    """Answer split at headings and blank lines: a block is about one thing."""
    return [b for b in re.split(r"\n\s*\n|\n(?=#)", answer) if b.strip()]


_SECTION_REF = re.compile(r"(?<![\w-])T\d+[A-Za-z]?(?:-T?\d+[A-Za-z]?)*(?![\w-])")
_CLAUSE_SPLIT = re.compile(r",|;|\bwhile\b|\bbut\b|\bwhereas\b|\band\b|\bthen\b|\bwith\b", re.I)


_HEADING = re.compile(r"^\s*(?:#{1,6}\s+|\*\*[^*\n]+\*\*\s*$)")


def _parts(answer: str) -> list[tuple[list[str] | None, str]]:
    """
    The answer cut at headings (markdown '#', or a line that is only bold). Each part is
    (the sections its heading names, or None if the heading names none, its body).
    """
    parts, head, buf = [], None, []

    def flush():
        if buf:
            parts.append((head, "\n".join(buf)))

    for line in answer.split("\n"):
        if _HEADING.match(line):
            refs = _SECTION_REF.findall(_norm(line))
            if not refs and not line.lstrip().startswith("#"):
                buf.append(line)                   # "**LAW faster by 0.139s**": a bold line, not a new section
                continue
            flush()
            head, buf = (refs or None), []
        else:
            buf.append(line)
    flush()
    return parts


def _claim_belongs_to(sentence: str, section_id: str, default_refs: list[str] | None = None, claim_test=None) -> bool:
    """
    Is the firm braking claim in this sentence about `section_id`? A sentence can talk about
    several sections ("faster at T1-T2 on braking, and at T4 on apex speed"), so each clause is
    attributed to the section it names, or to the last section named before it.
    """
    claim_test = claim_test or (lambda clause: bool(_FIRM_BRAKING.search(clause)))
    current: list[str] | None = default_refs
    seen: list[str] = []                                # every section named so far in this sentence
    for clause in _CLAUSE_SPLIT.split(_norm(sentence)):
        refs = _SECTION_REF.findall(clause)
        if refs:
            current = refs
            seen += refs
        if claim_test(clause):
            if refs:                                     # the clause names its own section(s)
                if section_id in [_norm(r) for r in refs]:
                    return True
            elif seen:                                   # "...through T1-T2, T4 and T10-T11, where his later braking...":
                if section_id in [_norm(r) for r in seen]:      # the claim covers the sections listed before it
                    return True
            elif current is None or section_id in [_norm(r) for r in current]:
                return True
    return False


SURNAMES = DRIVER_SURNAMES


def _names_driver(text: str, code: str) -> bool:
    t = _norm(text).lower().replace("\u00fc", "u").replace("\u00e9", "e")
    return bool(re.search(rf"\b{code.lower()}\b", t)) or SURNAMES.get(code, "\0") in t


def _about_drivers(sentence: str, drivers: list[str]) -> bool:
    """Does the sentence talk about this chunk's drivers? Every driver of a two-driver chunk, or the one driver."""
    return bool(drivers) and all(_names_driver(sentence, d) for d in drivers)


def uncertain_chunks(sources: list[dict]) -> list[dict]:
    out = []
    for s in sources:
        text = (s.get("text") or "").lower()
        if s.get("section") and any(m in text for m in UNCERTAIN_MARKERS):
            out.append({"section": s["section"], "drivers": list(s.get("drivers") or [])})
    return out


def uncertain_sections(sources: list[dict]) -> list[str]:
    seen = []
    for c in uncertain_chunks(sources):
        if c["section"] not in seen:
            seen.append(c["section"])
    return seen


def uncertainty_compliance(answer: str, sources: list[dict], deltas: dict | None = None) -> CheckResult:
    """Fail if the answer states, without hedging, who braked earlier/later in a section whose chunk says it could not tell."""
    chunks = uncertain_chunks(sources)
    flagged = uncertain_sections(sources)
    if not flagged:
        return CheckResult("uncertainty_compliance", "n/a", None, "No retrieved chunk reports a data gap.")
    violations = []
    for sec in flagged:
        for heading_refs, body in _parts(answer):
            if heading_refs is not None and sec not in heading_refs:
                continue                                  # this part is headed by other sections
            if heading_refs is None and not _mentions(body, sec):
                continue
            last_refs = heading_refs                       # the section the text is currently about
            for sent in _sentences(body):
                sent_refs = _SECTION_REF.findall(_norm(sent))
                firm = _FIRM_BRAKING.search(sent) and not _HEDGE.search(sent) and not _INDIRECT_QUESTION.search(sent)
                if firm:
                    about = [c for c in chunks if c["section"] == sec]
                    about_these_drivers = any(_about_drivers(sent, c["drivers"]) or not c["drivers"] for c in about) \
                        or any(_about_drivers(body, c["drivers"]) for c in about)
                    # a sentence that names no section belongs to the one named just before it
                    if about_these_drivers and _claim_belongs_to(sent, sec, default_refs=last_refs):
                        violations.append({"section": sec, "sentence": sent.strip()})
                if sent_refs:
                    last_refs = sent_refs
    if violations:
        return CheckResult("uncertainty_compliance", "fail", 0.0,
                           f"{len(violations)} unhedged braking claim(s) in section(s) the data could not resolve.",
                           violations)
    return CheckResult("uncertainty_compliance", "pass", 1.0,
                       f"{len(flagged)} section(s) had a data gap; no unhedged braking claim about them.", flagged)


# ── 2. Number grounding ──────────────────────────────────────────────────────

_NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(km/h|s|m)(?![\w/])")
TOL = {"s": 0.0015, "km/h": 0.06, "m": 0.6}
MAX_SUM_TERMS = {"s": 4, "km/h": 2, "m": 2}


_MIN_SEC = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2}(?:\.\d+)?)(?:\s*s\b)?(?![\d:])")


def numbers_with_units(text: str) -> list[tuple[float, str]]:
    # 1:19.911 is a lap time in seconds even when no "s" follows (chunks write "Lap 1: 1:19.911")
    t = _MIN_SEC.sub(lambda m: f"{int(m.group(1)) * 60 + float(m.group(2)):.3f}s", _norm(text))
    out = []
    for m in _NUM.finditer(t):
        value, unit = float(m.group(1).replace(",", "")), m.group(2)
        if unit == "s" and "." not in m.group(1):          # "1s", "2s": prose, not a time
            continue
        if unit == "m" and re.match(r"\s*(?:in|ph|illi)", t[m.end():m.end() + 4]):
            continue
        out.append((value, unit))
    return out


def _derivable(value: float, unit: str, pool: list[float]) -> str | None:
    tol = TOL[unit]
    pool = sorted(set(round(p, 3) for p in pool))
    for p in pool:
        if abs(p - value) <= tol:
            return "stated"
    if unit == "s" and abs(value * 10 - round(value * 10)) < 1e-6:        # "about 0.2s" for a stated 0.186s
        if any(abs(p - value) <= 0.05 + 1e-9 for p in pool):
            return "rounded"                                              # acceptable only if the exact figure is shown too
    for a, b in itertools.combinations(pool, 2):
        if abs(abs(a - b) - value) <= tol:
            return "difference"
    for k in range(2, MAX_SUM_TERMS[unit] + 1):
        if k > len(pool) or len(pool) > 80 and k > 2:
            continue
        for combo in itertools.combinations(pool, k):
            if abs(sum(combo) - value) <= tol:
                return f"sum of {k}"
    return None


def number_grounding(answer: str, sources: list[dict], deltas: dict | None = None) -> CheckResult:
    """Every number with a unit (km/h, s, m) must be in the chunks, or a difference / sum of numbers that are."""
    pool: dict[str, list[float]] = {"s": [], "km/h": [], "m": []}
    for s in sources:
        for v, u in numbers_with_units(s.get("text") or ""):
            pool[u].append(v)
    if deltas:                                                     # the lap comparison is also evidence
        pool["s"].append(abs(float(deltas.get("lap_gap_s") or 0)))
        for sec in deltas.get("sections", []):
            pool["s"].append(abs(float(sec["gain_s"])))
            pool["m"] += [float(sec.get("start_m", 0)), float(sec.get("end_m", 0))]
    claimed = list(dict.fromkeys(numbers_with_units(answer)))
    if not claimed:
        return CheckResult("number_grounding", "n/a", None, "The answer contains no numbers with units.")
    sentence_numbers = [numbers_with_units(sent) for sent in _sentences(answer)]
    ungrounded, how = [], {}
    for value, unit in claimed:
        found = _derivable(value, unit, pool[unit])
        if found == "rounded":
            exact = [p for p in pool[unit] if abs(p - value) <= 0.05 + 1e-9]
            shown = any((value, unit) in nums and any(u == unit and abs(n - e) <= TOL[unit] for n, u in nums for e in exact)
                        for nums in sentence_numbers)
            if not shown:
                ungrounded.append(f"{value:g}{unit} (rounded, exact figure not shown)")
                continue
            found = "rounded, exact shown"
        if found is None:
            ungrounded.append(f"{value:g}{unit}")
        else:
            how[found] = how.get(found, 0) + 1
    score = 1 - len(ungrounded) / len(claimed)
    if ungrounded:
        return CheckResult("number_grounding", "fail", round(score, 3),
                           f"{len(ungrounded)} of {len(claimed)} numbers are not in the evidence: {', '.join(ungrounded)}",
                           ungrounded)
    return CheckResult("number_grounding", "pass", 1.0,
                       f"All {len(claimed)} numbers are grounded ({how}).", how)


# ── 2b. Totals ───────────────────────────────────────────────────────────────
# A total can be wrong even when its digits appear in the evidence (0.122s is also one section's
# gain). So a claimed total is recomputed from the section gains of the sections the answer discusses.

LAP_LEVEL_CHUNKS = {"driver_lap_summary", "head_to_head", "driver_sector", "session_pace_evolution", "lap_comparison",
                    "grid", "team_summary", "team_advantage"}
_STRONG_TOTAL = re.compile(r"\b(total|net|combined(?! with)|altogether|in sum)\b|across (?:these|all|the|those)\b[^.]*\bsections\b", re.I)
_WEAK_TOTAL = re.compile(r"\b(overall|together)\b", re.I)


MAX_TOTAL_S = 1.0
_SINGLE_SECTION_CUE = re.compile(r"\b(?:largest|biggest|greatest|single|one section|that section|this section|that one|window|sequence|complex)\b", re.I)


def _next_to_section_ref(sentence: str, value: float) -> bool:
    """Is this number written right next to a single section name ('0.092s at T1-T2', 'T4: 0.122s')?"""
    t = _norm(sentence)
    for m in _NUM.finditer(t):
        if m.group(2) != "s" or abs(float(m.group(1).replace(",", "")) - value) > 1e-9:
            continue
        near = t[max(0, m.start() - 22): m.end() + 22]
        if _SECTION_REF.search(near) or re.search(r"\b(?:straight|corner|chicane|complex|section|sequence|hairpin|esses)\b", near, re.I):
            return True
    return False


_SUM_EXPRESSION = re.compile(r"((?:\d+\.\d+)\s*s?(?:\s*\+\s*(?:\d+\.\d+)\s*s?)+)\s*=\s*(\d+\.\d+)\s*s?")


def total_consistency(answer: str, sources: list[dict], deltas: dict | None = None) -> CheckResult:
    """A claimed total in seconds must equal the net of all discussed sections, the sum of one driver's gains, or the lap gap."""
    if not deltas or not deltas.get("sections"):
        return CheckResult("total_consistency", "n/a", None, "No section deltas to recompute a total from.")
    claims = []
    wrong_sums = []
    for sent in _sentences(answer):
        arithmetic = _SUM_EXPRESSION.search(sent)
        if arithmetic:                                     # "0.086s + 0.103s = 0.189s": check the arithmetic, not the figures as totals
            terms = [float(x) for x in re.findall(r"\d+\.\d+", arithmetic.group(1))]
            if abs(sum(terms) - float(arithmetic.group(2))) > TOL["s"] * len(terms):
                wrong_sums.append({"sentence": sent.strip()[:200], "terms_sum": round(sum(terms), 3), "stated": float(arithmetic.group(2))})
            continue
        if re.search(r"\bnot (?:a |the )?(?:full |complete )?total\b|\bdo(?:es)? not (?:add|sum)\b|\bcannot be reconciled\b|\bnot a complete accounting\b", sent, re.I):
            continue                                       # the answer itself says these figures are not a total
        if re.search(r"\d\.\d+\s*s?\s*\+\s*\d\.\d+", sent):
            continue                                       # a sum written out without a result is a list of terms, not a claimed total
        if re.search(r"\bsectors?\b", sent, re.I):         # sector figures are not in the section list this check recomputes from
            continue
        strong = bool(_STRONG_TOTAL.search(sent))
        if strong or _WEAK_TOTAL.search(sent):
            claims += [(v, sent.strip(), strong) for v, u in numbers_with_units(sent) if u == "s"]
    if wrong_sums:
        return CheckResult("total_consistency", "fail", 0.0,
                           f"{len(wrong_sums)} written-out sum(s) do not add up: the terms give {wrong_sums[0]['terms_sum']:g}s, the answer says {wrong_sums[0]['stated']:g}s",
                           wrong_sums)
    if not claims:
        return CheckResult("total_consistency", "n/a", None, "The answer states no total in seconds.")
    mentioned = [(s["section"], float(s["gain_s"])) for s in deltas["sections"]
                 if section_mentioned(answer, s["section"], s.get("start_m"))]
    gains = [g for _, g in mentioned][:14]
    lap_level = [v for src in sources if src.get("chunk_type") in LAP_LEVEL_CHUNKS
                 for v, u in numbers_with_units(src.get("text") or "") if u == "s"]
    lap_gap = abs(float(deltas.get("lap_gap_s") or 0))
    bad = []
    single = [abs(g) for g in gains]
    for value, sent, strong in claims:
        if value >= MAX_TOTAL_S:                           # section and lap times (6.442s, 80.4s) are not gap totals
            continue
        if _SINGLE_SECTION_CUE.search(sent) and any(abs(value - g) <= TOL["s"] for g in single):
            continue                                       # "the largest margin ... 0.456s": one section's own gain
        if _next_to_section_ref(sent, value):              # "gained 0.092s at T1-T2": one section's gain, stated as such
            continue
        # a weak word ("overall") often means "through this section overall": one section's own gain is fine
        ok = (not strong and any(abs(value - g) <= TOL["s"] for g in single)) \
            or abs(value - lap_gap) <= TOL["s"] or abs(abs(sum(gains)) - value) <= TOL["s"] \
            or any(abs(value - v) <= TOL["s"] for v in lap_level)           # lap gap, the net of all, or a stated lap figure
        for side in ([g for g in gains if g > 0], [g for g in gains if g < 0]):                 # or one driver's gains
            for k in range(2, len(side) + 1):
                if ok:
                    break
                ok = any(abs(abs(sum(c)) - value) <= TOL["s"] for c in itertools.combinations(side, k))
        if not ok:
            bad.append({"claimed_s": value, "sentence": sent[:200]})
    if bad:
        return CheckResult("total_consistency", "fail", 0.0,
                           f"{len(bad)} total(s) do not match the net, or either driver's total, over the {len(gains)} discussed sections: "
                           + ", ".join(f"{b['claimed_s']:g}s" for b in bad), bad)
    return CheckResult("total_consistency", "pass", 1.0, f"{len(claims)} total(s) match a sum of discussed sections.")


# ── 2c. Inference rate ───────────────────────────────────────────────────────
# Not a pass/fail check: it measures how much of the answer is interpretation rather than quoted data.

_CAUSAL = re.compile(
    r"\b(because|despite|although|however|which (?:explains|means|suggests|allowed|gave)|"
    r"(?:translat\w+|convert\w+|carry|carries|carried over|feeds?|fed|fell|led to|result\w* in|"
    r"explains?|suggest\w*|indicat\w*|impl(?:y|ies|ied)|reflect\w*|due to|thanks to|as a result|"
    r"consequence|compromis\w+|platform for|allowing|enabled?))\b", re.I)


def inference_rate(answer: str, sources: list[dict] | None = None, deltas: dict | None = None) -> CheckResult:
    """Share of sentences using causal or interpretive wording. Reported, not judged."""
    sents = [s for s in _sentences(answer) if len(s.split()) >= 5 and not s.lstrip().startswith("|")]
    if not sents:
        return CheckResult("inference_rate", "n/a", None, "No prose sentences.")
    hits = [s.strip() for s in sents if _CAUSAL.search(s)]
    rate = len(hits) / len(sents)
    return CheckResult("inference_rate", "info", round(rate, 3),
                       f"{len(hits)} of {len(sents)} sentences use causal or interpretive wording.", hits)


# ── 3. Coverage of the biggest time gaps ─────────────────────────────────────

TOP_N = 3


def coverage(answer: str, sources: list[dict], deltas: dict | None, top_n: int = TOP_N) -> list[CheckResult]:
    """The sections where the pair differs most should be retrieved, and the answer should talk about them."""
    if not deltas or not deltas.get("sections"):
        return [CheckResult("retrieval_coverage", "n/a", None, "No driver-pair comparison for this question."),
                CheckResult("answer_coverage", "n/a", None, "No driver-pair comparison for this question.")]
    top = sorted(deltas["sections"], key=lambda s: -abs(s["gain_s"]))[:top_n]
    names = [s["section"] for s in top]
    retrieved = [s["section"] for s in top if s.get("retrieved")]
    starts = {s["section"]: s.get("start_m") for s in top}
    mentioned = [n for n in names if section_mentioned(answer, n, starts.get(n))]
    def result(name, hit, what):
        frac = len(hit) / len(names)
        label = "pass" if frac >= 2 / 3 else "fail"
        missed = [n for n in names if n not in hit]
        return CheckResult(name, label, round(frac, 3),
                           f"{len(hit)}/{len(names)} of the biggest-gap sections {what}" + (f"; missed: {', '.join(missed)}" if missed else "."),
                           {"top_sections": names, "hit": hit})
    return [result("retrieval_coverage", retrieved, "were retrieved"),
            result("answer_coverage", mentioned, "are discussed in the answer")]


def run_all(answer: str, sources: list[dict], deltas: dict | None = None, question: str | None = None,
            session: str | None = None) -> list[CheckResult]:
    from evals import claim_checks                       # imported here: claim_checks builds on this module
    return [uncertainty_compliance(answer, sources, deltas),
            number_grounding(answer, sources, deltas),
            total_consistency(answer, sources, deltas),
            inference_rate(answer, sources, deltas),
            *coverage(answer, sources, deltas),
            *claim_checks.run_all(answer, sources, deltas, question, session)]
