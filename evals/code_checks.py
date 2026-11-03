"""
code_checks.py
--------------
Deterministic checks on a RAG answer. Each takes the answer text and the evidence the
model was given, and returns a CheckResult. No LLM and no network: the same input always
gives the same verdict, so they are cheap to run on every trace.

  uncertainty_compliance : did the answer claim something a chunk said it could not tell?
  number_grounding       : does every number in the answer come from the chunks?
  total_consistency      : does a claimed total equal the sum of the sections it covers?
  coverage               : were the sections with the biggest time gaps retrieved / discussed?

`sources` is the list the API returns: [{"section", "text", "drivers", "chunk_type"}, ...].
`deltas` is the API's lap comparison: {"lap_gap_s", "sections": [{"section", "gain_s", "retrieved"}]}.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    name: str
    label: str                  # "pass" | "fail" | "n/a"
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
    r"no firm|gap in|overlap|at least|between \d+\s*m|unknown|not possible|couldn't|could not",
    re.I,
)


def _sentences(text: str) -> list[str]:
    out = []
    for line in re.split(r"\n+", text):
        out.extend(s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(*#])", line) if s.strip())
    return out


def _mentions(sentence_or_block: str, section_id: str) -> bool:
    sid = re.escape(_norm(section_id))
    return re.search(rf"(?<![\w-]){sid}(?![\w-])", _norm(sentence_or_block)) is not None


def section_mentioned(answer: str, section_id: str) -> bool:
    """Is the section discussed? Accepts the exact id, and the ways answers refer to straights."""
    if _mentions(answer, section_id):
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


def uncertain_sections(sources: list[dict]) -> list[str]:
    seen = []
    for s in sources:
        text = (s.get("text") or "").lower()
        if s.get("section") and any(m in text for m in UNCERTAIN_MARKERS) and s["section"] not in seen:
            seen.append(s["section"])
    return seen


def uncertainty_compliance(answer: str, sources: list[dict], deltas: dict | None = None) -> CheckResult:
    """Fail if the answer states, without hedging, who braked earlier/later in a section whose chunk says it could not tell."""
    flagged = uncertain_sections(sources)
    if not flagged:
        return CheckResult("uncertainty_compliance", "n/a", None, "No retrieved chunk reports a data gap.")
    violations = []
    for sec in flagged:
        for block in _blocks(answer):
            if not _mentions(block, sec):
                continue
            for sent in _sentences(block):
                if _FIRM_BRAKING.search(sent) and not _HEDGE.search(sent) and _mentions(block, sec):
                    violations.append({"section": sec, "sentence": sent.strip()})
    if violations:
        return CheckResult("uncertainty_compliance", "fail", 0.0,
                           f"{len(violations)} unhedged braking claim(s) in section(s) the data could not resolve.",
                           violations)
    return CheckResult("uncertainty_compliance", "pass", 1.0,
                       f"{len(flagged)} section(s) had a data gap; no unhedged braking claim about them.", flagged)


# ── 2. Number grounding ──────────────────────────────────────────────────────

_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(km/h|s|m)(?![\w/])")
TOL = {"s": 0.0015, "km/h": 0.06, "m": 0.6}
MAX_SUM_TERMS = {"s": 4, "km/h": 2, "m": 2}


def numbers_with_units(text: str) -> list[tuple[float, str]]:
    t = _norm(text)
    out = []
    for m in _NUM.finditer(t):
        value, unit = float(m.group(1)), m.group(2)
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
    ungrounded, how = [], {}
    for value, unit in claimed:
        found = _derivable(value, unit, pool[unit])
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

_TOTAL_WORDS = re.compile(r"\b(total|net|combined|overall|altogether|together|in sum|cumulative)\b", re.I)


def total_consistency(answer: str, sources: list[dict], deltas: dict | None = None) -> CheckResult:
    """A claimed total in seconds must equal the net of all discussed sections, the sum of one driver's gains, or the lap gap."""
    if not deltas or not deltas.get("sections"):
        return CheckResult("total_consistency", "n/a", None, "No section deltas to recompute a total from.")
    claims = []
    for sent in _sentences(answer):
        if _TOTAL_WORDS.search(sent):
            claims += [(v, sent.strip()) for v, u in numbers_with_units(sent) if u == "s"]
    if not claims:
        return CheckResult("total_consistency", "n/a", None, "The answer states no total in seconds.")
    mentioned = [(s["section"], float(s["gain_s"])) for s in deltas["sections"] if section_mentioned(answer, s["section"])]
    gains = [g for _, g in mentioned][:14]
    lap_gap = abs(float(deltas.get("lap_gap_s") or 0))
    bad = []
    for value, sent in claims:
        ok = abs(value - lap_gap) <= TOL["s"] or abs(abs(sum(gains)) - value) <= TOL["s"]      # lap gap, or the net of all
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
    mentioned = [n for n in names if section_mentioned(answer, n)]
    def result(name, hit, what):
        frac = len(hit) / len(names)
        label = "pass" if frac >= 2 / 3 else "fail"
        missed = [n for n in names if n not in hit]
        return CheckResult(name, label, round(frac, 3),
                           f"{len(hit)}/{len(names)} of the biggest-gap sections {what}" + (f"; missed: {', '.join(missed)}" if missed else "."),
                           {"top_sections": names, "hit": hit})
    return [result("retrieval_coverage", retrieved, "were retrieved"),
            result("answer_coverage", mentioned, "are discussed in the answer")]


def run_all(answer: str, sources: list[dict], deltas: dict | None = None) -> list[CheckResult]:
    return [uncertainty_compliance(answer, sources, deltas),
            number_grounding(answer, sources, deltas),
            total_consistency(answer, sources, deltas),
            *coverage(answer, sources, deltas)]
