"""The Validator: plain code, no LLM. An LLM judging numbers can make the same mistakes it is
meant to catch, so the checks are deterministic and unit-tested.

Three checks per report:
  (a) numbers: every number in the text must match a value in the fact sheet, within rounding
      (half a unit of the last digit written, or 0.5% for numbers >= 100: "2,100 ha" may stand
      for 2,104 ha, "2,000 ha" may not). Numbers with a unit (ha, %, km, ...) are always checked;
      bare integers 0-10 are allowed as ordinary counting words ("3 reports").
  (b) caveats: each required caveat must appear, i.e. its key ideas must co-occur in one
      paragraph or bullet (e.g. "paddy" + "seasonal" + "ambiguous").
  (c) citations: every sentence of the Method section (8+ words) cites an SOP chunk like [S3],
      and every cited id exists.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

CAVEATS = {
    "paddy_ambiguity": (
        "standing water in rice paddies is ambiguous: it may be seasonal (deliberate) paddy "
        "flooding rather than flood damage",
        [r"padd(y|ies)|\brice\b", r"seasonal|deliberate|on purpose|intentional|farming|cultivat",
         r"ambigu|cannot|can ?not|can't|not possible|confound|uncertain|unclear|indisting"],
    ),
    "ndvi_proxy": (
        "NDVI drop is a proxy for mud or damage, not proof",
        [r"ndvi|vegetation loss", r"proxy|not proof|not prove|cannot prove|indirect|not confirm"
         r"|does not confirm|not direct"],
    ),
    "aoi_coverage": (
        "the AOI does not cover the whole affected area",
        [r"\baoi\b|area of interest|study area|analysis area|coverage",
         r"outside|not cover|does not (include|cover)|only (part|partial)|partial|limit"],
    ),
    "port_excluded": (
        "Port of Valencia change is excluded from the urban-change total",
        [r"\bport\b|harbou?r", r"exclud|separate|not (counted|included)|apart|left out"],
    ),
}  # fmt: skip

UNIT = r"\s*(%|percent\b|ha\b|hectares?\b|km\b|m\b|days?\b|towns?\b|scenes?\b)"
NUMBER = re.compile(r"(?<![\w.\-/])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w.]\d|[A-Za-z])")
CITATION = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


@dataclass
class Found:
    value: float
    decimals: int
    text: str
    has_unit: bool
    context: str


def fact_numbers(facts) -> list[float]:
    """Every number in the fact sheet: numeric leaves, numbers inside strings (dates,
    definitions), and fractions in [0, 1] also as percentages (0.989 -> 98.9)."""
    out = []

    def walk(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, int | float):
            out.append(float(x))
            if 0 <= x <= 1:
                out.append(float(x) * 100)
        elif isinstance(x, str):
            out.extend(float(m) for m in re.findall(r"\d+(?:\.\d+)?", x))
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, list | tuple):
            for v in x:
                walk(v)

    walk(facts)
    return out


def numbers_in(text: str) -> list[Found]:
    text = CITATION.sub("", text)
    found = []
    for m in NUMBER.finditer(text):
        whole, frac = m.group(1), m.group(2) or ""
        after = text[m.end() : m.end() + 12]
        context = text[max(0, m.start() - 50) : m.end() + 20].replace("\n", " ")
        value = float(whole.replace(",", "") + frac)
        found.append(Found(value, max(len(frac) - 1, 0), m.group(0),
                           bool(re.match(UNIT, after)), context))  # fmt: skip
    return found


def matches(f: Found, allowed: list[float]) -> bool:
    half_unit = 0.5 * 10 ** (-f.decimals)
    for v in allowed:
        if abs(f.value - v) <= half_unit + 1e-9:
            return True
        if f.value >= 100 and math.isclose(f.value, v, rel_tol=0.005):
            return True
    return False


def check_numbers(report: str, facts: dict) -> list[str]:
    allowed = fact_numbers(facts)
    errors = []
    for f in numbers_in(report):
        if not f.has_unit and f.decimals == 0 and f.value <= 10:
            continue  # "3 reports", "Phase 2", list numbering
        if not matches(f, allowed):
            errors.append(f"number {f.text!r} (in: ...{f.context.strip()}...) does not match any "
                          "value in the fact sheet; use the exact figure or remove it")  # fmt: skip
    return errors


def units(report: str) -> list[str]:
    """Paragraphs and bullet items (with their wrapped lines): where a caveat's ideas must
    co-occur. A blank line, a heading or a new bullet starts a new unit."""
    out, block = [], []
    for line in report.splitlines():
        stripped = line.strip()
        starts_new = not stripped or stripped.startswith("#")
        starts_new |= bool(re.match(r"^([-*+]|\d+[.)])\s", stripped))
        if starts_new and block:
            out.append(" ".join(block))
            block = []
        if stripped.startswith("#"):
            out.append(stripped)
        elif stripped:
            block.append(stripped)
    if block:
        out.append(" ".join(block))
    return out


def sentences(block: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", block) if s.strip()]


def check_caveats(report: str, required=tuple(CAVEATS)) -> list[str]:
    errors = []
    blocks = [u.lower() for u in units(report)]
    for cid in required:
        meaning, patterns = CAVEATS[cid]
        if not any(all(re.search(p, b) for p in patterns) for b in blocks):
            errors.append(f"missing caveat [{cid}]: state in one paragraph that {meaning}")
    return errors


def check_citations(report: str, n_chunks: int) -> list[str]:
    errors = []
    for m in CITATION.finditer(report):
        for cid in re.findall(r"S(\d+)", m.group(1)):
            if not 1 <= int(cid) <= n_chunks:
                errors.append(f"citation [S{cid}] does not exist (only S1-S{n_chunks} were given)")
    method = re.search(r"(?ims)^#+\s*method[^\n]*\n(.*?)(?=^#+\s|\Z)", report)
    if not method:
        return errors + ["no '## Method' section: describe the method and cite the SOPs there"]
    for block in units(method.group(1)):
        for sentence in sentences(block):
            if len(sentence.split()) >= 8 and not CITATION.search(sentence):
                errors.append(f"method claim without an SOP citation: '{sentence[:80]}...'; "
                              "add [S#]")  # fmt: skip
    return errors


def validate(report: str, facts: dict, n_chunks: int) -> list[str]:
    return (check_numbers(report, facts) + check_caveats(report)
            + check_citations(report, n_chunks))  # fmt: skip
