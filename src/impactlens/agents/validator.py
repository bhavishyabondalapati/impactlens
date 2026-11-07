"""The Validator: plain code, no LLM. An LLM judging numbers can make the same mistakes it is
meant to catch, so the checks are deterministic and unit-tested.

Three checks per report:
  (a) numbers: every number in the text must match a value in the fact sheet, within rounding
      (half a unit of the last digit written, or 0.5% for numbers >= 100: "2,100 ha" may stand
      for 2,104 ha, "2,000 ha" may not). Numbers with a unit (ha, %, km, ...) are always checked;
      bare integers 0-10 are allowed as ordinary counting words ("3 reports").
  (b) caveats: each required caveat must appear, i.e. its key ideas must co-occur in one
      paragraph or bullet (e.g. "paddy" + "seasonal" + "ambiguous").
  (c) citations: every sentence of the Method section (8+ words) cites an SOP chunk like [S3]
      or is tagged [project] (our own design choice), and every cited id exists. Whether the
      cited excerpt really supports the sentence is NOT checked (see docs/AGENTS.md).
  (d) rankings: towns listed as a ranking (numbered list, or "highest", "worst", "top" ...)
      must be in the order of the fact-sheet metric the list quotes.
  (e) overclaims: "confirmed", "proven", "definitely", "certainly" are rejected in sentences
      about NDVI drop, vegetation loss, mud or paddy water, unless negated or conditional.
"""

from __future__ import annotations

import math
import re
import unicodedata
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
# A Method sentence is backed either by an SOP excerpt [S#] or by [project]: "this step is our
# own design choice (e.g. the land-use priority order), not something the guidance says".
# [project] keeps the model from borrowing an unrelated SOP just to satisfy the rule.
SOURCE_TAG = re.compile(r"\[(?:S\d+(?:\s*,\s*S\d+)*|project)\]", re.IGNORECASE)
# Tags at the very start of a piece, e.g. the "[S1]" left over from "... the date. [S1] Next"
LEADING_TAGS = re.compile(r"^(\s*\[(?:S\d+(?:\s*,\s*S\d+)*|project)\][\s,.;]*)+", re.IGNORECASE)


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
    """Split a paragraph into sentences. A citation written after the full stop ("... the
    date. [S1]") lands at the start of the next piece; it is moved back onto the sentence right
    before it, and only that one sentence (it cannot cover two uncited sentences)."""
    out = []
    for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", block):
        tags = LEADING_TAGS.match(piece)
        if out and tags:
            out[-1] = f"{out[-1]} {tags.group(0).strip()}"
            piece = piece[tags.end() :]
        if piece.strip():
            out.append(piece)
    return out


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
            if len(sentence.split()) >= 8 and not SOURCE_TAG.search(sentence):
                errors.append(
                    f"method sentence without a citation: '{sentence.strip()}'. End it, before "
                    "the full stop, with [S#] if an SOP excerpt supports it, or [project] if it "
                    "is our own design choice"
                )
    return errors


# ------------------------------------------------------------------- (d) ranking order


RANK_WORDS = re.compile(
    r"\b(highest|worst|most|largest|biggest|greatest|top|rank(ed|ing)?|lowest|least|smallest)\b",
    re.IGNORECASE,
)
ASCENDING_WORDS = re.compile(r"\b(lowest|least|smallest)\b", re.IGNORECASE)
LIST_ITEM = re.compile(r"^\s*([-*+]|(\d+)[.)])\s+(.*)$")


def _plain(text: str) -> str:
    """Lowercase without accents, so 'Sedavi' and 'Valencia' match 'Sedaví' and 'València'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def town_metrics(facts: dict) -> dict[str, dict[str, float]]:
    """Every per-town number in the fact sheet, as {metric: {town: value}}."""
    veg = facts.get("vegetation_loss_likely_mud", {})
    metrics = {
        "NDVI-drop rate (% of vegetated land)": veg.get("pct_of_vegetated_land_by_town", {}),
        "NDVI-drop hectares": veg.get("by_town", {}),
        "urban change hectares": facts.get("urban_change", {}).get("by_town", {}),
    }
    for row in facts.get("by_town", []):
        for key, value in row.items():
            if key != "name" and isinstance(value, int | float):
                metrics.setdefault(key, {})[row["name"]] = value
    return {m: v for m, v in metrics.items() if v}


@dataclass
class RankedList:
    lead: str  # the heading / sentence / table header just before the list
    numbered: bool
    items: list[str]


def ranked_lists(report: str) -> list[RankedList]:
    """Markdown lists and tables, each with the line that introduces it. Wrapped lines join
    their item; blank lines between items do not end the list."""
    lists, lead, current = [], "", None
    for line in report.splitlines():
        s = line.strip()
        item = LIST_ITEM.match(line)
        is_row = s.startswith("|") and not re.fullmatch(r"[|:\-\s]+", s)
        if item or is_row:
            if current is None:
                current = RankedList(lead, bool(item and item.group(2)), [])
                lists.append(current)
                if is_row:  # the first table row is its header: it counts as the lead
                    current.lead += " " + s
                    continue
            current.items.append(item.group(3) if item else s)
        elif re.fullmatch(r"[|:\-\s]+", s or "x"):
            continue  # table separator row
        elif not s:
            continue  # a blank line inside a list does not end it
        elif current is not None and line[:1].isspace():
            current.items[-1] += " " + s  # wrapped line of the last item
        else:
            current, lead = None, s
    return lists


def check_rankings(report: str, facts: dict) -> list[str]:
    """If towns are listed as a ranking (numbered, or introduced by 'highest', 'worst',
    'top', ...), their order must be the order of some per-town metric in the fact sheet:
    the metric whose values the items quote, or any metric if they quote none."""
    metrics = town_metrics(facts)
    towns = {_plain(t): t for m in metrics.values() for t in m}
    errors = []
    for rl in ranked_lists(report):
        if not (rl.numbered or RANK_WORDS.search(rl.lead)):
            continue
        rows = []  # (town, item text)
        grouped = False  # an item naming several towns: a to-do list, not a ranking
        for text in rl.items:
            named = {t for k, t in towns.items()
                     if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", _plain(text))}  # fmt: skip
            grouped |= len(named) > 1
            if len(named) == 1:
                rows.append((named.pop(), text))
        names = [t for t, _ in rows]
        if grouped or len(rows) < 2 or len(set(names)) < len(names):
            continue
        # metrics that every item quotes (its number for that town is in the item)
        quoted = [m for m, vals in metrics.items()
                  if all(t in vals and matches_any(numbers_in(x), vals[t])
                         for t, x in rows)]  # fmt: skip
        candidates = quoted or [m for m, vals in metrics.items() if all(t in vals for t, _ in rows)]
        descending = not ASCENDING_WORDS.search(rl.lead)
        if not candidates or any(_in_order([metrics[m][t] for t, _ in rows], descending)
                                 for m in candidates):  # fmt: skip
            continue
        m = candidates[0]
        expected = sorted((t for t, _ in rows), key=lambda t: metrics[m][t], reverse=descending)
        written = ", ".join(t for t, _ in rows)
        errors.append(f"ranking out of order under '{rl.lead[:60]}': you listed {written}, but "
                      f"by {m} the fact sheet order is {', '.join(expected)}. List them in "
                      "that order (or drop the ranking wording)")  # fmt: skip
    return errors


def matches_any(found: list[Found], value: float) -> bool:
    return any(matches(f, [value]) for f in found)


def _in_order(values: list[float], descending: bool) -> bool:
    pairs = zip(values, values[1:], strict=False)
    return all(a >= b if descending else a <= b for a, b in pairs)


# --------------------------------------------------------------------- (e) overclaims


PROXY_TOPIC = re.compile(r"ndvi|vegetation[- ]?(loss|drop|damage)|loss of vegetation|\bmud"
                         r"|padd(y|ies)|\brice\b", re.IGNORECASE)  # fmt: skip
OVERCLAIM = re.compile(r"\b(confirmed|proven|definitely|certainly)\b", re.IGNORECASE)
# "cannot be confirmed", "before damage can be confirmed", "not proven" are honest, not claims
HEDGE = re.compile(r"\b(not|no|never|cannot|can't|nor|without|before|until|unless|can|could"
                   r"|must|should|would|need|needs|needed|requires?|required|to)\b",
                   re.IGNORECASE)  # fmt: skip


def check_overclaims(report: str) -> list[str]:
    """NDVI drop and paddy water are proxies. In a sentence about them, 'confirmed', 'proven',
    'definitely' or 'certainly' is an overclaim, unless it is negated or conditional within
    the four words before it ('cannot be confirmed', 'must be confirmed in the field')."""
    errors = []
    for block in units(report):
        for sentence in sentences(block):
            if not PROXY_TOPIC.search(sentence):
                continue
            for m in OVERCLAIM.finditer(sentence):
                before = sentence[: m.start()].split()[-4:]
                if not any(HEDGE.fullmatch(w.strip(".,;:()*")) for w in before):
                    errors.append(f"overclaim {m.group(0)!r} in: '{sentence.strip()}'. NDVI drop "
                                  "and paddy water are proxies, not proof; write 'indicates' or "
                                  "'suggests' instead")  # fmt: skip
    return errors


def validate(report: str, facts: dict, n_chunks: int) -> list[str]:
    return (check_numbers(report, facts) + check_caveats(report)
            + check_citations(report, n_chunks) + check_rankings(report, facts)
            + check_overclaims(report))  # fmt: skip
