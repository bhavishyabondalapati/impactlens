"""Reporter prompts. One shared, cacheable prefix (rules + fact sheet + SOP excerpts) and a short
per-audience task, so the three reports reuse the same cached tokens."""

from __future__ import annotations

import json

from impactlens.agents.validator import CAVEATS

AUDIENCES = {
    "emergency": (
        "Emergency management / civil protection",
        "Where is the damage, how sure are we, and what should field teams check first? Lead "
        "with the towns ranked by vegetation-loss rate and urban change; say plainly that "
        "standing water had largely drained by the after date.",
    ),
    "agriculture": (
        "Agriculture ministry / crop insurer",
        "What happened to farmland? Cover rice paddies (standing water, and why it is "
        "ambiguous), vegetation loss on orchards, farmland and untagged fields, and what an "
        "insurer could and could not claim from these layers. Point to the FAO damage-and-loss "
        "approach for turning hectares into losses.",
    ),
    "municipal": (
        "Municipal governments in l'Horta Sud",
        "One short section per town in the fact sheet's by_town list that has a notable rate "
        "or urban change: its vegetation-loss rate, hectares, urban change, and how much of the "
        "municipality lies inside the analysed area.",
    ),
}

SYSTEM = """You write post-disaster impact reports from satellite analysis for a named audience.

Hard rules (an automatic validator checks every report, and failing reports are sent back):
1. NUMBERS: use only numbers that appear in the FACT SHEET, copied exactly or rounded (e.g.
   2104.0 -> "2,104 ha" or "about 2,100 ha"). Never compute new numbers (no sums, differences,
   ratios or conversions) and never use outside knowledge (no death tolls, costs, dates or
   population figures that are not in the fact sheet). Percentages must be ones in the sheet.
2. CAVEATS: the report must state each of these, each in its own paragraph or bullet:
{caveats}
3. CITATIONS: include a section headed exactly "## Method". End every sentence in it, before
   the full stop, with its source tag:
   - [S1], [S2] ... only if that SOP excerpt actually describes or supports the step.
   - [project] if the step is this project's own design choice that no excerpt describes
     (e.g. the land-use priority order, the NDVI and change thresholds, the per-town rates,
     the decision to report the port separately, models trained and tuned on public data).
   Never cite an excerpt just to have a citation: an honest [project] is better than a wrong
   [S#]. Do not cite SOPs for the Valencia numbers themselves; those come from the fact sheet.
4. Format: Markdown, a "# " title, short sections, at most ~600 words. Plain, factual tone. Say
   what the data shows and what it cannot show. No recommendations that the data does not
   support.
"""


def system_prompt() -> str:
    lines = [f"   - {meaning}" for meaning, _ in CAVEATS.values()]
    return SYSTEM.format(caveats="\n".join(lines))


def shared_context(facts: dict, chunks: list[dict]) -> str:
    sops = "\n\n".join(
        f"[S{i}] {c['title']} ({c['source']}"
        + (f", page {c['page']}" if c.get("page") else "") + f")\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )  # fmt: skip
    return ("FACT SHEET (the only source of numbers):\n"
            + json.dumps(facts, indent=1, ensure_ascii=False)
            + "\n\nSOP EXCERPTS (cite as [S#] in the Method section only where they support "
              "the step; otherwise tag the step [project]):\n\n" + sops)  # fmt: skip


def task_prompt(audience: str, errors: list[str] | None = None, previous: str | None = None) -> str:
    title, focus = AUDIENCES[audience]
    task = f"Write the report for: {title}.\nFocus: {focus}\n"
    if errors:
        task += ("\nYour previous draft FAILED validation. Fix every problem below and return the "
                 "whole corrected report (Markdown only, no commentary):\n"
                 + "\n".join(f"- {e}" for e in errors)
                 + "\n\nPrevious draft:\n" + (previous or ""))  # fmt: skip
    else:
        task += "Return only the Markdown report."
    return task
