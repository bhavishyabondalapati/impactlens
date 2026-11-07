"""The guidance corpus (SOPs) the Retriever searches. One entry per downloaded document.

Licences are written down as found IN the document or on its page, not guessed. Documents
without an explicit open licence are used only locally for retrieval (data/raw/ is gitignored,
nothing is redistributed) and are always cited with their publisher.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    sid: str  # short id used in chunk ids, e.g. "fao"
    title: str
    publisher: str
    url: str
    file: str  # name under data/raw/sops/
    licence: str


SOURCES = (
    Source(
        "fao",
        "FAO's methodology for damage and loss assessment in agriculture (Statistics Working "
        "Paper 19-17, 2020)",
        "Food and Agriculture Organization of the United Nations (FAO)",
        "https://www.fao.org/3/ca6990en/ca6990en.pdf",
        "fao_damage_loss_methodology.pdf",
        "CC BY-NC-SA 3.0 IGO (stated on the document's copyright page)",
    ),
    Source(
        "pdna",
        "Post-Disaster Needs Assessments Guidelines, Volume A (2013)",
        "European Union, United Nations Development Group, World Bank (GFDRR)",
        "https://fpi.ec.europa.eu/system/files/2021-05/pdna_vol._a_en.pdf",
        "pdna_guidelines_vol_a.pdf",
        "No licence statement in the document; publicly distributed guidance. Used locally "
        "for retrieval only, not redistributed.",
    ),
    Source(
        "cems_methods",
        "Copernicus EMS Rapid Mapping Manual: Detection methods and Damage Assessment",
        "European Commission, Copernicus Emergency Management Service",
        "https://mapping.emergency.copernicus.eu/about/rapid-mapping-manual/"
        "detection-methods-damage-assessment/",
        "cems_manual_detection-methods-damage-assessment.html",
        "(c) European Union (page footer); EU reuse policy, Commission Decision 2011/833/EU: "
        "reuse allowed with attribution",
    ),
    Source(
        "cems_quality",
        "Copernicus EMS Rapid Mapping Manual: Quality Control",
        "European Commission, Copernicus Emergency Management Service",
        "https://mapping.emergency.copernicus.eu/about/rapid-mapping-manual/quality-control/",
        "cems_manual_quality-control.html",
        "(c) European Union (page footer); EU reuse policy, Commission Decision 2011/833/EU: "
        "reuse allowed with attribution",
    ),
    Source(
        "cems_product",
        "Copernicus EMS Rapid Mapping Manual: What is a Product?",
        "European Commission, Copernicus Emergency Management Service",
        "https://mapping.emergency.copernicus.eu/about/rapid-mapping-manual/product-overview/"
        "what-is-a-product/",
        "cems_manual_product-overview_what-is-a-product.html",
        "(c) European Union (page footer); EU reuse policy, Commission Decision 2011/833/EU: "
        "reuse allowed with attribution",
    ),
    Source(
        "unspider_steps",
        "UN-SPIDER Recommended Practice: Flood Mapping and Damage Assessment using Sentinel-2 "
        "data, Step by Step",
        "UN-SPIDER (UNOOSA)",
        "https://un-spider.org/es/node/11413",
        "unspider_s2_flood_step_by_step.html",
        "No licence statement on the page. Used locally for retrieval only, not redistributed.",
    ),
    Source(
        "unspider_detail",
        "UN-SPIDER Recommended Practice: Flood Mapping and Damage Assessment using Sentinel-2 "
        "data, In Detail",
        "UN-SPIDER (UNOOSA)",
        "https://un-spider.org/es/node/11412",
        "unspider_s2_flood_in_detail.html",
        "No licence statement on the page. Used locally for retrieval only, not redistributed.",
    ),
)


def sources_markdown() -> str:
    lines = ["# SOP corpus sources", "",
             "Downloaded by `impactlens sops`. Licences as stated in each document or page.", "",
             "| id | document | publisher | URL | licence |", "|---|---|---|---|---|"]  # fmt: skip
    for s in SOURCES:
        lines.append(f"| {s.sid} | {s.title} | {s.publisher} | {s.url} | {s.licence} |")
    return "\n".join(lines) + "\n"
