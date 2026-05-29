"""arXiv API wrapper.

Public Atom API: http://export.arxiv.org/api/query
Rate limits: 1 request per 3 seconds (self-imposed courtesy).

Useful categories for aging / longevity research:
  - q-bio.BM  Biomolecules
  - q-bio.MN  Molecular Networks
  - q-bio.QM  Quantitative Methods
  - q-bio.CB  Cell Behavior
  - q-bio.GN  Genomics
  - cs.LG     Machine Learning (aging biomarkers / epigenetic clocks)
  - cs.AI     Artificial Intelligence
  - stat.ML   Machine Learning (statistics side)
"""

import asyncio
import re
import httpx
from xml.etree import ElementTree as ET

from agesensei.schema import Paper

BASE_URL = "http://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

AGING_DEFAULT_CATEGORIES = ["q-bio.BM", "q-bio.MN", "q-bio.QM", "q-bio.CB", "q-bio.GN", "cs.LG"]

MAX_RETRIES = 3
RETRY_DELAY = 2.0


def _build_category_filter(categories: list[str]) -> str:
    """Build arXiv search filter that ORs a list of categories."""
    if not categories:
        return ""
    clauses = [f"cat:{c}" for c in categories]
    return "(" + " OR ".join(clauses) + ")"


def _build_search_query(query: str, categories: list[str] | None) -> str:
    """Construct the search_query param for arXiv API.

    Format docs: https://info.arxiv.org/help/api/user-manual.html#query_details
    """
    # Full-text match across title + abstract
    terms = f'(ti:"{query}" OR abs:"{query}")'
    cat_filter = _build_category_filter(categories or [])
    return f"{terms} AND {cat_filter}" if cat_filter else terms


async def _request_with_retry(client: httpx.AsyncClient, params: dict) -> httpx.Response:
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(BASE_URL, params=params)
            if response.status_code == 429:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        except (httpx.ConnectError, httpx.ReadTimeout):
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise
    raise httpx.ConnectError("Max retries exceeded")


def _parse_arxiv_id(id_url: str) -> str:
    """Turn 'http://arxiv.org/abs/2409.05591v2' into '2409.05591'."""
    m = re.search(r"abs/([0-9.]+v?\d*)", id_url)
    if not m:
        return id_url
    return m.group(1).split("v")[0]


def _parse_entry(entry: ET.Element) -> Paper:
    def _find_text(path: str, default: str = "") -> str:
        el = entry.find(path, NS)
        return (el.text or default).strip() if el is not None and el.text else default

    id_url = _find_text("atom:id")
    arxiv_id = _parse_arxiv_id(id_url)

    title = re.sub(r"\s+", " ", _find_text("atom:title"))
    summary = re.sub(r"\s+", " ", _find_text("atom:summary"))
    published = _find_text("atom:published")
    year = None
    if len(published) >= 4 and published[:4].isdigit():
        year = int(published[:4])

    authors = [
        (a.findtext("atom:name", default="", namespaces=NS) or "").strip()
        for a in entry.findall("atom:author", NS)
    ]
    authors = [a for a in authors if a]

    categories = [
        c.attrib.get("term", "")
        for c in entry.findall("atom:category", NS)
        if c.attrib.get("term")
    ]

    doi = None
    for link in entry.findall("atom:link", NS):
        if link.attrib.get("title") == "doi":
            href = link.attrib.get("href", "")
            if "doi.org/" in href:
                doi = href.split("doi.org/", 1)[1]

    return Paper(
        title=title,
        abstract=summary,
        authors=authors,
        year=year,
        url=id_url,
        doi=doi,
        arxiv_id=arxiv_id,
        categories=categories,
        source="arxiv",
        journal="arXiv",
    )


async def search_arxiv(
    query: str,
    max_results: int = 50,
    categories: list[str] | None = None,
    sort_by: str = "relevance",
) -> list[Paper]:
    """Search arXiv and return Paper objects.

    Args:
        query: Natural-language search phrase.
        max_results: Max papers to return (hard cap 2000 server-side).
        categories: List of arXiv categories to restrict to. Pass None to search all,
                    or [] to use aging defaults (AGING_DEFAULT_CATEGORIES).
        sort_by: "relevance" | "lastUpdatedDate" | "submittedDate".

    Returns:
        List of Paper with source="arxiv". Empty list on network failure.
    """
    if categories is None:
        categories = AGING_DEFAULT_CATEGORIES

    params = {
        "search_query": _build_search_query(query, categories),
        "start": 0,
        "max_results": min(max_results, 200),
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await _request_with_retry(client, params)

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        return []

    entries = root.findall("atom:entry", NS)
    papers = [_parse_entry(e) for e in entries]
    return papers[:max_results]
