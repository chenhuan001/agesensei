"""Semantic Scholar API wrapper.

API Documentation: https://api.semanticscholar.org/api-docs/
Rate limits: 100 requests per 5 minutes (no key), 5000 req/5min with key
"""

import asyncio
import httpx
from agesensei.config import config
from agesensei.schema import Paper

BASE_URL = config.semantic_scholar.base_url


async def search_papers(query: str, max_results: int = 100) -> list[Paper]:
    """Search Semantic Scholar for papers.

    Args:
        query: Search query
        max_results: Maximum results (API returns max 100 per request)

    Returns:
        List of Paper objects with citation counts and relevance scores
    """
    # S2 API returns max 100 results per request, need pagination for more
    limit = min(max_results, 100)
    offset = 0
    all_papers = []

    headers = {}
    if config.semantic_scholar.api_key:
        headers["x-api-key"] = config.semantic_scholar.api_key

    async with httpx.AsyncClient(timeout=30.0) as client:
        while len(all_papers) < max_results:
            params = {
                "query": query,
                "limit": limit,
                "offset": offset,
                "fields": "paperId,title,abstract,authors,year,citationCount,url,externalIds",
            }

            for attempt in range(3):
                response = await client.get(
                    f"{BASE_URL}/paper/search",
                    params=params,
                    headers=headers,
                )
                if response.status_code == 429:
                    wait = 2 ** attempt + 1
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                break
            else:
                break  # all retries exhausted

            data = response.json()

            papers_data = data.get("data", [])
            if not papers_data:
                break

            for item in papers_data:
                paper = _parse_s2_paper(item)
                all_papers.append(paper)

            # Check if there are more results
            total = data.get("total", 0)
            if len(all_papers) >= total or len(all_papers) >= max_results:
                break

            offset += limit

            # Rate limiting
            if not config.semantic_scholar.api_key:
                await asyncio.sleep(0.6)  # ~100 req / 5 min = 1 req / 3s

    return all_papers[:max_results]


async def get_paper_details(paper_id: str) -> dict:
    """Get detailed paper info including references and citations.

    Args:
        paper_id: Semantic Scholar paper ID

    Returns:
        Dict with full paper metadata
    """
    headers = {}
    if config.semantic_scholar.api_key:
        headers["x-api-key"] = config.semantic_scholar.api_key

    params = {
        "fields": "paperId,title,abstract,authors,year,citationCount,referenceCount,"
                  "influentialCitationCount,url,externalIds,publicationDate,journal"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{BASE_URL}/paper/{paper_id}",
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


async def get_citations(paper_id: str, limit: int = 100) -> list[Paper]:
    """Get papers that cite this paper.

    Args:
        paper_id: Semantic Scholar paper ID
        limit: Maximum citations to retrieve

    Returns:
        List of citing papers
    """
    headers = {}
    if config.semantic_scholar.api_key:
        headers["x-api-key"] = config.semantic_scholar.api_key

    params = {
        "fields": "paperId,title,abstract,authors,year,citationCount,url,externalIds",
        "limit": min(limit, 1000),
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{BASE_URL}/paper/{paper_id}/citations",
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        papers = []
        for item in data.get("data", []):
            citing_paper = item.get("citingPaper", {})
            paper = _parse_s2_paper(citing_paper)
            papers.append(paper)

        return papers


def _parse_s2_paper(data: dict) -> Paper:
    """Parse Semantic Scholar API response into Paper object."""
    # Extract external IDs
    external_ids = data.get("externalIds", {})
    pmid = external_ids.get("PubMed")
    doi = external_ids.get("DOI")

    # Extract authors
    authors = []
    for author in data.get("authors", []):
        name = author.get("name", "")
        if name:
            authors.append(name)

    # Extract other fields
    paper = Paper(
        pmid=pmid,
        doi=doi,
        title=data.get("title", "") or "",
        authors=authors,
        journal="",
        year=data.get("year"),
        abstract=data.get("abstract") or "",
        url=data.get("url") or "",
        citation_count=data.get("citationCount") or 0,
    )

    return paper


async def search_by_doi(doi: str) -> Paper | None:
    """Search for a paper by DOI.

    Args:
        doi: Digital Object Identifier

    Returns:
        Paper object or None if not found
    """
    headers = {}
    if config.semantic_scholar.api_key:
        headers["x-api-key"] = config.semantic_scholar.api_key

    params = {
        "fields": "paperId,title,abstract,authors,year,citationCount,url,externalIds"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{BASE_URL}/paper/DOI:{doi}",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            return _parse_s2_paper(data)
        except httpx.HTTPStatusError:
            return None
