"""PubMed E-utilities API wrapper.

API Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25501/
Rate limits: 3 requests/second without API key, 10 req/s with key
"""

import asyncio
import httpx
from xml.etree import ElementTree as ET
from agesensei.config import config
from agesensei.schema import Paper

ESEARCH_URL = f"{config.pubmed.base_url}/esearch.fcgi"
EFETCH_URL = f"{config.pubmed.base_url}/efetch.fcgi"

MAX_RETRIES = 3
RETRY_DELAY = 2.0


async def _request_with_retry(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    """Make HTTP request with retry on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(url, params=params)
            if response.status_code == 429:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise
    raise httpx.ConnectError("Max retries exceeded")


async def search_pubmed(query: str, max_results: int = 100) -> list[str]:
    """Search PubMed and return list of PMIDs.

    Args:
        query: PubMed search query (supports boolean operators and MeSH terms)
               Example: "cellular senescence[MeSH] AND drug targets"
        max_results: Maximum number of results

    Returns:
        List of PMID strings
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }

    if config.pubmed.email:
        params["email"] = config.pubmed.email
    if config.pubmed.api_key:
        params["api_key"] = config.pubmed.api_key

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await _request_with_retry(client, ESEARCH_URL, params)
        data = response.json()

        pmids = data.get("esearchresult", {}).get("idlist", [])
        return pmids


async def fetch_abstracts(pmids: list[str]) -> list[Paper]:
    """Fetch paper details (title, abstract, authors, journal) for given PMIDs.

    Args:
        pmids: List of PubMed IDs

    Returns:
        List of Paper objects
    """
    if not pmids:
        return []

    # PubMed recommends batches of 200-500
    batch_size = 200
    papers = []

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        batch_papers = await _fetch_batch(batch)
        papers.extend(batch_papers)

        # Rate limiting: 3 req/s without key, 10 req/s with key
        if not config.pubmed.api_key:
            await asyncio.sleep(0.34)  # ~3 req/s

    return papers


async def _fetch_batch(pmids: list[str]) -> list[Paper]:
    """Fetch a batch of papers via EFetch."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    if config.pubmed.email:
        params["email"] = config.pubmed.email
    if config.pubmed.api_key:
        params["api_key"] = config.pubmed.api_key

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await _request_with_retry(client, EFETCH_URL, params)

        return _parse_pubmed_xml(response.text)


def _parse_pubmed_xml(xml_text: str) -> list[Paper]:
    """Parse PubMed XML response into Paper objects."""
    papers = []
    root = ET.fromstring(xml_text)

    for article in root.findall(".//PubmedArticle"):
        try:
            # Extract PMID
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            # Extract title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""

            # Extract abstract
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join([
                (elem.text or "") for elem in abstract_parts
            ])

            # Extract authors
            authors = []
            for author in article.findall(".//Author"):
                lastname = author.find("LastName")
                forename = author.find("ForeName")
                if lastname is not None and forename is not None:
                    authors.append(f"{forename.text} {lastname.text}")

            # Extract journal
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""

            # Extract year
            year_elem = article.find(".//PubDate/Year")
            year = int(year_elem.text) if year_elem is not None else None

            # Extract DOI
            doi = None
            for article_id in article.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break

            # Construct URL
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            paper = Paper(
                pmid=pmid,
                doi=doi,
                title=title,
                authors=authors,
                journal=journal,
                year=year,
                abstract=abstract,
                url=url,
            )
            papers.append(paper)

        except Exception as e:
            # Skip malformed entries
            continue

    return papers


async def search_and_fetch(query: str, max_results: int = 100) -> list[Paper]:
    """Convenience function: search + fetch in one call.

    Example:
        papers = await search_and_fetch("cellular senescence AND drug targets", max_results=50)
    """
    pmids = await search_pubmed(query, max_results)
    papers = await fetch_abstracts(pmids)
    return papers
