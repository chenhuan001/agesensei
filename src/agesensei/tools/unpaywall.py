"""Unpaywall API client — free full-text access for OA papers via DOI.

Unpaywall covers ~47M open-access papers. Given a DOI, it returns the
best OA location (PDF URL or HTML URL) which we can then fetch and parse.

API docs: https://unpaywall.org/products/api
Rate limit: 100k requests/day with polite email header.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "agesensei.project@gmail.com")


@dataclass
class OALocation:
    """Open access location for a paper."""
    doi: str
    title: str
    pdf_url: str | None
    html_url: str | None
    is_oa: bool


async def get_oa_location(doi: str) -> OALocation | None:
    """Look up OA availability for a DOI via Unpaywall."""
    url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data.get("is_oa"):
            return None

        best = data.get("best_oa_location", {}) or {}
        return OALocation(
            doi=doi,
            title=data.get("title", ""),
            pdf_url=best.get("url_for_pdf"),
            html_url=best.get("url_for_landing_page"),
            is_oa=True,
        )


async def fetch_fulltext_via_unpaywall(doi: str) -> dict[str, str] | None:
    """Fetch full text for a DOI via Unpaywall OA links.

    Returns dict of section_name -> text, similar to PMC format.
    Falls back to HTML parsing if PDF is not available.
    """
    location = await get_oa_location(doi)
    if not location:
        return None

    # Try HTML first (easier to parse than PDF)
    if location.html_url:
        try:
            sections = await _fetch_html_fulltext(location.html_url)
            if sections:
                return sections
        except Exception as e:
            logger.debug(f"HTML fetch failed for {doi}: {e}")

    # Try PDF
    if location.pdf_url:
        try:
            sections = await _fetch_pdf_fulltext(location.pdf_url)
            if sections:
                return sections
        except Exception as e:
            logger.debug(f"PDF fetch failed for {doi}: {e}")

    return None


async def _fetch_html_fulltext(url: str) -> dict[str, str] | None:
    """Fetch and parse HTML full text from publisher page."""
    async with httpx.AsyncClient(
        timeout=20,
        follow_redirects=True,
        headers={"User-Agent": "AgeSensei/1.0 (mailto:agesensei@example.com)"},
    ) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return None

        html = resp.text
        if len(html) < 500:
            return None

        return _parse_html_sections(html)


def _parse_html_sections(html: str) -> dict[str, str]:
    """Extract text sections from a scientific paper HTML page.

    Uses simple regex/string parsing — no heavy dependencies.
    """
    sections: dict[str, str] = {}

    # Remove script/style tags
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Extract abstract
    abstract_match = re.search(
        r'(?:class="[^"]*abstract[^"]*"|id="[^"]*abstract[^"]*")[^>]*>(.*?)(?=<(?:h[12]|section|div\s+class))',
        html, re.DOTALL | re.IGNORECASE,
    )
    if abstract_match:
        text = re.sub(r'<[^>]+>', ' ', abstract_match.group(1))
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > 100:
            sections["abstract"] = text[:3000]

    # Extract body text by looking for common section patterns
    section_patterns = [
        (r'(?:introduction|background)', 'introduction'),
        (r'(?:results?)', 'results'),
        (r'(?:methods?|materials?\s+and\s+methods?|experimental)', 'methods'),
        (r'(?:discussion)', 'discussion'),
        (r'(?:conclusion)', 'conclusion'),
    ]

    for pattern, name in section_patterns:
        match = re.search(
            rf'<h[23][^>]*>\s*(?:<[^>]+>)*\s*{pattern}\s*(?:<[^>]+>)*\s*</h[23]>(.*?)(?=<h[23]|$)',
            html, re.DOTALL | re.IGNORECASE,
        )
        if match:
            text = re.sub(r'<[^>]+>', ' ', match.group(1))
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 100:
                sections[name] = text[:3000]

    # Fallback: if no sections found, extract all paragraph text
    if not sections:
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE)
        all_text = []
        for p in paragraphs:
            text = re.sub(r'<[^>]+>', ' ', p)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 50:
                all_text.append(text)
        if all_text:
            combined = "\n\n".join(all_text[:30])
            if len(combined) > 200:
                sections["body"] = combined[:5000]

    return sections if sections else None


async def _fetch_pdf_fulltext(url: str) -> dict[str, str] | None:
    """Fetch PDF and extract text using pdfplumber (if available)."""
    try:
        import pdfplumber
    except ImportError:
        logger.debug("pdfplumber not installed, skipping PDF extraction")
        return None

    import io

    async with httpx.AsyncClient(
        timeout=30,
        follow_redirects=True,
        headers={"User-Agent": "AgeSensei/1.0 (mailto:agesensei@example.com)"},
    ) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return None

        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type and not url.endswith(".pdf"):
            return None

        pdf_bytes = resp.content
        if len(pdf_bytes) < 1000:
            return None

    # Parse PDF
    sections = {}
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            all_text = []
            for page in pdf.pages[:30]:  # Cap at 30 pages
                text = page.extract_text()
                if text:
                    all_text.append(text)

            if all_text:
                full_text = "\n".join(all_text)
                # Try to split by section headers
                section_splits = re.split(
                    r'\n\s*(Abstract|Introduction|Background|Results|Methods|'
                    r'Materials and Methods|Discussion|Conclusion)\s*\n',
                    full_text, flags=re.IGNORECASE,
                )

                if len(section_splits) > 2:
                    # Paired: [pre, header, content, header, content, ...]
                    for i in range(1, len(section_splits) - 1, 2):
                        name = section_splits[i].lower().strip()
                        content = section_splits[i + 1].strip()
                        if len(content) > 100:
                            sections[name] = content[:3000]
                else:
                    sections["body"] = full_text[:8000]
    except Exception as e:
        logger.debug(f"PDF parsing failed: {e}")
        return None

    return sections if sections else None
