"""PubMed Central (PMC) full-text wrapper.

PMC serves open-access biomedical literature as structured JATS XML.
Same NCBI E-utilities endpoint as PubMed, different ``db`` parameter.

Typical flow:
    1. Start from a PMID (PubMed abstract) and resolve to PMCID via elink.
    2. Fetch PMC JATS XML via efetch.
    3. Parse <sec> hierarchy into a {section_name: text} dict for
       progressive / on-demand reading.

Not every PubMed article is in PMC (only open-access + deposited). Callers
must handle ``None`` gracefully.
"""

import httpx
from xml.etree import ElementTree as ET

from agesensei.config import config
from agesensei.tools.pubmed import _request_with_retry

ELINK_URL = f"{config.pubmed.base_url}/elink.fcgi"
EFETCH_URL = f"{config.pubmed.base_url}/efetch.fcgi"


def _auth_params() -> dict:
    params = {}
    if config.pubmed.email:
        params["email"] = config.pubmed.email
    if config.pubmed.api_key:
        params["api_key"] = config.pubmed.api_key
    return params


async def pmid_to_pmcid(pmid: str) -> str | None:
    """Resolve a PubMed ID to its PMC ID (if open-access and deposited).

    Returns PMCID WITHOUT the "PMC" prefix (just the digits), or None.
    """
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid,
        "retmode": "xml",
        **_auth_params(),
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await _request_with_retry(client, ELINK_URL, params)

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        return None

    # ELink response: //LinkSet/LinkSetDb/Link/Id
    for link_set in root.findall("LinkSet"):
        for link_set_db in link_set.findall("LinkSetDb"):
            dbname = link_set_db.findtext("DbTo", "")
            if dbname != "pmc":
                continue
            first_id = link_set_db.findtext("Link/Id")
            if first_id:
                return first_id
    return None


def _normalize_pmcid(pmcid: str) -> str:
    """Accept 'PMC544940' or '544940', always return just the digits."""
    s = pmcid.strip()
    if s.upper().startswith("PMC"):
        s = s[3:]
    return s


def _element_text(el: ET.Element) -> str:
    """Flatten element text including tail text from children, dropping tags."""
    parts = []
    if el.text:
        parts.append(el.text)
    for child in el:
        parts.append(_element_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(p.strip() for p in parts if p and p.strip())


def _section_title(sec: ET.Element) -> str:
    title_el = sec.find("title")
    if title_el is not None:
        return _element_text(title_el).lower()
    sec_type = sec.attrib.get("sec-type", "").lower()
    return sec_type or "unnamed"


def _normalize_section_name(name: str) -> str:
    """Map heterogeneous section titles to canonical keys."""
    n = name.lower().strip()
    mapping = {
        "introduction": ["introduction", "background", "intro"],
        "methods": ["methods", "method", "materials and methods", "experimental procedures"],
        "results": ["results", "result"],
        "discussion": ["discussion", "discussion and conclusions"],
        "conclusion": ["conclusion", "conclusions", "concluding remarks"],
        "abstract": ["abstract"],
    }
    for canonical, aliases in mapping.items():
        if any(a in n for a in aliases):
            return canonical
    return n


async def fetch_full_text(pmcid: str) -> dict[str, str]:
    """Fetch a PMC article and return {canonical_section_name: section_text}.

    Returns an empty dict if the article cannot be fetched or parsed. Abstract
    is included under the ``"abstract"`` key when present.
    """
    pmcid = _normalize_pmcid(pmcid)
    params = {
        "db": "pmc",
        "id": pmcid,
        "rettype": "xml",
        "retmode": "xml",
        **_auth_params(),
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await _request_with_retry(client, EFETCH_URL, params)

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        return {}

    sections: dict[str, str] = {}

    # Abstract lives outside body in JATS
    for abs_el in root.iter("abstract"):
        txt = _element_text(abs_el)
        if txt:
            sections["abstract"] = txt
            break

    body = next(iter(root.iter("body")), None)
    if body is None:
        return sections

    for sec in body.findall("sec"):
        name = _normalize_section_name(_section_title(sec))
        # Drop nested sec titles from the top-level text
        text_parts = []
        for child in sec:
            if child.tag == "title":
                continue
            text_parts.append(_element_text(child))
        text = " ".join(p for p in text_parts if p).strip()
        if not text:
            continue
        if name in sections:
            sections[name] = sections[name] + "\n\n" + text
        else:
            sections[name] = text

    return sections
