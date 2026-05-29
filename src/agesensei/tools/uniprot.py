"""UniProt REST API wrapper.

Fetches protein sequences, annotations, and functional data.
API docs: https://rest.uniprot.org/docs/
Rate limit: generous, no key needed.
"""

import asyncio
import json
import httpx
from pathlib import Path
from agesensei.config import CACHE_DIR

BASE_URL = "https://rest.uniprot.org"

_cache: dict[str, dict] = {}


async def fetch_protein(uniprot_id: str) -> dict:
    """Fetch protein entry by UniProt accession ID.

    Returns:
        Dict with keys: accession, gene, name, organism, sequence,
        length, function, subcellular_location, go_terms, pdb_ids
    """
    if uniprot_id in _cache:
        return _cache[uniprot_id]

    cache_file = CACHE_DIR / "uniprot" / f"{uniprot_id}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        _cache[uniprot_id] = data
        return data

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{BASE_URL}/uniprotkb/{uniprot_id}.json")
        response.raise_for_status()
        raw = response.json()

    data = _parse_entry(raw)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    _cache[uniprot_id] = data
    return data


async def search_by_gene(gene_symbol: str, organism: str = "Homo sapiens") -> str | None:
    """Search UniProt by gene symbol, return best matching accession ID."""
    query = f"(gene:{gene_symbol}) AND (organism_name:{organism}) AND (reviewed:true)"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{BASE_URL}/uniprotkb/search",
            params={"query": query, "format": "json", "size": 1, "fields": "accession"},
        )
        response.raise_for_status()
        data = response.json()

    results = data.get("results", [])
    if results:
        return results[0]["primaryAccession"]
    return None


async def get_sequence(gene_symbol: str) -> tuple[str, str]:
    """Get protein sequence for a gene symbol.

    Returns:
        Tuple of (uniprot_id, amino_acid_sequence)
    """
    accession = await search_by_gene(gene_symbol)
    if not accession:
        return ("", "")
    protein = await fetch_protein(accession)
    return (accession, protein.get("sequence", ""))


async def batch_fetch(gene_symbols: list[str]) -> dict[str, dict]:
    """Fetch protein data for multiple genes in parallel."""
    results = {}

    accession_tasks = [search_by_gene(g) for g in gene_symbols]
    accessions = await asyncio.gather(*accession_tasks, return_exceptions=True)

    fetch_tasks = []
    gene_map = {}
    for gene, acc in zip(gene_symbols, accessions):
        if isinstance(acc, str) and acc:
            fetch_tasks.append(fetch_protein(acc))
            gene_map[acc] = gene

    proteins = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    for protein in proteins:
        if isinstance(protein, dict):
            gene = gene_map.get(protein.get("accession", ""), "")
            if gene:
                results[gene] = protein

    return results


def _parse_entry(raw: dict) -> dict:
    """Parse UniProt JSON entry into simplified dict."""
    genes = raw.get("genes", [])
    gene_symbol = genes[0].get("geneName", {}).get("value", "") if genes else ""

    protein_desc = raw.get("proteinDescription", {})
    rec_name = protein_desc.get("recommendedName", {})
    protein_name = rec_name.get("fullName", {}).get("value", "")

    sequence_data = raw.get("sequence", {})
    sequence = sequence_data.get("value", "")
    length = sequence_data.get("length", 0)

    function = ""
    subcellular = ""
    for comment in raw.get("comments", []):
        ctype = comment.get("commentType", "")
        if ctype == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                function = texts[0].get("value", "")
        elif ctype == "SUBCELLULAR LOCATION":
            locs = comment.get("subcellularLocations", [])
            loc_names = [loc.get("location", {}).get("value", "") for loc in locs]
            subcellular = "; ".join(filter(None, loc_names))

    go_terms = []
    pdb_ids = []
    for xref in raw.get("uniProtKBCrossReferences", []):
        db = xref.get("database")
        if db == "GO":
            props = {p["key"]: p["value"] for p in xref.get("properties", [])}
            go_terms.append({
                "id": xref.get("id", ""),
                "term": props.get("GoTerm", ""),
                "source": props.get("GoEvidenceType", ""),
            })
        elif db == "PDB":
            pdb_ids.append(xref["id"])

    return {
        "accession": raw.get("primaryAccession", ""),
        "gene": gene_symbol,
        "name": protein_name,
        "organism": raw.get("organism", {}).get("scientificName", ""),
        "sequence": sequence,
        "length": length,
        "function": function,
        "subcellular_location": subcellular,
        "go_terms": go_terms,
        "pdb_ids": pdb_ids,
    }
