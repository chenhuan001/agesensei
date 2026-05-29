"""Open Targets Platform GraphQL API wrapper.

Integrates GWAS, expression, literature, animal models, pathways, drugs.
GraphQL API: https://platform-docs.opentargets.org/
No API key needed. Rate limit: ~10 req/s.
"""

import asyncio
import httpx

API_URL = "https://api.platform.opentargets.org/api/v4/graphql"
MAX_RETRIES = 3


async def _graphql(query: str, variables: dict) -> dict:
    """Execute GraphQL query with retry."""
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    API_URL,
                    json={"query": query, "variables": variables},
                )
                if response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                return response.json().get("data", {})
        except (httpx.ConnectError, httpx.ReadTimeout):
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise
    return {}


async def search_target(gene_symbol: str) -> str | None:
    """Search for Ensembl gene ID by gene symbol.

    Returns: Ensembl ID (e.g., "ENSG00000141510" for TP53) or None
    """
    query = """
    query SearchTarget($q: String!) {
      search(queryString: $q, entityNames: ["target"], page: {size: 1, index: 0}) {
        hits { id }
      }
    }
    """
    data = await _graphql(query, {"q": gene_symbol})
    hits = data.get("search", {}).get("hits", [])
    return hits[0]["id"] if hits else None


async def get_target_info(ensembl_id: str) -> dict:
    """Get target information including tractability and safety."""
    query = """
    query TargetInfo($id: String!) {
      target(ensemblId: $id) {
        id
        approvedSymbol
        approvedName
        biotype
        tractability {
          label
          modality
          value
        }
        safetyLiabilities {
          event
          datasource
          url
        }
      }
    }
    """
    data = await _graphql(query, {"id": ensembl_id})
    return data.get("target", {})


async def get_disease_associations(ensembl_id: str, top_n: int = 10) -> list[dict]:
    """Get top disease associations for a target."""
    query = """
    query DiseaseAssoc($id: String!) {
      target(ensemblId: $id) {
        associatedDiseases(page: {size: 10, index: 0}) {
          rows {
            disease { id name }
            score
          }
        }
      }
    }
    """
    data = await _graphql(query, {"id": ensembl_id})
    rows = data.get("target", {}).get("associatedDiseases", {}).get("rows", [])

    results = []
    for row in rows[:top_n]:
        disease = row.get("disease", {})
        results.append({
            "disease_id": disease.get("id", ""),
            "disease_name": disease.get("name", ""),
            "overall_score": row.get("score", 0),
        })
    return results


async def get_known_drugs(ensembl_id: str) -> list[dict]:
    """Get known drugs targeting this gene via drug search."""
    # First get the gene symbol
    info = await get_target_info(ensembl_id)
    gene_symbol = info.get("approvedSymbol", "")
    if not gene_symbol:
        return []

    # Search for drugs mentioning this gene
    query = """
    query SearchDrugs($q: String!) {
      search(queryString: $q, entityNames: ["drug"], page: {size: 10, index: 0}) {
        hits { id name entity }
      }
    }
    """
    data = await _graphql(query, {"q": gene_symbol})
    hits = data.get("search", {}).get("hits", [])

    drugs = []
    for hit in hits:
        if hit.get("entity") == "drug":
            drugs.append({
                "drug_id": hit.get("id", ""),
                "name": hit.get("name", ""),
                "type": "",
                "mechanism": "",
                "max_phase": 0,  # would need separate drug query
                "status": "unknown",
                "indication": "",
            })
    return drugs


async def assess_target(gene_symbol: str) -> dict:
    """Full Open Targets assessment for a gene.

    Returns: ensembl_id, tractability, safety, diseases, drugs
    """
    ensembl_id = await search_target(gene_symbol)
    if not ensembl_id:
        return {"gene": gene_symbol, "found": False}

    # Parallel fetch
    info, diseases, drugs = await asyncio.gather(
        get_target_info(ensembl_id),
        get_disease_associations(ensembl_id, top_n=5),
        get_known_drugs(ensembl_id),
    )

    # Parse tractability
    tractability = {}
    for t in info.get("tractability", []):
        modality = t.get("modality", "")
        if t.get("value", False):
            tractability[modality] = tractability.get(modality, [])
            tractability[modality].append(t.get("label", ""))

    return {
        "gene": gene_symbol,
        "found": True,
        "ensembl_id": ensembl_id,
        "approved_name": info.get("approvedName", ""),
        "tractability": tractability,
        "safety_signals": len(info.get("safetyLiabilities", [])),
        "top_diseases": diseases,
        "known_drugs": drugs,
        "n_drugs": len(drugs),
        "max_drug_phase": max((d.get("max_phase", 0) for d in drugs), default=0),
    }
