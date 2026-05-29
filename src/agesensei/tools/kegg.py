"""KEGG REST API wrapper for pathway analysis.

API: https://rest.kegg.jp/
Free, no key needed. Simple text-based API.
"""

import asyncio
import httpx

KEGG_API = "https://rest.kegg.jp"
MAX_RETRIES = 3

# Aging-related KEGG pathways (curated)
AGING_PATHWAYS = {
    "hsa04210": "Apoptosis",
    "hsa04115": "p53 signaling pathway",
    "hsa04150": "mTOR signaling pathway",
    "hsa04152": "AMPK signaling pathway",
    "hsa04068": "FoxO signaling pathway",
    "hsa04140": "Autophagy - animal",
    "hsa04217": "Necroptosis",
    "hsa04110": "Cell cycle",
    "hsa04218": "Cellular senescence",
    "hsa04630": "JAK-STAT signaling pathway",
    "hsa04010": "MAPK signaling pathway",
    "hsa04151": "PI3K-Akt signaling pathway",
    "hsa04310": "Wnt signaling pathway",
    "hsa04350": "TGF-beta signaling pathway",
    "hsa04064": "NF-kappa B signaling pathway",
    "hsa04066": "HIF-1 signaling pathway",
    "hsa03410": "Base excision repair",
    "hsa03420": "Nucleotide excision repair",
}


async def _get_text(url: str) -> str:
    """GET request returning text with retry."""
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                if response.status_code == 404:
                    return ""
                response.raise_for_status()
                return response.text
        except (httpx.ConnectError, httpx.ReadTimeout):
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
    return ""


async def map_gene_to_kegg(gene_symbol: str) -> str | None:
    """Map gene symbol to KEGG gene ID (e.g., "hsa:596" for BCL2).

    Strategy: Use UniProt to get Entrez Gene ID, then construct hsa:ID.
    """
    # Use UniProt to get Entrez Gene ID
    try:
        from agesensei.tools.hagr import load_genage
        genage = load_genage()
        match = genage[genage['symbol'].str.upper() == gene_symbol.upper()]
        if not match.empty:
            entrez_id = match.iloc[0].get('entrez_gene_id')
            if entrez_id and str(entrez_id) != 'nan':
                return f"hsa:{int(entrez_id)}"
    except Exception:
        pass

    # Fallback: try KEGG find API
    text = await _get_text(f"{KEGG_API}/find/hsa/{gene_symbol}")
    for line in text.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].startswith("hsa:"):
            # Check if gene symbol is in the description
            desc = parts[1]
            if gene_symbol.upper() in desc.upper().split(";")[0]:
                return parts[0]

    # Last fallback: return first result
    for line in text.strip().split("\n"):
        if line.startswith("hsa:"):
            return line.split("\t")[0]

    return None


async def get_pathways_for_gene(gene_symbol: str) -> list[dict]:
    """Get KEGG pathways containing this gene.

    Returns list of: pathway_id, pathway_name, is_aging_related
    """
    kegg_id = await map_gene_to_kegg(gene_symbol)
    if not kegg_id:
        return []

    text = await _get_text(f"{KEGG_API}/link/pathway/{kegg_id}")

    pathways = []
    seen = set()
    for line in text.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            pathway_id = parts[1].replace("path:", "")
            if pathway_id in seen or not pathway_id.startswith("hsa"):
                continue
            seen.add(pathway_id)

            # Use local dict for name (avoid extra API calls)
            name = AGING_PATHWAYS.get(pathway_id, pathway_id)

            pathways.append({
                "pathway_id": pathway_id,
                "pathway_name": name,
                "is_aging_related": pathway_id in AGING_PATHWAYS,
            })

    return pathways


async def get_aging_pathway_enrichment(gene_symbols: list[str]) -> list[dict]:
    """Check which aging-related KEGG pathways are enriched in a gene set.

    Args:
        gene_symbols: List of gene symbols to analyze

    Returns:
        List of aging pathways with hit count and gene list
    """
    # Get pathways for each gene in parallel
    tasks = [get_pathways_for_gene(g) for g in gene_symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count pathway hits
    pathway_genes: dict[str, list[str]] = {}
    pathway_names: dict[str, str] = {}

    for gene, gene_pathways in zip(gene_symbols, results):
        if isinstance(gene_pathways, Exception):
            continue
        for pw in gene_pathways:
            pid = pw["pathway_id"]
            if pid in AGING_PATHWAYS:
                if pid not in pathway_genes:
                    pathway_genes[pid] = []
                    pathway_names[pid] = pw["pathway_name"]
                pathway_genes[pid].append(gene)

    # Sort by hit count
    enriched = []
    for pid, genes in sorted(pathway_genes.items(), key=lambda x: len(x[1]), reverse=True):
        enriched.append({
            "pathway_id": pid,
            "pathway_name": pathway_names[pid],
            "hit_count": len(genes),
            "genes": genes,
            "total_genes_in_set": len(gene_symbols),
        })

    return enriched
