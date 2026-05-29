"""STRING database API wrapper for protein-protein interaction networks.

STRING: https://string-db.org/
API docs: https://string-db.org/help/api/
No API key needed. Rate limit: 600 requests per hour.
"""

import httpx

STRING_API = "https://string-db.org/api"


async def get_interaction_partners(
    gene_symbol: str, species: int = 9606, score_threshold: int = 400
) -> list[dict]:
    """Get protein-protein interaction partners from STRING.

    Args:
        gene_symbol: Gene symbol (e.g. "TP53")
        species: NCBI taxonomy ID (9606 = human)
        score_threshold: Minimum combined score (0-1000)

    Returns:
        List of dicts: {partner, combined_score, experimental, database, textmining}
    """
    # TODO: implement
    raise NotImplementedError


async def get_interaction_network(
    gene_symbols: list[str], species: int = 9606
) -> dict:
    """Get interaction network for a set of genes.

    Returns:
        Dict with nodes, edges, and network statistics
    """
    raise NotImplementedError


async def get_enrichment(gene_symbols: list[str], species: int = 9606) -> list[dict]:
    """Get functional enrichment (GO, KEGG, Pfam) for a gene set."""
    raise NotImplementedError
