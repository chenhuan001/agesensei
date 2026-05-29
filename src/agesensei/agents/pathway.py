"""Pathway analysis agent: KEGG enrichment for aging-related pathways."""

from agesensei.schema import Target, PathwayInfo
from agesensei.tools import kegg


class PathwayAgent:
    """Analyze pathway context of discovered targets.

    Uses KEGG to identify which aging-related pathways are enriched
    in the set of extracted targets.

    Example:
        agent = PathwayAgent()
        pathways = await agent.analyze(targets)
    """

    async def analyze(self, targets: list[Target], top_n: int = 10) -> list[PathwayInfo]:
        """Pathway enrichment analysis for a set of targets.

        Args:
            targets: List of Target objects
            top_n: Number of top targets to analyze

        Returns:
            List of PathwayInfo objects, sorted by hit count
        """
        gene_symbols = [t.gene_symbol for t in targets[:top_n]]
        print(f"\nPathway analysis for {len(gene_symbols)} genes: {', '.join(gene_symbols)}")

        # Get aging pathway enrichment
        enriched = await kegg.get_aging_pathway_enrichment(gene_symbols)

        # Convert to PathwayInfo objects
        pathways = []
        for pw in enriched:
            pathways.append(PathwayInfo(
                pathway_id=pw["pathway_id"],
                pathway_name=pw["pathway_name"],
                source="KEGG",
                genes_in_pathway=pw["genes"],
                aging_relevance=f"Aging-related pathway with {pw['hit_count']}/{pw['total_genes_in_set']} target genes",
            ))

        if pathways:
            print(f"  Found {len(pathways)} aging-related pathways:")
            for pw in pathways:
                print(f"    {pw.pathway_id} {pw.pathway_name}: {len(pw.genes_in_pathway)} genes ({', '.join(pw.genes_in_pathway)})")
        else:
            print("  No aging-related pathways found")

        return pathways
