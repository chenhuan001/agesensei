"""Protein analysis agent: UniProt annotation + ESM-2 embeddings + mutation effects.

Integrates:
    - UniProt: protein sequence, function, GO terms, PDB structures
    - ESM-2: embeddings, zero-shot mutation scanning, sensitivity profiling
"""

from __future__ import annotations

import asyncio
from agesensei.schema import Target, ProteinAnalysis
from agesensei.tools import uniprot


class ProteinAnalyzer:
    """Analyze target proteins using UniProt + ESM-2.

    Pipeline:
        1. Fetch protein sequence + annotations from UniProt
        2. Compute ESM-2 embeddings
        3. Run mutation sensitivity profiling
        4. Identify functionally important positions

    Example:
        analyzer = ProteinAnalyzer()
        result = await analyzer.analyze("TP53")
    """

    def __init__(self, esm_model: str | None = None, skip_esm: bool = False):
        """
        Args:
            esm_model: ESM-2 model name (None = use config default)
            skip_esm: If True, skip ESM-2 analysis (UniProt only)
        """
        self.skip_esm = skip_esm
        self._esm = None
        self._esm_model = esm_model

    @property
    def esm(self):
        if self._esm is None:
            from agesensei.models.esm import ESMWrapper
            self._esm = ESMWrapper(model_name=self._esm_model)
        return self._esm

    async def analyze(self, gene_symbol: str, scan_positions: int = 50) -> ProteinAnalysis:
        """Full analysis pipeline for a target protein.

        Args:
            gene_symbol: Gene symbol (e.g., "TP53")
            scan_positions: Number of positions to mutation-scan (0 = skip)

        Returns:
            ProteinAnalysis with sequence, embeddings, and mutation data
        """
        print(f"  Analyzing {gene_symbol}...")

        # Step 1: Fetch from UniProt
        accession = await uniprot.search_by_gene(gene_symbol)
        if not accession:
            print(f"    {gene_symbol}: not found in UniProt")
            return ProteinAnalysis(gene_symbol=gene_symbol)

        protein = await uniprot.fetch_protein(accession)
        sequence = protein.get("sequence", "")

        result = ProteinAnalysis(
            gene_symbol=gene_symbol,
            uniprot_id=accession,
            sequence=sequence,
            sequence_length=len(sequence),
        )

        print(f"    UniProt: {accession} | {protein['name'][:40]} | {len(sequence)} aa | {len(protein['pdb_ids'])} PDB")

        if self.skip_esm or not sequence:
            return result

        # Step 2: ESM-2 embedding
        # Truncate very long sequences for memory
        max_len = 1022
        seq_for_esm = sequence[:max_len]

        emb = self.esm.embed(seq_for_esm)
        result.embedding_dim = emb["mean"].shape[0]
        result.mean_embedding = emb["mean"].tolist()
        print(f"    ESM-2: embedding dim={result.embedding_dim}")

        # Step 3: Mutation sensitivity profiling
        if scan_positions > 0:
            # Scan evenly spaced positions across the sequence
            n_pos = min(scan_positions, len(seq_for_esm))
            if n_pos < len(seq_for_esm):
                step = len(seq_for_esm) // n_pos
                positions = list(range(0, len(seq_for_esm), step))[:n_pos]
            else:
                positions = list(range(len(seq_for_esm)))

            # Direct mutation_scan on selected positions only (fast)
            mutations = self.esm.mutation_scan(seq_for_esm, positions=positions)

            # Aggregate per position
            by_pos: dict[int, list[float]] = {}
            for m in mutations:
                pos = m["position"]
                if pos not in by_pos:
                    by_pos[pos] = []
                by_pos[pos].append(m["score"])

            profile = []
            for pos in sorted(by_pos.keys()):
                scores = by_pos[pos]
                profile.append({
                    "position": pos,
                    "residue": seq_for_esm[pos],
                    "mean_score": round(sum(scores) / len(scores), 4),
                    "min_score": round(min(scores), 4),
                })

            # Sort by mean_score (most sensitive first)
            profile.sort(key=lambda x: x["mean_score"])
            result.sensitive_positions = profile[:20]
            result.conservation_scores = [p["mean_score"] for p in profile]

            n_sensitive = sum(1 for p in profile if p["mean_score"] < -2.0)
            print(f"    Mutation scan: {len(positions)} positions, {n_sensitive} highly sensitive")

        return result

    async def analyze_targets(self, targets: list[Target], top_n: int = 5,
                              scan_positions: int = 30) -> dict[str, ProteinAnalysis]:
        """Analyze top N targets from extraction results.

        Args:
            targets: List of Target objects (sorted by score)
            top_n: Number of top targets to analyze
            scan_positions: Positions to mutation-scan per protein

        Returns:
            Dict mapping gene_symbol -> ProteinAnalysis
        """
        top_targets = targets[:top_n]
        print(f"\nAnalyzing top {len(top_targets)} targets with ESM-2...")

        results = {}
        for target in top_targets:
            try:
                analysis = await self.analyze(target.gene_symbol, scan_positions=scan_positions)
                results[target.gene_symbol] = analysis
            except Exception as e:
                print(f"    {target.gene_symbol}: analysis failed - {e}")

        print(f"  Completed: {len(results)}/{len(top_targets)} proteins analyzed")
        return results
