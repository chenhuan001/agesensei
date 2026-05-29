"""Structure Predictor Agent — 3D protein structure prediction using Protenix.

Predicts protein structures using ByteDance's Protenix (AlphaFold3-class, 464M params).
Integrates with the discovery pipeline to provide structural context for drug targets.

Pipeline:
    1. Fetch sequence from UniProt (or accept directly)
    2. Run Protenix structure prediction
    3. Parse confidence metrics (pLDDT, pTM, ipTM)
    4. Optionally feed structure into CADD docking pipeline
"""
from __future__ import annotations

from pathlib import Path

from agesensei.schema import StructurePrediction, Target
from agesensei.tools.protenix import (
    ProtenixResult,
    check_protenix_available,
    predict_structure,
)
from agesensei.tools.uniprot import get_sequence


class StructurePredictor:
    """Predict 3D protein structures using Protenix (AlphaFold3-class).

    Attributes:
        model: Protenix model checkpoint name.
        output_dir: Base directory for structure outputs.
        timeout: Max seconds per prediction.
    """

    def __init__(
        self,
        model: str = "protenix_base_default_v1.0.0",
        output_dir: Path | None = None,
        timeout: int = 600,
    ):
        self.model = model
        self.output_dir = output_dir or Path("./artifacts/structures")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        """Check if Protenix is installed."""
        return check_protenix_available()

    async def predict(
        self,
        gene_symbol: str,
        sequence: str | None = None,
    ) -> StructurePrediction:
        """Predict structure for a single protein.

        Args:
            gene_symbol: Gene symbol (e.g., "BCL2L1").
            sequence: Protein sequence. If None, fetches from UniProt.

        Returns:
            StructurePrediction with confidence metrics and CIF path.
        """
        # Fetch sequence if not provided
        if not sequence:
            _, sequence = await get_sequence(gene_symbol)
            if not sequence:
                return StructurePrediction(
                    gene_symbol=gene_symbol,
                    error=f"Could not fetch sequence for {gene_symbol}",
                )

        # Check availability
        if not self.available:
            return StructurePrediction(
                gene_symbol=gene_symbol,
                sequence=sequence,
                num_residues=len(sequence),
                error="Protenix not installed. Install with: pip install protenix",
            )

        # Run prediction
        out_dir = self.output_dir / gene_symbol.lower()
        out_dir.mkdir(parents=True, exist_ok=True)

        result: ProtenixResult = await predict_structure(
            sequences=[{"name": gene_symbol, "sequence": sequence}],
            output_dir=out_dir,
            model=self.model,
            timeout=self.timeout,
        )

        return StructurePrediction(
            gene_symbol=gene_symbol,
            sequence=sequence,
            model_used=result.model_used,
            cif_path=result.cif_path,
            plddt_mean=result.plddt_mean,
            ptm=result.ptm,
            iptm=result.iptm,
            num_residues=result.num_residues or len(sequence),
            prediction_time_sec=result.prediction_time_sec,
            error=result.error,
        )

    async def predict_targets(
        self,
        targets: list[Target],
        top_n: int = 5,
    ) -> dict[str, StructurePrediction]:
        """Predict structures for top N discovery targets.

        Args:
            targets: Ranked list of Target objects from discovery pipeline.
            top_n: Number of top targets to predict structures for.

        Returns:
            Dict mapping gene_symbol -> StructurePrediction.
        """
        results = {}
        selected = targets[:top_n]

        for target in selected:
            print(f"    Predicting structure: {target.gene_symbol}...")
            pred = await self.predict(target.gene_symbol)
            results[target.gene_symbol] = pred

            if pred.error:
                print(f"      Warning: {pred.error}")
            else:
                print(
                    f"      pLDDT={pred.plddt_mean:.1f}  "
                    f"pTM={pred.ptm:.3f}  "
                    f"time={pred.prediction_time_sec:.1f}s"
                )

        return results
