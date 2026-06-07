"""CADD Agent — virtual screening via docking + QSAR filtering.

Wraps the cadd/ toolkit into an agent that the Orchestrator can call
as a first-class pipeline step.  Given a set of targets (with predicted
structures), the agent:

    1. Fetches candidate compounds from ChEMBL for each target
    2. Runs molecular docking (AutoDock Vina / affinity_ensemble backends)
    3. Computes QSAR descriptors + Lipinski drug-likeness filter
    4. Ranks and returns top hits per target
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agesensei.schema import StructurePrediction, Target


@dataclass
class DockingHit:
    """A single compound hit from virtual screening."""

    smiles: str
    molecule_chembl_id: str = ""
    affinity_kcal: float | None = None  # Vina binding energy (kcal/mol)
    pchembl_value: float | None = None  # experimental pChEMBL (if known)
    qed: float = 0.0  # quantitative estimate of drug-likeness
    mw: float = 0.0
    logp: float = 0.0
    lipinski_pass: bool = True
    pose_path: str = ""  # path to docked pose file
    error: str | None = None


@dataclass
class ScreeningResult:
    """Virtual screening result for one target."""

    gene_symbol: str
    chembl_target_id: str = ""
    compounds_fetched: int = 0
    compounds_docked: int = 0
    compounds_passed_filter: int = 0
    top_hits: list[DockingHit] = field(default_factory=list)
    error: str | None = None


class CADDAgent:
    """Virtual screening agent: ChEMBL fetch -> docking -> QSAR filter.

    Designed to run after StructurePredictAgent provides 3D structures, so
    structure-based docking has a receptor to dock against.

    Attributes:
        work_dir: Base directory for docking intermediates.
        exhaustiveness: Vina search exhaustiveness (higher = slower + better).
        top_k: Number of top hits to keep per target.
        max_compounds: Max compounds to fetch from ChEMBL per target.
    """

    def __init__(
        self,
        work_dir: Path | None = None,
        exhaustiveness: int = 8,
        top_k: int = 10,
        max_compounds: int = 200,
    ):
        self.work_dir = work_dir or Path("./artifacts/cadd")
        self.exhaustiveness = exhaustiveness
        self.top_k = top_k
        self.max_compounds = max_compounds

    @property
    def available(self) -> bool:
        """Check if CADD dependencies (RDKit, Vina) are importable."""
        try:
            from rdkit import Chem  # noqa: F401
            return True
        except ImportError:
            return False

    async def screen_target(
        self,
        gene_symbol: str,
        chembl_target_id: str,
        structure: StructurePrediction | None = None,
    ) -> ScreeningResult:
        """Run virtual screening for a single target.

        Args:
            gene_symbol: Target gene symbol.
            chembl_target_id: ChEMBL target ID for compound retrieval.
            structure: Predicted 3D structure (for structure-based docking).

        Returns:
            ScreeningResult with ranked hits.
        """
        if not self.available:
            return ScreeningResult(
                gene_symbol=gene_symbol,
                chembl_target_id=chembl_target_id,
                error="CADD deps not installed. Install: pip install agesensei[cadd]",
            )

        try:
            from agesensei.cadd.chembl_fetch import fetch_activities, dedup_by_molecule
            from agesensei.cadd.chem_utils import (
                smiles_to_mol,
                compute_descriptors,
                passes_lipinski,
            )
        except ImportError as e:
            return ScreeningResult(
                gene_symbol=gene_symbol,
                chembl_target_id=chembl_target_id,
                error=f"Import error: {e}",
            )

        # Step 1: Fetch compounds from ChEMBL
        try:
            df = fetch_activities(
                chembl_target_id,
                limit=self.max_compounds,
            )
            df = dedup_by_molecule(df)
            n_fetched = len(df)
        except Exception as e:
            return ScreeningResult(
                gene_symbol=gene_symbol,
                chembl_target_id=chembl_target_id,
                error=f"ChEMBL fetch failed: {e}",
            )

        if n_fetched == 0:
            return ScreeningResult(
                gene_symbol=gene_symbol,
                chembl_target_id=chembl_target_id,
                compounds_fetched=0,
                error="No compounds found in ChEMBL",
            )

        # Step 2: QSAR descriptors + Lipinski filter
        hits: list[DockingHit] = []
        for _, row in df.iterrows():
            smi = row["smiles"]
            mol = smiles_to_mol(smi)
            if mol is None:
                continue

            desc = compute_descriptors(mol)
            if desc is None:
                continue

            lip = passes_lipinski(desc, strict=False)
            hit = DockingHit(
                smiles=smi,
                molecule_chembl_id=row.get("molecule_chembl_id", ""),
                pchembl_value=row.get("pchembl_value"),
                qed=desc.qed,
                mw=desc.mw,
                logp=desc.logp,
                lipinski_pass=lip,
            )
            hits.append(hit)

        passed = [h for h in hits if h.lipinski_pass]

        # Step 3: Docking (if structure is available and Vina is installed)
        n_docked = 0
        if structure and structure.cif_path and not structure.error:
            n_docked = await self._dock_hits(
                gene_symbol, structure, passed
            )

        # Step 4: Rank by best available metric
        #   - If docked: sort by affinity (more negative = better)
        #   - Else: sort by pChEMBL value (higher = more potent)
        if n_docked > 0:
            passed.sort(
                key=lambda h: h.affinity_kcal if h.affinity_kcal is not None else 0.0
            )
        else:
            passed.sort(
                key=lambda h: -(h.pchembl_value or 0.0)
            )

        top = passed[: self.top_k]

        return ScreeningResult(
            gene_symbol=gene_symbol,
            chembl_target_id=chembl_target_id,
            compounds_fetched=n_fetched,
            compounds_docked=n_docked,
            compounds_passed_filter=len(passed),
            top_hits=top,
        )

    async def _dock_hits(
        self,
        gene_symbol: str,
        structure: StructurePrediction,
        hits: list[DockingHit],
    ) -> int:
        """Attempt docking against the predicted structure. Returns count docked."""
        try:
            from agesensei.cadd.vina_runner import (
                prepare_receptor_pdbqt,
                prepare_ligand_pdbqt,
                dock_one,
            )
        except ImportError:
            return 0

        target_dir = self.work_dir / gene_symbol.lower()
        target_dir.mkdir(parents=True, exist_ok=True)

        # Prepare receptor (CIF -> PDBQT)
        try:
            receptor_pdbqt = prepare_receptor_pdbqt(
                Path(structure.cif_path), target_dir / "receptor.pdbqt"
            )
        except Exception:
            return 0

        # Simple box centered on the protein (fallback)
        box_config = {
            "center_x": 0.0, "center_y": 0.0, "center_z": 0.0,
            "size_x": 30.0, "size_y": 30.0, "size_z": 30.0,
        }

        n_docked = 0
        for hit in hits:
            try:
                lig_pdbqt = prepare_ligand_pdbqt(
                    hit.smiles, target_dir / f"lig_{n_docked}.pdbqt"
                )
                pose_out = target_dir / f"pose_{n_docked}.pdbqt"
                result = dock_one(
                    receptor_pdbqt, lig_pdbqt, box_config, pose_out,
                    exhaustiveness=self.exhaustiveness,
                )
                if result.error is None:
                    hit.affinity_kcal = result.affinity_top
                    hit.pose_path = str(pose_out)
                    n_docked += 1
            except Exception:
                continue

        return n_docked

    async def screen_targets(
        self,
        targets: list[Target],
        structures: dict[str, StructurePrediction] | None = None,
        top_n: int = 5,
    ) -> dict[str, ScreeningResult]:
        """Run virtual screening for top N discovery targets.

        Args:
            targets: Ranked target list from the pipeline.
            structures: Optional dict of gene_symbol -> StructurePrediction
                        (enables structure-based docking).
            top_n: Number of top targets to screen.

        Returns:
            Dict mapping gene_symbol -> ScreeningResult.
        """
        from agesensei.cadd.targets import SENOLYTIC_TARGETS

        results = {}
        structures = structures or {}

        for target in targets[:top_n]:
            gene = target.gene_symbol
            # Resolve ChEMBL target ID
            target_info = SENOLYTIC_TARGETS.get(gene, {})
            chembl_id = target_info.get("chembl_id", "")

            if not chembl_id:
                print(f"    {gene}: no ChEMBL target ID, skipping CADD")
                results[gene] = ScreeningResult(
                    gene_symbol=gene,
                    error="No ChEMBL target ID mapped",
                )
                continue

            print(f"    Screening {gene} ({chembl_id})...")
            struct = structures.get(gene)
            result = await self.screen_target(gene, chembl_id, structure=struct)
            results[gene] = result

            if result.error:
                print(f"      Warning: {result.error}")
            else:
                print(
                    f"      Fetched {result.compounds_fetched} compounds, "
                    f"{result.compounds_passed_filter} passed filter, "
                    f"docked {result.compounds_docked}, "
                    f"top-{len(result.top_hits)} hits"
                )

        return results
