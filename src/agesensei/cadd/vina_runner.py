"""AutoDock Vina runner (compatibility shim over affinity_ensemble).

This module is now a thin re-export layer. The implementation moved to
`affinity_ensemble.backends.vina` so that other backends (Uni-Dock, Boltz-2,
DiffDock-L) can plug into the same abstraction.

Existing scripts that did:
    from agesensei.cadd.vina_runner import prepare_ligand_pdbqt, dock_one
    ...
keep working unchanged. New code should prefer:
    from affinity_ensemble import get_backend, ReceptorSpec, LigandSpec, BoxSpec
    backend = get_backend("vina")
    result = backend.dock(receptor, ligand, box, work_dir)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


from affinity_ensemble import (  # noqa: E402
    BoxSpec,
    LigandSpec,
    ReceptorSpec,
    get_backend,
)
from affinity_ensemble.backends.vina import (  # noqa: E402
    pose_rmsd_from_pdbqt,
    prepare_ligand_pdbqt,
    prepare_ligand_pdbqt_from_pdb,
    prepare_receptor_pdbqt,
    smiles_to_3d_mol,
)


@dataclass
class DockResult:
    """Legacy DockResult kept for backwards compat with w2_day*.py callers.

    New code should use affinity_ensemble.types.DockResult, which carries
    backend name, confidence, etc.
    """
    smiles: str
    affinity_top: float | None
    affinity_all: list[float]
    pose_pdbqt: str | None
    error: str | None = None


def dock_one(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    box_config: dict,
    pose_pdbqt_out: Path,
    exhaustiveness: int = 8,
    n_poses: int = 9,
    cpu: int = 4,
) -> DockResult:
    """Legacy wrapper. Internally calls VinaBackend.dock with pre-prepped inputs."""
    backend = get_backend("vina", cpu=cpu)
    receptor = ReceptorSpec(pdbqt_path=Path(receptor_pdbqt))
    ligand = LigandSpec(pdbqt_path=Path(ligand_pdbqt))
    box = BoxSpec.from_dict(box_config)
    work_dir = Path(pose_pdbqt_out).parent

    result = backend.dock(
        receptor, ligand, box, work_dir,
        exhaustiveness=exhaustiveness, n_poses=n_poses,
    )

    # Legacy callers expect poses written to the user-supplied path. The
    # backend writes to work_dir/<name>_poses.pdbqt — copy/rename if needed.
    if result.pose_path is not None and Path(result.pose_path) != Path(pose_pdbqt_out):
        Path(pose_pdbqt_out).write_bytes(Path(result.pose_path).read_bytes())

    return DockResult(
        smiles=result.smiles or "",
        affinity_top=result.affinity_top,
        affinity_all=result.affinity_all,
        pose_pdbqt=str(pose_pdbqt_out) if result.pose_path else None,
        error=result.error,
    )


__all__ = [
    "DockResult",
    "dock_one",
    "pose_rmsd_from_pdbqt",
    "prepare_ligand_pdbqt",
    "prepare_ligand_pdbqt_from_pdb",
    "prepare_receptor_pdbqt",
    "smiles_to_3d_mol",
]
