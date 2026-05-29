"""Common dataclasses for backend I/O.

These are intentionally backend-agnostic. Each concrete backend translates
them into its own native format (PDBQT for Vina, mmCIF for Boltz, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReceptorSpec:
    pdb_path: Path | str | None = None
    sequence: str | None = None
    chain_id: str | None = None
    # Pre-prepared formats (filled by backend if it caches conversions)
    pdbqt_path: Path | str | None = None
    mmcif_path: Path | str | None = None

    def __post_init__(self) -> None:
        if self.pdb_path is None and self.sequence is None and self.pdbqt_path is None:
            raise ValueError("ReceptorSpec needs pdb_path, pdbqt_path, or sequence")


@dataclass
class LigandSpec:
    smiles: str | None = None
    sdf_path: Path | str | None = None
    pdb_path: Path | str | None = None
    pdbqt_path: Path | str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not any([self.smiles, self.sdf_path, self.pdb_path, self.pdbqt_path]):
            raise ValueError("LigandSpec needs smiles, sdf_path, pdb_path, or pdbqt_path")


@dataclass
class BoxSpec:
    """Search box for site-specific docking. Optional for blind backends."""
    center: tuple[float, float, float]
    size: tuple[float, float, float] = (20.0, 20.0, 20.0)

    @classmethod
    def from_dict(cls, d: dict) -> BoxSpec:
        return cls(
            center=(d["center_x"], d["center_y"], d["center_z"]),
            size=(d.get("size_x", 20.0), d.get("size_y", 20.0), d.get("size_z", 20.0)),
        )


@dataclass
class DockResult:
    """Unified output across all backends.

    Affinity convention: kcal/mol, more negative = better binding.
    confidence is in [0, 1] when the backend provides one (e.g., Boltz-2,
    DiffDock-L); None when the backend has no native confidence.
    """
    backend: str
    affinity_top: float | None
    affinity_all: list[float] = field(default_factory=list)
    pose_path: Path | str | None = None
    confidence: float | None = None
    confidence_per_pose: list[float] | None = None
    smiles: str | None = None
    ligand_name: str | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.affinity_top is not None
