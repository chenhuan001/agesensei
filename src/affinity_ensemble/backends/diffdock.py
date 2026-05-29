"""DiffDock-L backend stub.

DiffDock-L is a diffusion-model-based blind docking method. It generates
poses but does NOT return a native binding affinity - it has a separate
confidence head that scores pose plausibility. We use it primarily as a
pose-quality verifier in the L2 layer of the ensemble.
Reference: https://github.com/gcorso/DiffDock

Status (2026-05-08): NOT YET IMPLEMENTED. Planned for Week 12.
"""
from __future__ import annotations

from pathlib import Path

from affinity_ensemble.backend import DockingBackend
from affinity_ensemble.types import BoxSpec, DockResult, LigandSpec, ReceptorSpec

_INSTALL_HINT = (
    "DiffDock-L backend not yet implemented. Clone https://github.com/gcorso/DiffDock "
    "and follow their setup. Implementation tracked in Week 12."
)


class DiffDockBackend(DockingBackend):
    name = "diffdock"

    @property
    def supports_blind_docking(self) -> bool:
        return True

    @property
    def returns_native_affinity(self) -> bool:
        return False  # poses + confidence only, no dG estimate

    @property
    def returns_confidence(self) -> bool:
        return True

    def prepare_receptor(self, receptor: ReceptorSpec, work_dir: Path) -> ReceptorSpec:
        raise NotImplementedError(_INSTALL_HINT)

    def dock(
        self,
        receptor: ReceptorSpec,
        ligand: LigandSpec,
        box: BoxSpec | None,
        work_dir: Path,
        **kwargs,
    ) -> DockResult:
        raise NotImplementedError(_INSTALL_HINT)
