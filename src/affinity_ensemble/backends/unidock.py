"""Uni-Dock backend stub.

Uni-Dock is the GPU-accelerated reimplementation of AutoDock Vina from
DP Technology. Same scoring function, ~10-100x speedup on large libraries.
Reference: https://github.com/dptech-corp/Uni-Dock

Status (2026-05-08): NOT YET IMPLEMENTED. Methods raise NotImplementedError
with install hints. Planned for Week 11.
"""
from __future__ import annotations

from pathlib import Path

from affinity_ensemble.backend import DockingBackend
from affinity_ensemble.types import BoxSpec, DockResult, LigandSpec, ReceptorSpec

_INSTALL_HINT = (
    "Uni-Dock backend not yet implemented. To prepare: build from source at "
    "https://github.com/dptech-corp/Uni-Dock (requires CUDA 11.6+). The CLI "
    "binary is `unidock`. Implementation tracked in Week 11."
)


class UniDockBackend(DockingBackend):
    name = "unidock"

    @property
    def supports_blind_docking(self) -> bool:
        return False

    @property
    def returns_native_affinity(self) -> bool:
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

    def dock_batch(
        self,
        receptor: ReceptorSpec,
        ligands: list[LigandSpec],
        box: BoxSpec | None,
        work_dir: Path,
        **kwargs,
    ) -> list[DockResult]:
        raise NotImplementedError(_INSTALL_HINT)
