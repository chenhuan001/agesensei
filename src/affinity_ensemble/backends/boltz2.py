"""Boltz-2 backend stub.

Boltz-2 is an AlphaFold-3-style structure predictor with a binding affinity
head, achieving correlation approaching FEP+ on standard benchmarks while
being orders of magnitude faster. Reference:
https://github.com/jwohlwend/boltz

Status (2026-05-08): NOT YET IMPLEMENTED. Methods raise NotImplementedError
with install hints. Planned for Week 11-12.

Notable design points (for future implementer):
- Boltz-2 takes the full receptor sequence (no PDBQT required) plus ligand
  SMILES. Mapping ReceptorSpec.sequence -> Boltz input is straightforward.
- Output includes both predicted complex structure AND binding affinity in
  pKd units, plus a per-prediction confidence (pLDDT-style).
- ~30s per ligand on a single H100; batch by stacking multiple ligands per
  receptor and reusing the receptor MSA.
"""
from __future__ import annotations

from pathlib import Path

from affinity_ensemble.backend import DockingBackend
from affinity_ensemble.types import BoxSpec, DockResult, LigandSpec, ReceptorSpec

_INSTALL_HINT = (
    "Boltz-2 backend not yet implemented. To prepare: pip install boltz "
    "(needs torch>=2.2 + CUDA 12). See https://github.com/jwohlwend/boltz . "
    "Implementation tracked in Week 11-12."
)


class Boltz2Backend(DockingBackend):
    name = "boltz2"

    @property
    def supports_blind_docking(self) -> bool:
        return True  # Boltz-2 is site-agnostic

    @property
    def returns_native_affinity(self) -> bool:
        return True

    @property
    def returns_confidence(self) -> bool:
        return True  # native pLDDT-style confidence head

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
