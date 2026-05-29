"""Abstract base class for docking backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from affinity_ensemble.types import BoxSpec, DockResult, LigandSpec, ReceptorSpec


class DockingBackend(ABC):
    """Common interface every backend must implement.

    Backend instances are intended to be cheap to construct and stateful per
    receptor — call `prepare_receptor` once, then `dock` many times against
    the same prepared receptor for batch screening.
    """

    name: str = "abstract"

    # ---- capability flags (override in subclasses) ----

    @property
    def supports_blind_docking(self) -> bool:
        """True if backend works without an explicit BoxSpec."""
        return False

    @property
    def returns_native_affinity(self) -> bool:
        """True if backend outputs a binding free energy / score directly.

        DiffDock-L returns poses but no affinity, so this is False there.
        """
        return True

    @property
    def returns_confidence(self) -> bool:
        """True if backend exposes a per-prediction confidence."""
        return False

    # ---- lifecycle ----

    @abstractmethod
    def prepare_receptor(self, receptor: ReceptorSpec, work_dir: Path) -> ReceptorSpec:
        """Return a ReceptorSpec with backend-native fields (e.g., pdbqt_path) filled."""

    @abstractmethod
    def dock(
        self,
        receptor: ReceptorSpec,
        ligand: LigandSpec,
        box: BoxSpec | None,
        work_dir: Path,
        **kwargs,
    ) -> DockResult:
        """Run docking and return a DockResult.

        kwargs are backend-specific (e.g., exhaustiveness for Vina,
        recycling_steps for Boltz). Always supply sensible defaults.
        """

    # ---- optional batch path ----

    def dock_batch(
        self,
        receptor: ReceptorSpec,
        ligands: list[LigandSpec],
        box: BoxSpec | None,
        work_dir: Path,
        **kwargs,
    ) -> list[DockResult]:
        """Default fallback: serial loop. GPU backends should override."""
        prepared = self.prepare_receptor(receptor, work_dir)
        return [self.dock(prepared, lig, box, work_dir, **kwargs) for lig in ligands]
