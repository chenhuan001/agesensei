"""affinity_ensemble — backend-agnostic protein-ligand affinity prediction."""
from affinity_ensemble.types import (
    BoxSpec,
    DockResult,
    LigandSpec,
    ReceptorSpec,
)
from affinity_ensemble.backend import DockingBackend
from affinity_ensemble.registry import (
    available_backends,
    get_backend,
    register_backend,
)

__version__ = "0.0.1"

__all__ = [
    "BoxSpec",
    "DockResult",
    "DockingBackend",
    "LigandSpec",
    "ReceptorSpec",
    "available_backends",
    "get_backend",
    "register_backend",
]
