"""Backend registry and lazy factory.

Backends are imported lazily so that missing optional dependencies (e.g., torch
for Boltz-2) don't crash users who only need Vina.
"""
from __future__ import annotations

from typing import Callable

from affinity_ensemble.backend import DockingBackend

_REGISTRY: dict[str, Callable[[], type[DockingBackend]]] = {}


def register_backend(name: str, loader: Callable[[], type[DockingBackend]]) -> None:
    """Register a backend by name. `loader` returns the class on call."""
    _REGISTRY[name] = loader


def available_backends() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_backend(name: str, **init_kwargs) -> DockingBackend:
    """Instantiate the backend registered under `name`.

    Raises:
        KeyError: unknown backend name.
        ImportError: backend's optional dependencies are missing — message
            includes the install hint.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown backend {name!r}. Available: {available_backends()}"
        )
    cls = _REGISTRY[name]()
    return cls(**init_kwargs)


# ---- built-in registrations (loaders are lazy) ----

def _load_vina() -> type[DockingBackend]:
    from affinity_ensemble.backends.vina import VinaBackend
    return VinaBackend


def _load_unidock() -> type[DockingBackend]:
    from affinity_ensemble.backends.unidock import UniDockBackend
    return UniDockBackend


def _load_boltz2() -> type[DockingBackend]:
    from affinity_ensemble.backends.boltz2 import Boltz2Backend
    return Boltz2Backend


def _load_diffdock() -> type[DockingBackend]:
    from affinity_ensemble.backends.diffdock import DiffDockBackend
    return DiffDockBackend


register_backend("vina", _load_vina)
register_backend("unidock", _load_unidock)
register_backend("boltz2", _load_boltz2)
register_backend("diffdock", _load_diffdock)
