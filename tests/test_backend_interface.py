"""Smoke tests for the backend abstraction layer.

These tests do NOT require Vina/Boltz/etc. to be installed - they only verify
that the registry, types, and capability flags behave correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from affinity_ensemble import (
    BoxSpec,
    DockResult,
    LigandSpec,
    ReceptorSpec,
    available_backends,
    get_backend,
)


def test_registry_contains_all_planned_backends():
    assert set(available_backends()) >= {"vina", "unidock", "boltz2", "diffdock"}


def test_unknown_backend_raises_keyerror():
    with pytest.raises(KeyError):
        get_backend("not-a-backend")


def test_box_spec_from_legacy_dict():
    box = BoxSpec.from_dict({
        "center_x": 1.0, "center_y": 2.0, "center_z": 3.0,
        "size_x": 20.0, "size_y": 22.0, "size_z": 24.0,
    })
    assert box.center == (1.0, 2.0, 3.0)
    assert box.size == (20.0, 22.0, 24.0)


def test_ligand_spec_requires_at_least_one_input():
    with pytest.raises(ValueError):
        LigandSpec()


def test_receptor_spec_requires_at_least_one_input():
    with pytest.raises(ValueError):
        ReceptorSpec()


def test_dock_result_ok_flag():
    good = DockResult(backend="vina", affinity_top=-7.5, affinity_all=[-7.5, -6.9])
    bad = DockResult(backend="vina", affinity_top=None, error="ligand_prep_failed")
    assert good.ok is True
    assert bad.ok is False


def test_capability_flags_distinguish_backends():
    # Stubs only - we don't call dock(), just inspect the class attributes.
    from affinity_ensemble.backends.vina import VinaBackend
    from affinity_ensemble.backends.boltz2 import Boltz2Backend
    from affinity_ensemble.backends.diffdock import DiffDockBackend

    assert VinaBackend().supports_blind_docking is False
    assert Boltz2Backend().supports_blind_docking is True
    assert DiffDockBackend().returns_native_affinity is False
    assert DiffDockBackend().returns_confidence is True
