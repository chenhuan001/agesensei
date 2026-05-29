"""AutoDock Vina backend.

Wraps the empirical-scoring docking primitives originally written for AgeSensei
W2. The legacy module goal3_ai_research/projects/agesensei/src/cadd/vina_runner.py
is now a thin compatibility shim over this one.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from affinity_ensemble.backend import DockingBackend
from affinity_ensemble.types import BoxSpec, DockResult, LigandSpec, ReceptorSpec


def prepare_receptor_pdbqt(pdb_in: Path, pdbqt_out: Path) -> Path:
    """Protein PDB -> PDBQT via OpenBabel (rigid, pH 7.4, Gasteiger)."""
    cmd = [
        "obabel", str(pdb_in), "-O", str(pdbqt_out),
        "-xr", "-p", "7.4", "--partialcharge", "gasteiger",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"obabel receptor failed: {res.stderr}")
    return pdbqt_out


def smiles_to_3d_mol(smiles: str, seed: int = 42):
    """SMILES -> 3D-embedded RDKit Mol with hydrogens. None on embed failure."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True) != 0:
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=200) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    return mol


def prepare_ligand_pdbqt(smiles: str, pdbqt_out: Path) -> Path | None:
    """SMILES -> 3D mol -> SDF -> PDBQT via Meeko."""
    from rdkit import Chem

    mol = smiles_to_3d_mol(smiles)
    if mol is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
        sdf_path = f.name
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()
    try:
        cmd = ["mk_prepare_ligand.py", "-i", sdf_path, "-o", str(pdbqt_out)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode != 0 or not Path(pdbqt_out).exists():
            return None
        return pdbqt_out
    finally:
        os.unlink(sdf_path)


def prepare_ligand_pdbqt_from_pdb(pdb_in: Path, pdbqt_out: Path) -> Path | None:
    """PDB ligand -> PDBQT (preserves coords; for redock validation)."""
    cmd = [
        "obabel", str(pdb_in), "-O", str(pdbqt_out),
        "-h", "--partialcharge", "gasteiger",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0 or not Path(pdbqt_out).exists():
        return None
    return pdbqt_out


def pose_rmsd_from_pdbqt(
    ref_pdb: Path,
    dock_pdbqt: Path,
    model_idx: int = 1,
    work_dir: Path | None = None,
) -> float | None:
    """Heavy-atom RMSD via OpenBabel OBAlign. None on failure."""
    from openbabel import openbabel as ob

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="rmsd_"))
    pose_pdb = Path(work_dir) / f"pose_{model_idx}.pdb"
    cmd = ["obabel", str(dock_pdbqt), "-O", str(pose_pdb),
           f"-f{model_idx}", f"-l{model_idx}"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if res.returncode != 0 or not pose_pdb.exists():
        return None

    def _load(path: str):
        conv = ob.OBConversion()
        conv.SetInFormat(path.split(".")[-1])
        mol = ob.OBMol()
        conv.ReadFile(mol, path)
        mol.DeleteHydrogens()
        return mol

    try:
        ref = _load(str(ref_pdb))
        pose = _load(str(pose_pdb))
        align = ob.OBAlign(False, False)
        align.SetRefMol(ref)
        align.SetTargetMol(pose)
        if not align.Align():
            return None
        return float(align.GetRMSD())
    except Exception:
        return None


class VinaBackend(DockingBackend):
    name = "vina"

    def __init__(self, cpu: int = 4, sf_name: str = "vina"):
        self.cpu = cpu
        self.sf_name = sf_name

    @property
    def supports_blind_docking(self) -> bool:
        return False

    @property
    def returns_native_affinity(self) -> bool:
        return True

    @property
    def returns_confidence(self) -> bool:
        return False

    def prepare_receptor(self, receptor: ReceptorSpec, work_dir: Path) -> ReceptorSpec:
        if receptor.pdbqt_path is not None:
            return receptor
        if receptor.pdb_path is None:
            raise ValueError("VinaBackend needs pdb_path or pdbqt_path")
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        pdbqt = work_dir / "receptor.pdbqt"
        prepare_receptor_pdbqt(Path(receptor.pdb_path), pdbqt)
        return ReceptorSpec(
            pdb_path=receptor.pdb_path,
            pdbqt_path=pdbqt,
            chain_id=receptor.chain_id,
        )

    def _prepare_ligand_pdbqt(self, ligand: LigandSpec, work_dir: Path) -> Path | None:
        if ligand.pdbqt_path is not None:
            return Path(ligand.pdbqt_path)
        out = Path(work_dir) / f'{ligand.name or "ligand"}.pdbqt'
        if ligand.smiles is not None:
            return prepare_ligand_pdbqt(ligand.smiles, out)
        if ligand.pdb_path is not None:
            return prepare_ligand_pdbqt_from_pdb(Path(ligand.pdb_path), out)
        return None

    def dock(
        self,
        receptor: ReceptorSpec,
        ligand: LigandSpec,
        box: BoxSpec | None,
        work_dir: Path,
        *,
        exhaustiveness: int = 8,
        n_poses: int = 9,
    ) -> DockResult:
        if box is None:
            return DockResult(
                backend=self.name, affinity_top=None,
                smiles=ligand.smiles, ligand_name=ligand.name,
                error="VinaBackend requires BoxSpec; use blind backend for site-free docking",
            )

        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        if receptor.pdbqt_path is None:
            receptor = self.prepare_receptor(receptor, work_dir)

        lig_pdbqt = self._prepare_ligand_pdbqt(ligand, work_dir)
        if lig_pdbqt is None:
            return DockResult(
                backend=self.name, affinity_top=None,
                smiles=ligand.smiles, ligand_name=ligand.name,
                error="ligand_prep_failed",
            )

        try:
            from vina import Vina
        except ImportError as e:
            raise ImportError(
                "VinaBackend requires the vina package. "
                "Install via: pip install affinity-ensemble[vina]"
            ) from e

        try:
            v = Vina(sf_name=self.sf_name, cpu=self.cpu, verbosity=0)
            v.set_receptor(str(receptor.pdbqt_path))
            v.set_ligand_from_file(str(lig_pdbqt))
            v.compute_vina_maps(center=list(box.center), box_size=list(box.size))
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

            energies = v.energies(n_poses=n_poses)
            affinities = energies[:, 0].tolist() if len(energies) > 0 else []

            pose_out = work_dir / f'{ligand.name or "ligand"}_poses.pdbqt'
            v.write_poses(str(pose_out), n_poses=n_poses, overwrite=True)

            return DockResult(
                backend=self.name,
                affinity_top=affinities[0] if affinities else None,
                affinity_all=affinities,
                pose_path=str(pose_out),
                smiles=ligand.smiles,
                ligand_name=ligand.name,
            )
        except Exception as e:
            return DockResult(
                backend=self.name, affinity_top=None,
                smiles=ligand.smiles, ligand_name=ligand.name,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )
