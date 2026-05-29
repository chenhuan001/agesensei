"""SMILES → fingerprints / descriptors / drug-likeness 工具。

Used by:
- W2 Day 1: RDKit basics (this file demos)
- W2 Day 4-5: QSAR feature engineering
- W4: docking_agent + reporter feature extraction
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED


@dataclass
class MolDescriptors:
    """Lightweight 12-feature descriptor set, common in QSAR."""
    mw: float
    logp: float
    hba: int
    hbd: int
    rotbonds: int
    aromatic_rings: int
    fraction_csp3: float
    tpsa: float
    qed: float
    rings: int
    heavy_atoms: int
    formal_charge: int

    def to_array(self) -> np.ndarray:
        return np.array([
            self.mw, self.logp, self.hba, self.hbd,
            self.rotbonds, self.aromatic_rings, self.fraction_csp3,
            self.tpsa, self.qed, self.rings, self.heavy_atoms,
            self.formal_charge,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> list[str]:
        return ["MW", "LogP", "HBA", "HBD", "RotBonds", "AromaticRings",
                "FractionCSP3", "TPSA", "QED", "Rings", "HeavyAtoms",
                "FormalCharge"]


def smiles_to_mol(smiles: str):
    """SMILES → RDKit Mol，失败返回 None。"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol
    except Exception:
        return None


def morgan_fingerprint(mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """生成 Morgan / ECFP fingerprint。返回 0/1 numpy array。

    radius=2 → ECFP4（业界标准）。n_bits=2048 平衡分辨率与维度。
    """
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def compute_descriptors(mol) -> MolDescriptors | None:
    """12 维 QSAR 描述符 + 类药性."""
    if mol is None:
        return None
    return MolDescriptors(
        mw=Descriptors.MolWt(mol),
        logp=Descriptors.MolLogP(mol),
        hba=Lipinski.NumHAcceptors(mol),
        hbd=Lipinski.NumHDonors(mol),
        rotbonds=Lipinski.NumRotatableBonds(mol),
        aromatic_rings=Lipinski.NumAromaticRings(mol),
        fraction_csp3=Lipinski.FractionCSP3(mol),
        tpsa=Descriptors.TPSA(mol),
        qed=QED.qed(mol),
        rings=Lipinski.RingCount(mol),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        formal_charge=Chem.GetFormalCharge(mol),
    )


def passes_lipinski(desc: MolDescriptors, strict: bool = False) -> bool:
    """Lipinski Rule of Five.

    strict=True 全部满足；False 允许 1 条违反（业界惯例）。
    """
    violations = sum([
        desc.mw > 500,
        desc.logp > 5,
        desc.hba > 10,
        desc.hbd > 5,
    ])
    return violations == 0 if strict else violations <= 1


def smiles_to_features(
    smiles: str,
    fp_radius: int = 2,
    fp_bits: int = 2048,
) -> tuple[np.ndarray, MolDescriptors] | None:
    """One-shot：SMILES → (fingerprint, descriptors)。失败返回 None。"""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    fp = morgan_fingerprint(mol, fp_radius, fp_bits)
    desc = compute_descriptors(mol)
    if fp is None or desc is None:
        return None
    return fp, desc


def batch_to_matrix(
    smiles_list: list[str],
    fp_radius: int = 2,
    fp_bits: int = 2048,
) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """批量：SMILES list → (fp_matrix, desc_matrix, valid_mask)。

    无效 SMILES 在矩阵里位置全 0，valid_mask 标记 True/False。
    """
    n = len(smiles_list)
    fp_mat = np.zeros((n, fp_bits), dtype=np.uint8)
    desc_mat = np.zeros((n, len(MolDescriptors.feature_names())), dtype=np.float32)
    valid = [False] * n
    for i, smi in enumerate(smiles_list):
        r = smiles_to_features(smi, fp_radius, fp_bits)
        if r is None:
            continue
        fp_mat[i] = r[0]
        desc_mat[i] = r[1].to_array()
        valid[i] = True
    return fp_mat, desc_mat, valid


if __name__ == "__main__":
    # Demo: ABT-263 (Navitoclax, 经典 senolytic, BCL-xL/2 抑制剂)
    abt263 = "Cc1ccc(C(=O)NS(=O)(=O)c2ccc(NCC3CCN(Cc4ccc(-c5cc(Cl)c(Cl)cc5C(F)(F)F)cc4)CC3)c([N+](=O)[O-])c2)cc1"
    fp, desc = smiles_to_features(abt263)
    print(f"ABT-263 (Navitoclax):")
    print(f"  Morgan fp (2048b): {fp.sum()} bits set")
    print(f"  MW={desc.mw:.1f} LogP={desc.logp:.2f} HBA={desc.hba} HBD={desc.hbd}")
    print(f"  TPSA={desc.tpsa:.1f} QED={desc.qed:.3f}")
    print(f"  Lipinski-pass (strict)? {passes_lipinski(desc, strict=True)}")
    print(f"  Lipinski-pass (loose)?  {passes_lipinski(desc, strict=False)}")
    # Navitoclax is known to fail strict Lipinski (large MW ~975) - 真实 senolytic 经常超 RoF
