"""抗衰老相关靶点常量。

主要按 senolytic / senescence pathway 整理。
"""

SENOLYTIC_TARGETS = {
    # BCL-2 family anti-apoptotic: senescent cells 依赖它们存活
    "BCL-xL": {
        "uniprot": "Q07817", "chembl_id": "CHEMBL4625",
        "pdb_examples": ["2YXJ", "1YSI", "4QVE"],
        "rationale": "经典 senolytic 靶点，Navitoclax (ABT-263) 抑制",
    },
    "BCL-2":  {
        "uniprot": "P10415", "chembl_id": "CHEMBL4860",
        "pdb_examples": ["2W3L", "4LVT"],
        "rationale": "Venetoclax (ABT-199) 已上市抗白血病",
    },
    "MCL-1":  {
        "uniprot": "Q07820", "chembl_id": "CHEMBL5113",
        "pdb_examples": ["6QGG"],
        "rationale": "BCL family 第三靶点",
    },
    # mTOR / IGF1R - geroprotective
    "mTOR": {
        "uniprot": "P42345", "chembl_id": "CHEMBL2842",
        "pdb_examples": ["4JT5"],
        "rationale": "Rapamycin 靶点，限制热量信号",
    },
    # Senescence-associated metabolic
    "p16-INK4a (CDKN2A)": {
        "uniprot": "P42771", "chembl_id": "CHEMBL2095192",
        "pdb_examples": ["1A5E"],
        "rationale": "senescence marker，难成药但生物学重要",
    },
}


def get_target(name: str) -> dict:
    """Look up a target by short name."""
    if name not in SENOLYTIC_TARGETS:
        raise KeyError(f"Unknown target {name}. Available: {list(SENOLYTIC_TARGETS)}")
    return SENOLYTIC_TARGETS[name]
