"""HAGR (Human Ageing Genomic Resources) database wrapper.

Databases included:
    - GenAge: curated aging-related genes (307 human genes)
    - CellAge: genes associated with cellular senescence
    - DrugAge: drugs/compounds that extend lifespan in model organisms
    - LongevityMap: genetic variants associated with human longevity

Data source: https://genomics.senescence.info/
All databases available as CSV downloads, no API key needed.
"""

import pandas as pd
from pathlib import Path
import httpx
import zipfile
import io
from agesensei.config import DB_DIR

HAGR_URLS = {
    "genage_human": "https://genomics.senescence.info/genes/human_genes.zip",
    "genage_model": "https://genomics.senescence.info/genes/models_genes.zip",
    "cellage": "https://genomics.senescence.info/cells/cellAge.zip",
    "drugage": "https://genomics.senescence.info/drugs/dataset.zip",
    "longevitymap": "https://genomics.senescence.info/longevity/longevity_genes.zip",
}

# Cache for loaded dataframes
_cache = {}


def download_and_extract(url: str, cache_dir: Path) -> Path:
    """Download zip file and extract CSV."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine cache filename from URL
    db_name = url.split("/")[-2]  # e.g., "genes", "cells", "drugs"
    csv_name = f"{db_name}.csv"
    csv_path = cache_dir / csv_name

    # Return cached file if exists (CSV or TSV)
    for ext in ['.csv', '.tsv']:
        csv_path = cache_dir / f"{db_name}{ext}"
        if csv_path.exists():
            return csv_path

    # Download and extract
    print(f"Downloading {db_name} from HAGR...")
    response = httpx.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()

    # Extract data file from zip
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # Find the data file (CSV or TSV)
        data_files = [f for f in zf.namelist() if f.endswith(('.csv', '.tsv'))]
        if not data_files:
            raise ValueError(f"No CSV/TSV file found in {url}")

        src_name = data_files[0]
        ext = Path(src_name).suffix  # .csv or .tsv
        csv_path = cache_dir / f"{db_name}{ext}"

        with zf.open(src_name) as data_file:
            csv_path.write_bytes(data_file.read())

    print(f"  Saved to {csv_path}")
    return csv_path


def _load_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV or TSV file based on extension."""
    sep = "\t" if csv_path.suffix == ".tsv" else ","
    df = pd.read_csv(csv_path, sep=sep)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def load_genage(cache_dir: Path = DB_DIR, force_download: bool = False) -> pd.DataFrame:
    """Load GenAge human aging genes database.

    Returns:
        DataFrame with columns:
            - symbol: gene symbol (e.g., "TP53")
            - name: gene name
            - entrez_gene_id: Entrez Gene ID
            - why: reason for inclusion in database
    """
    if not force_download and "genage_human" in _cache:
        return _cache["genage_human"]

    csv_path = download_and_extract(HAGR_URLS["genage_human"], cache_dir)
    df = _load_data(csv_path)

    _cache["genage_human"] = df
    return df


def load_cellage(cache_dir: Path = DB_DIR, force_download: bool = False) -> pd.DataFrame:
    """Load CellAge cellular senescence genes.

    Returns:
        DataFrame with columns:
            - gene_symbol: gene symbol
            - gene_name: gene name
            - senescence_effect: Induces / Inhibits
            - type_of_senescence: Replicative, Oncogene-induced, Stress-induced, etc.
    """
    if not force_download and "cellage" in _cache:
        return _cache["cellage"]

    csv_path = download_and_extract(HAGR_URLS["cellage"], cache_dir)
    df = _load_data(csv_path)

    _cache["cellage"] = df
    return df


def load_drugage(cache_dir: Path = DB_DIR, force_download: bool = False) -> pd.DataFrame:
    """Load DrugAge lifespan-extending compounds.

    Returns:
        DataFrame with columns vary by version, typically:
            - compound_name: drug/compound name
            - species: organism tested
            - avg_lifespan_change: average lifespan change percentage
    """
    if not force_download and "drugage" in _cache:
        return _cache["drugage"]

    csv_path = download_and_extract(HAGR_URLS["drugage"], cache_dir)
    df = _load_data(csv_path)

    _cache["drugage"] = df
    return df


def _get_symbol_column(df: pd.DataFrame) -> str:
    """Find the gene symbol column name in a DataFrame."""
    for col in ['symbol', 'gene_symbol', 'gene']:
        if col in df.columns:
            return col
    raise ValueError(f"No gene symbol column found. Columns: {list(df.columns)}")


def is_known_aging_gene(gene_symbol: str, include_cellage: bool = True) -> bool:
    """Check if a gene is in GenAge or CellAge database."""
    gene_symbol = gene_symbol.upper()

    genage = load_genage()
    sym_col = _get_symbol_column(genage)
    if gene_symbol in genage[sym_col].str.upper().values:
        return True

    if include_cellage:
        try:
            cellage = load_cellage()
            sym_col = _get_symbol_column(cellage)
            if gene_symbol in cellage[sym_col].str.upper().values:
                return True
        except Exception:
            pass  # CellAge might fail to download

    return False


def get_aging_evidence(gene_symbol: str) -> dict | None:
    """Get aging-related evidence for a gene from HAGR databases."""
    gene_symbol = gene_symbol.upper()
    evidence = {}

    # Check GenAge
    genage = load_genage()
    sym_col = _get_symbol_column(genage)
    genage_match = genage[genage[sym_col].str.upper() == gene_symbol]
    if not genage_match.empty:
        row = genage_match.iloc[0]
        evidence['genage'] = {col: str(row.get(col, '')) for col in genage.columns}

    # Check CellAge
    try:
        cellage = load_cellage()
        sym_col = _get_symbol_column(cellage)
        cellage_match = cellage[cellage[sym_col].str.upper() == gene_symbol]
        if not cellage_match.empty:
            row = cellage_match.iloc[0]
            evidence['cellage'] = {col: str(row.get(col, '')) for col in cellage.columns}
    except Exception:
        pass

    return evidence if evidence else None


def get_all_aging_genes() -> set[str]:
    """Get set of all known aging/senescence gene symbols."""
    genes = set()

    genage = load_genage()
    sym_col = _get_symbol_column(genage)
    genes.update(genage[sym_col].str.upper())

    try:
        cellage = load_cellage()
        sym_col = _get_symbol_column(cellage)
        genes.update(cellage[sym_col].str.upper())
    except Exception:
        pass

    return genes
