"""ChEMBL bioactivity fetcher.

Pulls activity records for a given target ChEMBL ID and writes a clean
parquet/csv with: molecule_chembl_id / smiles / pchembl / standard_type / source_assay.

Usage:
    python -m src.cadd.chembl_fetch --target CHEMBL4625 --out data/cadd/bcl_xl/activities.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from chembl_webresource_client.new_client import new_client


def fetch_activities(
    target_chembl_id: str,
    standard_types: list[str] = ("IC50", "Ki", "Kd"),
    require_pchembl: bool = True,
    require_smiles: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    """Pull activity records for a target."""
    activity = new_client.activity
    flt = {
        "target_chembl_id": target_chembl_id,
        "standard_type__in": list(standard_types),
    }
    if require_pchembl:
        flt["pchembl_value__isnull"] = False
    if require_smiles:
        flt["canonical_smiles__isnull"] = False

    qs = activity.filter(**flt).only([
        "molecule_chembl_id", "canonical_smiles",
        "standard_type", "standard_value", "standard_units",
        "pchembl_value", "assay_chembl_id", "assay_type",
        "target_chembl_id",
    ])

    n_total = len(qs)
    print(f"Target {target_chembl_id}: {n_total} records match filter")
    if limit:
        n_total = min(n_total, limit)
        qs = qs[:limit]

    rows = []
    t0 = time.time()
    for i, a in enumerate(qs):
        rows.append({
            "molecule_chembl_id": a["molecule_chembl_id"],
            "smiles": a["canonical_smiles"],
            "standard_type": a["standard_type"],
            "standard_value": float(a["standard_value"]) if a["standard_value"] else None,
            "standard_units": a["standard_units"],
            "pchembl_value": float(a["pchembl_value"]) if a["pchembl_value"] else None,
            "assay_chembl_id": a["assay_chembl_id"],
            "assay_type": a["assay_type"],
        })
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{n_total}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    print(f"Fetched {len(df)} records in {time.time()-t0:.1f}s")
    return df


def dedup_by_molecule(df: pd.DataFrame, agg: str = "median") -> pd.DataFrame:
    """One molecule may have many activity entries (different assays).
    Aggregate per molecule_chembl_id by median pchembl_value (业界通用).
    """
    if agg not in {"median", "mean", "max"}:
        raise ValueError(agg)
    g = df.groupby("molecule_chembl_id").agg({
        "smiles": "first",
        "pchembl_value": agg,
        "standard_type": lambda x: ",".join(sorted(set(x))),
        "assay_chembl_id": "nunique",
    }).reset_index()
    g = g.rename(columns={"assay_chembl_id": "n_assays"})
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="ChEMBL target id (e.g. CHEMBL4625)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-pchembl-filter", action="store_true")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_activities(
        args.target,
        require_pchembl=not args.no_pchembl_filter,
        limit=args.limit,
    )
    print(f"Raw shape: {df.shape}")

    # Save raw
    raw_path = args.out.with_suffix(".raw.parquet")
    df.to_parquet(raw_path, index=False)
    print(f"Raw saved: {raw_path}")

    # Dedup per molecule (median pchembl)
    df_unique = dedup_by_molecule(df)
    print(f"Unique molecules: {len(df_unique)}")
    print(f"pChEMBL distribution: min={df_unique['pchembl_value'].min():.2f} "
          f"median={df_unique['pchembl_value'].median():.2f} "
          f"max={df_unique['pchembl_value'].max():.2f}")
    df_unique.to_parquet(args.out, index=False)
    print(f"Unique saved: {args.out}")

    # Quick activity buckets
    print("\nActivity buckets:")
    bins = [(0, 5, "weak"), (5, 6, "modest"), (6, 7, "good"), (7, 100, "potent")]
    for lo, hi, label in bins:
        n = ((df_unique["pchembl_value"] >= lo) & (df_unique["pchembl_value"] < hi)).sum()
        print(f"  pChEMBL [{lo},{hi})  {label:8s}: {n:5d}")


if __name__ == "__main__":
    main()
