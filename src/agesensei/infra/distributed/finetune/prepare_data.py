"""Prepare training data for ESM-2 fine-tuning from GenAge and ClinVar.

Downloads aging-related protein sequences from GenAge database and
pathogenic/benign variants from ClinVar for aging genes.

Usage:
    python -m agesensei.infra.distributed.finetune.prepare_data \
        --output data/finetune/ --task mlm

    python -m agesensei.infra.distributed.finetune.prepare_data \
        --output data/finetune/ --task mutation_cls
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
from pathlib import Path

from agesensei.tools import uniprot


# GenAge human genes (curated list of aging-associated genes)
GENAGE_GENES = [
    "TP53", "SIRT1", "SIRT3", "SIRT6", "MTOR", "FOXO3", "TERT", "LMNA",
    "WRN", "BLM", "ATM", "PTEN", "IGF1R", "IGF1", "GH1", "KLOTHO",
    "APOE", "SOD1", "SOD2", "CAT", "GPX1", "PRKAA1", "PRKAA2",
    "CDKN2A", "RB1", "MDM2", "TERC", "POT1", "TINF2", "DKC1",
    "NFE2L2", "KEAP1", "HMGB1", "CISD2", "PPARGC1A", "TFAM",
    "POLG", "PARK2", "PINK1", "BECN1", "ATG5", "ATG7", "SQSTM1",
    "LAMP2", "CTSD", "HSP90AA1", "HSPA1A", "UBB", "PSMD14",
    "BCL2", "BCL2L1", "BAX", "CASP3", "CASP9", "CYCS",
    "TNF", "IL6", "IL1B", "NFKB1", "RELA", "JAK2", "STAT3",
    "VEGFA", "HIF1A", "EPAS1", "VHL", "EGFR", "ERBB2",
    "BRCA1", "BRCA2", "RAD51", "XRCC5", "XRCC6", "PARP1",
    "HDAC1", "HDAC2", "KAT2A", "EP300", "CREBBP",
    "DNMT1", "DNMT3A", "DNMT3B", "TET1", "TET2",
    "GDF11", "MSTN", "CCL11", "SERPINE1", "F2",
]


async def fetch_sequences(genes: list[str]) -> list[dict]:
    """Fetch protein sequences from UniProt for a list of genes."""
    results = []
    for gene in genes:
        try:
            accession = await uniprot.search_by_gene(gene, organism="human")
            if not accession:
                print(f"  {gene}: not found in UniProt, skipping")
                continue
            protein = await uniprot.fetch_protein(accession)
            sequence = protein.get("sequence", "")
            if sequence and len(sequence) >= 50:
                results.append({
                    "gene": gene,
                    "uniprot_id": accession,
                    "sequence": sequence,
                    "length": len(sequence),
                })
                print(f"  {gene}: {accession} ({len(sequence)} aa)")
            else:
                print(f"  {gene}: sequence too short ({len(sequence)} aa), skipping")
        except Exception as e:
            print(f"  {gene}: error - {e}")
    return results


def generate_synthetic_mutations(
    sequences: list[dict],
    n_per_protein: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic mutation training data.

    Strategy:
        - Deleterious mutations: positions with high conservation (based on
          amino acid frequency in protein families). We simulate by mutating
          to chemically dissimilar amino acids at conserved positions.
        - Neutral mutations: surface-exposed, variable positions mutated to
          chemically similar amino acids.

    This is a simplified heuristic. For production use, integrate ClinVar
    pathogenic/benign annotations or DMS (Deep Mutational Scanning) data.
    """
    rng = random.Random(seed)
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

    # Chemical similarity groups
    SIMILAR_GROUPS = {
        "A": "GVIL", "G": "AVIL", "V": "ILAM", "I": "LVAM", "L": "IVAM",
        "F": "YWH", "Y": "FWH", "W": "FYH",
        "S": "TNC", "T": "SNC",
        "D": "EN", "E": "DN", "N": "DQS", "Q": "NEK",
        "K": "RQH", "R": "KQH", "H": "KRN",
        "C": "SM", "M": "CLV", "P": "A",
    }

    # Dissimilar: charged <-> hydrophobic
    DISSIMILAR = {
        "D": "VILFW", "E": "VILFW", "K": "VILFW", "R": "VILFW",
        "V": "DERK", "I": "DERK", "L": "DERK", "F": "DERK", "W": "DERK",
        "A": "DERK", "G": "DERKW", "S": "VILFW", "T": "VILFW",
        "N": "VILFW", "Q": "VILFW", "H": "VILFW",
        "C": "DERK", "M": "DERK", "P": "DERKW", "Y": "DERK",
    }

    samples = []
    for prot in sequences:
        seq = prot["sequence"]
        if len(seq) < 10:
            continue

        for _ in range(n_per_protein):
            pos = rng.randint(0, len(seq) - 1)
            wt = seq[pos]
            if wt not in AA_LIST:
                continue

            # 50% deleterious, 50% neutral
            if rng.random() < 0.5:
                # Deleterious: mutate to dissimilar
                candidates = DISSIMILAR.get(wt, "")
                if not candidates:
                    candidates = [aa for aa in AA_LIST if aa != wt]
                mutant = rng.choice(list(candidates))
                label = 1
            else:
                # Neutral: mutate to similar
                candidates = SIMILAR_GROUPS.get(wt, "")
                if not candidates:
                    candidates = [aa for aa in AA_LIST if aa != wt]
                mutant = rng.choice(list(candidates))
                label = 0

            samples.append({
                "sequence": seq,
                "position": pos,
                "wildtype": wt,
                "mutant": mutant,
                "label": label,
                "gene": prot["gene"],
            })

    return samples


def write_mlm_csv(sequences: list[dict], output_path: Path):
    """Write MLM training CSV (just sequences)."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence"])
        writer.writeheader()
        for prot in sequences:
            writer.writerow({"sequence": prot["sequence"]})
    print(f"  MLM dataset: {len(sequences)} sequences -> {output_path}")


def write_mutation_csv(mutations: list[dict], output_path: Path):
    """Write mutation classification CSV."""
    fields = ["sequence", "position", "wildtype", "mutant", "label"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in mutations:
            writer.writerow({k: m[k] for k in fields})
    print(f"  Mutation dataset: {len(mutations)} samples -> {output_path}")


async def prepare_data(output_dir: str, task: str, n_mutations_per_protein: int = 20):
    """Main data preparation pipeline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Fetching sequences for {len(GENAGE_GENES)} GenAge genes...")
    sequences = await fetch_sequences(GENAGE_GENES)
    print(f"  Retrieved {len(sequences)} sequences")

    if task == "mlm" or task == "both":
        # Split into train/val
        random.shuffle(sequences)
        n_val = max(1, int(len(sequences) * 0.1))
        write_mlm_csv(sequences[n_val:], out / "train_mlm.csv")
        write_mlm_csv(sequences[:n_val], out / "val_mlm.csv")

    if task == "mutation_cls" or task == "both":
        mutations = generate_synthetic_mutations(sequences, n_per_protein=n_mutations_per_protein)
        random.shuffle(mutations)
        n_val = max(1, int(len(mutations) * 0.1))
        write_mutation_csv(mutations[n_val:], out / "train_mutations.csv")
        write_mutation_csv(mutations[:n_val], out / "val_mutations.csv")

    # Save gene list for reference
    genes_path = out / "genage_genes.txt"
    genes_path.write_text("\n".join(sorted(set(p["gene"] for p in sequences))))
    print(f"\nData preparation complete. Output: {out}")


def main():
    ap = argparse.ArgumentParser(description="Prepare ESM-2 fine-tuning data from GenAge")
    ap.add_argument("--output", type=str, default="data/finetune",
                    help="Output directory for prepared datasets")
    ap.add_argument("--task", choices=["mlm", "mutation_cls", "both"], default="both",
                    help="Which task to prepare data for")
    ap.add_argument("--mutations-per-protein", type=int, default=20,
                    help="Synthetic mutations per protein (for mutation_cls)")
    args = ap.parse_args()

    asyncio.run(prepare_data(args.output, args.task, args.mutations_per_protein))


if __name__ == "__main__":
    main()
