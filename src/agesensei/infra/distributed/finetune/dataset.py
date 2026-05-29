"""Dataset for ESM-2 fine-tuning on aging-related mutation data.

Supports two tasks:
    1. Masked Language Modeling (MLM) on aging-related protein sequences
    2. Mutation Effect Prediction (binary: deleterious vs neutral)

Data sources:
    - GenAge database proteins (curated aging genes)
    - ClinVar pathogenic/benign variants on aging-related genes
    - Custom CSV with columns: sequence, position, wildtype, mutant, label
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset


class AgingMutationDataset(Dataset):
    """Dataset for mutation effect prediction on aging-related proteins.

    Each sample: (masked_sequence, label) where label is 1 (deleterious) or 0 (neutral).
    The model learns to predict whether a mutation at a given position is harmful.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_length: int = 1024,
        task: Literal["mlm", "mutation_cls"] = "mutation_cls",
        mlm_probability: float = 0.15,
        seed: int = 42,
    ):
        """
        Args:
            data_path: Path to CSV with columns: sequence, position, wildtype, mutant, label
                       For MLM task, only 'sequence' column is required.
            tokenizer: HuggingFace tokenizer (ESM-2)
            max_length: Maximum sequence length (ESM-2 supports up to 1024)
            task: "mlm" for masked language modeling, "mutation_cls" for classification
            mlm_probability: Fraction of tokens to mask (MLM only)
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.mlm_probability = mlm_probability
        self.rng = random.Random(seed)

        self.samples = self._load_csv(data_path)

    def _load_csv(self, path: str | Path) -> list[dict]:
        """Load and validate CSV data."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        samples = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.task == "mlm":
                    if "sequence" not in row:
                        raise ValueError("MLM task requires 'sequence' column")
                    samples.append({"sequence": row["sequence"]})
                else:
                    required = {"sequence", "position", "wildtype", "mutant", "label"}
                    if not required.issubset(row.keys()):
                        raise ValueError(f"mutation_cls requires columns: {required}")
                    samples.append({
                        "sequence": row["sequence"],
                        "position": int(row["position"]),
                        "wildtype": row["wildtype"],
                        "mutant": row["mutant"],
                        "label": int(row["label"]),
                    })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        sequence = sample["sequence"][:self.max_length - 2]  # leave room for special tokens

        if self.task == "mlm":
            return self._prepare_mlm(sequence)
        else:
            return self._prepare_mutation_cls(sample)

    def _prepare_mlm(self, sequence: str) -> dict[str, torch.Tensor]:
        """Prepare masked language modeling sample."""
        encoding = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels (copy of input_ids)
        labels = input_ids.clone()

        # Mask random positions (skip special tokens)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            input_ids.tolist(), already_has_special_tokens=True
        )
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # only compute loss on masked tokens

        # 80% mask, 10% random, 10% keep
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _prepare_mutation_cls(self, sample: dict) -> dict[str, torch.Tensor]:
        """Prepare mutation classification sample.

        Strategy: mask the mutation position and let the model predict.
        The label indicates whether the mutation is deleterious (1) or neutral (0).
        """
        sequence = sample["sequence"][:self.max_length - 2]
        position = sample["position"]

        # Mask the mutation position
        if position < len(sequence):
            masked_seq = sequence[:position] + self.tokenizer.mask_token + sequence[position + 1:]
        else:
            masked_seq = sequence

        encoding = self.tokenizer(
            masked_seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "position": torch.tensor(position + 1, dtype=torch.long),  # +1 for BOS
            "wildtype_id": torch.tensor(
                self.tokenizer.convert_tokens_to_ids(sample["wildtype"]), dtype=torch.long
            ),
            "mutant_id": torch.tensor(
                self.tokenizer.convert_tokens_to_ids(sample["mutant"]), dtype=torch.long
            ),
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


def create_genage_mlm_dataset(
    genage_sequences_path: str | Path,
    tokenizer,
    max_length: int = 1024,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple[AgingMutationDataset, AgingMutationDataset]:
    """Create train/val MLM datasets from GenAge protein sequences.

    Args:
        genage_sequences_path: CSV with 'sequence' column (GenAge proteins)
        tokenizer: ESM-2 tokenizer
        max_length: Max sequence length
        val_split: Fraction for validation
        seed: Random seed

    Returns:
        (train_dataset, val_dataset)
    """
    full_dataset = AgingMutationDataset(
        data_path=genage_sequences_path,
        tokenizer=tokenizer,
        max_length=max_length,
        task="mlm",
        seed=seed,
    )

    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_dataset, val_dataset
