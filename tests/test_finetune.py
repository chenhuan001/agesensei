"""Tests for ESM-2 fine-tuning pipeline."""

import csv
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch required for finetune tests")


def _create_mock_mlm_csv(path: Path, n_samples: int = 10):
    """Create a mock MLM training CSV."""
    sequences = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHISDKFR",
        "MSQSNRELVVDFLSYKLSQKGYSWSQFSDVEENRTEAPEGTESEMETPSAINGNPSWHLADSPAVNGATGHSSSLDARIFSPPPFASDPMELS",
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPL",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence"])
        writer.writeheader()
        for i in range(n_samples):
            writer.writerow({"sequence": sequences[i % len(sequences)]})


def _create_mock_mutation_csv(path: Path, n_samples: int = 20):
    """Create a mock mutation classification CSV."""
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHISDKFR"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "position", "wildtype", "mutant", "label"])
        writer.writeheader()
        for i in range(n_samples):
            pos = i % len(seq)
            wt = seq[pos]
            mutant = "A" if wt != "A" else "G"
            label = 1 if i % 2 == 0 else 0
            writer.writerow({
                "sequence": seq,
                "position": pos,
                "wildtype": wt,
                "mutant": mutant,
                "label": label,
            })


class TestAgingMutationDataset:
    """Test dataset loading and sample preparation."""

    def test_mlm_dataset_loads(self):
        """MLM dataset loads correctly from CSV."""
        from agesensei.infra.distributed.finetune.dataset import AgingMutationDataset
        from unittest.mock import MagicMock

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)
            _create_mock_mlm_csv(path, n_samples=5)

        tokenizer = MagicMock()
        tokenizer.mask_token_id = 32
        tokenizer.get_special_tokens_mask.return_value = [1] + [0] * 90 + [1]
        tokenizer.return_value = {
            "input_ids": __import__("torch").randint(0, 100, (1, 92)),
            "attention_mask": __import__("torch").ones(1, 92, dtype=__import__("torch").long),
        }
        tokenizer.__len__ = lambda self: 33

        dataset = AgingMutationDataset(
            data_path=path, tokenizer=tokenizer, max_length=92, task="mlm"
        )
        assert len(dataset) == 5
        path.unlink()

    def test_mutation_dataset_loads(self):
        """Mutation classification dataset loads correctly."""
        from agesensei.infra.distributed.finetune.dataset import AgingMutationDataset
        from unittest.mock import MagicMock

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)
            _create_mock_mutation_csv(path, n_samples=10)

        tokenizer = MagicMock()
        tokenizer.mask_token = "<mask>"
        tokenizer.convert_tokens_to_ids.return_value = 5
        tokenizer.return_value = {
            "input_ids": __import__("torch").randint(0, 100, (1, 92)),
            "attention_mask": __import__("torch").ones(1, 92, dtype=__import__("torch").long),
        }

        dataset = AgingMutationDataset(
            data_path=path, tokenizer=tokenizer, max_length=92, task="mutation_cls"
        )
        assert len(dataset) == 10

        sample = dataset[0]
        assert "input_ids" in sample
        assert "label" in sample
        assert sample["label"].item() in (0, 1)
        path.unlink()

    def test_missing_file_raises(self):
        """Dataset raises FileNotFoundError for missing file."""
        from agesensei.infra.distributed.finetune.dataset import AgingMutationDataset
        from unittest.mock import MagicMock

        with pytest.raises(FileNotFoundError):
            AgingMutationDataset(
                data_path="/nonexistent/path.csv",
                tokenizer=MagicMock(),
                task="mlm",
            )


class TestSyntheticMutationGeneration:
    """Test synthetic mutation data generation."""

    def test_generates_balanced_labels(self):
        """Synthetic mutations should be roughly 50/50 deleterious/neutral."""
        from agesensei.infra.distributed.finetune.prepare_data import generate_synthetic_mutations

        sequences = [
            {"gene": "TP53", "sequence": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"},
            {"gene": "SIRT1", "sequence": "MADEAALALQPGGSPSAAGADREAASSPAGEPLRKRPRRDGPGLERSPGEPGGAAPEREVP"},
        ]
        mutations = generate_synthetic_mutations(sequences, n_per_protein=100, seed=42)

        assert len(mutations) > 0
        n_deleterious = sum(1 for m in mutations if m["label"] == 1)
        n_neutral = sum(1 for m in mutations if m["label"] == 0)

        # Should be roughly balanced (within 60/40)
        ratio = n_deleterious / len(mutations)
        assert 0.3 < ratio < 0.7, f"Imbalanced: {ratio:.2f}"

    def test_mutations_have_valid_amino_acids(self):
        """All mutations should use valid amino acid codes."""
        from agesensei.infra.distributed.finetune.prepare_data import generate_synthetic_mutations

        AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
        sequences = [
            {"gene": "BCL2", "sequence": "MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPA"},
        ]
        mutations = generate_synthetic_mutations(sequences, n_per_protein=50)

        for m in mutations:
            assert m["wildtype"] in AA_SET, f"Invalid wildtype: {m['wildtype']}"
            assert m["mutant"] in AA_SET, f"Invalid mutant: {m['mutant']}"
            assert m["wildtype"] != m["mutant"], "Wildtype == mutant"
            assert 0 <= m["position"] < len(m["sequence"])


class TestESMFineTunerConfig:
    """Test trainer configuration."""

    def test_deepspeed_config_generation(self):
        """DeepSpeed config should be generated with correct parameters."""
        from agesensei.infra.distributed.finetune.trainer import ESMFineTuner

        trainer = ESMFineTuner(
            batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            use_deepspeed=True,
        )
        # Mock train_loader length
        trainer.train_loader = type("MockLoader", (), {"__len__": lambda self: 100})()
        trainer.epochs = 5

        config = trainer._get_deepspeed_config()
        assert config["train_micro_batch_size_per_gpu"] == 4
        assert config["gradient_accumulation_steps"] == 8
        assert config["zero_optimization"]["stage"] == 2
        assert config["optimizer"]["params"]["lr"] == 1e-5
