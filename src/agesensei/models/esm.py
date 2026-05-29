"""ESM-2 protein language model wrapper.

Provides:
    - Per-residue and mean embeddings
    - Zero-shot mutation effect prediction via masked marginal probability
    - Sequence similarity via cosine distance
"""

import os
import torch
from agesensei.config import config

# Force offline mode to avoid HuggingFace connection issues
# Model must be pre-cached locally
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Standard amino acid alphabet
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


class ESMWrapper:
    """Wrapper for ESM-2 protein language model.

    Uses lazy loading — model is only downloaded/loaded on first use.
    Default model: esm2_t33_650M_UR50D (650M params, good balance of speed/quality)
    For faster inference: esm2_t6_8M_UR50D (8M params)
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.esm.model_name
        self.device = config.esm.device
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Lazy load model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer, EsmForMaskedLM

        print(f"  Loading ESM-2 model: {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = EsmForMaskedLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        print(f"  Model loaded on {self.device}")

    @torch.no_grad()
    def embed(self, sequence: str) -> dict:
        """Compute embeddings for a protein sequence.

        Args:
            sequence: Amino acid sequence (e.g., "MKTAYIAKQRQISFVK...")

        Returns:
            dict with keys:
                - per_residue: tensor (seq_len, hidden_dim)
                - mean: tensor (hidden_dim,)
                - cls: tensor (hidden_dim,)
        """
        self._load()
        inputs = self._tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model.esm(**inputs)
        hidden = outputs.last_hidden_state[0]  # (seq_len+2, hidden_dim)

        return {
            "per_residue": hidden[1:-1].cpu(),  # exclude BOS/EOS
            "mean": hidden[1:-1].mean(dim=0).cpu(),
            "cls": hidden[0].cpu(),
        }

    @torch.no_grad()
    def mutation_scan(self, sequence: str, positions: list[int] | None = None) -> list[dict]:
        """Zero-shot mutation scanning using masked marginal probability.

        For each position, masks the residue and computes log-likelihood
        of every amino acid substitution. Negative score = likely deleterious.

        Args:
            sequence: Wild-type amino acid sequence
            positions: Specific positions to scan (0-indexed). None = scan all.

        Returns:
            List of dicts: {position, wildtype, mutant, score, wildtype_prob}
            Sorted by score (most deleterious first)
        """
        self._load()

        if positions is None:
            positions = list(range(len(sequence)))

        # Limit to avoid OOM on long sequences
        max_len = 1022  # ESM-2 max length
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
            positions = [p for p in positions if p < max_len]

        results = []
        mask_token_id = self._tokenizer.mask_token_id

        for pos in positions:
            wt_aa = sequence[pos]
            if wt_aa not in AA_LIST:
                continue

            # Create masked sequence
            masked_seq = sequence[:pos] + self._tokenizer.mask_token + sequence[pos + 1:]
            inputs = self._tokenizer(masked_seq, return_tensors="pt", add_special_tokens=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self._model(**inputs)
            logits = outputs.logits[0]  # (seq_len+2, vocab_size)

            # Position in tokenized sequence (offset by 1 for BOS token)
            token_pos = pos + 1
            probs = torch.softmax(logits[token_pos], dim=-1)

            # Get wildtype probability
            wt_token_id = self._tokenizer.convert_tokens_to_ids(wt_aa)
            wt_prob = probs[wt_token_id].item()

            # Score each mutation
            for mut_aa in AA_LIST:
                if mut_aa == wt_aa:
                    continue
                mut_token_id = self._tokenizer.convert_tokens_to_ids(mut_aa)
                mut_prob = probs[mut_token_id].item()

                # Log-likelihood ratio: positive = mutation favored, negative = deleterious
                import math
                score = math.log(mut_prob / max(wt_prob, 1e-10))

                results.append({
                    "position": pos,
                    "wildtype": wt_aa,
                    "mutant": mut_aa,
                    "score": round(score, 4),
                    "wildtype_prob": round(wt_prob, 6),
                    "mutant_prob": round(mut_prob, 6),
                })

        results.sort(key=lambda x: x["score"])
        return results

    @torch.no_grad()
    def sensitivity_profile(self, sequence: str) -> list[dict]:
        """Compute per-position mutation sensitivity.

        For each position, returns the average effect of all possible mutations.
        High sensitivity = important residue (likely conserved/functional).

        Args:
            sequence: Amino acid sequence

        Returns:
            List of dicts: {position, residue, mean_score, min_score}
            Sorted by position
        """
        mutations = self.mutation_scan(sequence)

        # Group by position
        by_pos: dict[int, list[float]] = {}
        for m in mutations:
            pos = m["position"]
            if pos not in by_pos:
                by_pos[pos] = []
            by_pos[pos].append(m["score"])

        profile = []
        for pos in sorted(by_pos.keys()):
            scores = by_pos[pos]
            profile.append({
                "position": pos,
                "residue": sequence[pos],
                "mean_score": round(sum(scores) / len(scores), 4),
                "min_score": round(min(scores), 4),  # most deleterious mutation
            })

        return profile

    @torch.no_grad()
    def similarity(self, seq1: str, seq2: str) -> float:
        """Compute cosine similarity between two protein sequences."""
        emb1 = self.embed(seq1)["mean"]
        emb2 = self.embed(seq2)["mean"]
        return torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item()
