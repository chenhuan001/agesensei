"""ESM-2 fine-tuning trainer with DDP/DeepSpeed support.

Implements domain-adaptive fine-tuning of ESM-2 on aging-related proteins
to improve mutation effect prediction for senescence-associated targets.

Usage:
    # Single GPU MLM fine-tuning
    python -m agesensei.infra.distributed.finetune.trainer \
        --data data/genage_sequences.csv --task mlm --epochs 5

    # Multi-GPU DDP
    torchrun --nproc_per_node=4 -m agesensei.infra.distributed.finetune.trainer \
        --data data/genage_sequences.csv --task mlm --epochs 5

    # DeepSpeed ZeRO-2
    deepspeed -m agesensei.infra.distributed.finetune.trainer \
        --data data/genage_sequences.csv --task mlm --epochs 5 --deepspeed
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_cosine_schedule_with_warmup

from agesensei.infra.distributed.finetune.dataset import AgingMutationDataset


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed():
    """Initialize distributed training if launched with torchrun/deepspeed."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


class ESMFineTuner:
    """Fine-tune ESM-2 for aging-related mutation prediction.

    Supports:
        - Masked Language Modeling (MLM) on aging protein sequences
        - Mutation effect classification (deleterious vs neutral)
        - DDP and DeepSpeed distributed training
        - Gradient accumulation for large effective batch sizes
        - Cosine learning rate schedule with warmup
        - Checkpoint saving and resumption
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        task: str = "mlm",
        output_dir: str = "artifacts/finetune",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_length: int = 512,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        epochs: int = 5,
        use_deepspeed: bool = False,
        deepspeed_config: str | None = None,
        fp16: bool = False,
        bf16: bool = False,
        save_every_n_steps: int = 500,
        eval_every_n_steps: int = 100,
        local_rank: int = 0,
    ):
        self.model_name = model_name
        self.task = task
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_length = max_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = deepspeed_config
        self.fp16 = fp16
        self.bf16 = bf16
        self.save_every_n_steps = save_every_n_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.local_rank = local_rank

        self.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        self.global_step = 0
        self.best_val_loss = float("inf")

    def load_model(self):
        """Load ESM-2 model and tokenizer."""
        if is_main_process():
            print(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.bfloat16 if self.bf16 else (torch.float16 if self.fp16 else torch.float32)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name, torch_dtype=dtype
        )
        self.model.to(self.device)

        if is_main_process():
            n_params = sum(p.numel() for p in self.model.parameters())
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"  Parameters: {n_params/1e6:.1f}M total, {n_trainable/1e6:.1f}M trainable")

    def prepare_data(self, train_path: str, val_path: str | None = None, val_split: float = 0.1):
        """Prepare train and validation dataloaders."""
        train_dataset = AgingMutationDataset(
            data_path=train_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            task=self.task,
        )

        if val_path:
            val_dataset = AgingMutationDataset(
                data_path=val_path,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                task=self.task,
            )
        else:
            n_val = max(1, int(len(train_dataset) * val_split))
            n_train = len(train_dataset) - n_val
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [n_train, n_val]
            )

        train_sampler = DistributedSampler(train_dataset) if is_distributed() else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed() else None

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size * 2,
            sampler=val_sampler,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        if is_main_process():
            print(f"  Train samples: {len(train_dataset)}")
            print(f"  Val samples: {len(val_dataset)}")

    def setup_optimizer(self):
        """Configure optimizer and learning rate schedule."""
        # Separate weight decay for different parameter groups
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        param_groups = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)

        total_steps = (len(self.train_loader) // self.gradient_accumulation_steps) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        if is_main_process():
            print(f"  Total steps: {total_steps}, warmup: {warmup_steps}")

    def wrap_model(self):
        """Wrap model with DDP or DeepSpeed."""
        if self.use_deepspeed:
            import deepspeed
            ds_config = self._get_deepspeed_config()
            self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
                config=ds_config,
            )
        elif is_distributed():
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )

    def _get_deepspeed_config(self) -> dict:
        """Load or generate DeepSpeed config."""
        if self.deepspeed_config:
            with open(self.deepspeed_config) as f:
                return json.load(f)

        return {
            "train_batch_size": self.batch_size * self.gradient_accumulation_steps * get_world_size(),
            "train_micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "betas": [0.9, 0.999],
                },
            },
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "warmup_num_steps": int(
                        len(self.train_loader) * self.epochs * self.warmup_ratio
                        / self.gradient_accumulation_steps
                    ),
                    "total_num_steps": len(self.train_loader) * self.epochs // self.gradient_accumulation_steps,
                },
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "none"},
                "allgather_partitions": True,
                "reduce_scatter": True,
                "overlap_comm": True,
            },
            "bf16": {"enabled": self.bf16},
            "fp16": {"enabled": self.fp16 and not self.bf16},
            "gradient_clipping": 1.0,
        }

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        for step, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            if self.task == "mlm":
                outputs = self._forward_mlm(batch)
            else:
                outputs = self._forward_mutation_cls(batch)

            loss = outputs["loss"] / self.gradient_accumulation_steps

            if self.use_deepspeed:
                self.model.backward(loss)
            else:
                loss.backward()

            total_loss += loss.item() * self.gradient_accumulation_steps
            n_batches += 1

            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_deepspeed:
                    self.model.step()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if is_main_process() and self.global_step % 10 == 0:
                    lr = self.scheduler.get_last_lr()[0] if not self.use_deepspeed else self.learning_rate
                    print(f"  step {self.global_step:>5d} | loss {loss.item() * self.gradient_accumulation_steps:.4f} | lr {lr:.2e}")

                # Evaluation
                if self.global_step % self.eval_every_n_steps == 0:
                    val_loss = self.evaluate()
                    if is_main_process() and val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best")
                    self.model.train()

                # Checkpoint
                if is_main_process() and self.global_step % self.save_every_n_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

        return total_loss / max(n_batches, 1)

    def _forward_mlm(self, batch: dict) -> dict:
        """Forward pass for MLM task."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    def _forward_mutation_cls(self, batch: dict) -> dict:
        """Forward pass for mutation classification.

        Uses the masked position's predicted probability to classify mutations.
        Loss = cross-entropy between predicted token distribution and actual outcome.
        """
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits  # (B, seq_len, vocab_size)

        # Extract logits at mutation positions
        positions = batch["position"]  # (B,)
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        position_logits = logits[batch_indices, positions]  # (B, vocab_size)

        # Compute log-likelihood ratio: P(mutant) / P(wildtype)
        probs = torch.softmax(position_logits, dim=-1)
        wt_probs = probs[batch_indices, batch["wildtype_id"]]
        mut_probs = probs[batch_indices, batch["mutant_id"]]

        # Score: log(P_mut / P_wt), negative = deleterious
        scores = torch.log(mut_probs + 1e-10) - torch.log(wt_probs + 1e-10)

        # Binary classification loss: deleterious mutations should have negative scores
        # Label 1 = deleterious (score should be negative), 0 = neutral (score ~0)
        labels = batch["label"].float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(-scores, labels)

        return {"loss": loss, "scores": scores}

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            if self.task == "mlm":
                outputs = self._forward_mlm(batch)
            else:
                outputs = self._forward_mutation_cls(batch)

            total_loss += outputs["loss"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if is_main_process():
            print(f"  [eval] step {self.global_step} | val_loss {avg_loss:.4f}")

        return avg_loss

    def save_checkpoint(self, tag: str):
        """Save model checkpoint."""
        save_dir = self.output_dir / tag
        save_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "epoch": getattr(self, "_current_epoch", 0),
        }
        torch.save(state, save_dir / "training_state.pt")
        print(f"  Checkpoint saved: {save_dir}")

    def train(self, train_path: str, val_path: str | None = None):
        """Full training loop."""
        self.load_model()
        self.prepare_data(train_path, val_path)
        self.setup_optimizer()
        self.wrap_model()

        if is_main_process():
            print(f"\n{'='*60}")
            print(f"ESM-2 Fine-tuning: {self.task.upper()}")
            print(f"Model: {self.model_name}")
            print(f"Device: {self.device} x {get_world_size()}")
            print(f"Epochs: {self.epochs}")
            print(f"Batch: {self.batch_size} x {self.gradient_accumulation_steps} accum x {get_world_size()} GPU")
            print(f"Effective batch: {self.batch_size * self.gradient_accumulation_steps * get_world_size()}")
            print(f"{'='*60}\n")

        for epoch in range(self.epochs):
            self._current_epoch = epoch
            if is_main_process():
                print(f"\n--- Epoch {epoch + 1}/{self.epochs} ---")

            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            elapsed = time.time() - t0

            if is_main_process():
                print(f"  Epoch {epoch + 1} done: train_loss={train_loss:.4f} time={elapsed:.1f}s")

            # End-of-epoch evaluation
            val_loss = self.evaluate()
            if is_main_process():
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")
                self.save_checkpoint(f"epoch_{epoch + 1}")

        if is_main_process():
            print(f"\nTraining complete. Best val_loss: {self.best_val_loss:.4f}")
            print(f"Best model: {self.output_dir / 'best'}")

        cleanup_distributed()


def main():
    ap = argparse.ArgumentParser(description="Fine-tune ESM-2 on aging-related proteins")
    ap.add_argument("--data", type=str, required=True, help="Training data CSV path")
    ap.add_argument("--val-data", type=str, default=None, help="Validation data CSV (optional)")
    ap.add_argument("--task", choices=["mlm", "mutation_cls"], default="mlm",
                    help="Fine-tuning task: mlm or mutation_cls")
    ap.add_argument("--model", type=str, default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--output-dir", type=str, default="artifacts/finetune")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed ZeRO-2")
    ap.add_argument("--ds-config", type=str, default=None, help="DeepSpeed config JSON")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    local_rank = setup_distributed()

    trainer = ESMFineTuner(
        model_name=args.model,
        task=args.task,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        epochs=args.epochs,
        use_deepspeed=args.deepspeed,
        deepspeed_config=args.ds_config,
        fp16=args.fp16,
        bf16=args.bf16,
        local_rank=local_rank,
    )
    trainer.train(args.data, args.val_data)


if __name__ == "__main__":
    main()
