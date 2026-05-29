"""Benchmark 工具：计时、显存、序列数据生成。"""
from __future__ import annotations

import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class BenchResult:
    label: str
    batch_size: int
    seq_len: int
    dtype: str
    forward_ms: float
    throughput_seq_per_s: float
    throughput_tokens_per_s: float
    peak_mem_gb: float
    allocated_mem_gb: float
    extras: dict = field(default_factory=dict)

    def to_row(self) -> list:
        return [
            self.label, self.batch_size, self.seq_len, self.dtype,
            f"{self.forward_ms:.1f}",
            f"{self.throughput_seq_per_s:.1f}",
            f"{self.throughput_tokens_per_s:.0f}",
            f"{self.peak_mem_gb:.2f}",
            f"{self.allocated_mem_gb:.2f}",
        ]

    @staticmethod
    def header() -> list[str]:
        return ["label", "batch", "seq_len", "dtype",
                "fwd_ms", "seq/s", "tok/s",
                "peak_mem_GB", "alloc_mem_GB"]


@contextmanager
def cuda_timer(device: int = 0):
    """精确 CUDA 计时（async-aware）。"""
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    try:
        yield lambda: (time.perf_counter() - t0) * 1000.0
    finally:
        torch.cuda.synchronize(device)


def reset_peak_memory(device: int = 0):
    torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: int = 0) -> float:
    return torch.cuda.max_memory_allocated(device) / 1024**3


def allocated_memory_gb(device: int = 0) -> float:
    return torch.cuda.memory_allocated(device) / 1024**3


def make_random_protein_batch(
    batch_size: int,
    seq_len: int,
    tokenizer,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """生成固定长度的随机蛋白 batch（20 标准氨基酸）。"""
    rng = random.Random(seed)
    aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
    seqs = [
        "".join(rng.choice(aa_vocab) for _ in range(seq_len))
        for _ in range(batch_size)
    ]
    enc = tokenizer(
        seqs,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len + 2,  # ESM 会加 <cls>/<eos>
    )
    return {k: v.to(device) for k, v in enc.items()}


def print_markdown_table(rows: list[list], header: list[str]) -> str:
    """rows: list of row lists (strings)."""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)
