"""ESM-2 650M DeepSpeed ZeRO-1/2/3 benchmark.

Launch:
    deepspeed --num_gpus=2 -m agesensei.infra.distributed.benchmark.bench_esm_deepspeed \
        --zero-stage 3 \
        --batch-sizes 2 4 8 \
        --iters 5
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
from transformers import AutoModelForMaskedLM, AutoTokenizer

from agesensei.infra.distributed.benchmark.utils import (
    BenchResult,
    allocated_memory_gb,
    cuda_timer,
    make_random_protein_batch,
    peak_memory_gb,
    print_markdown_table,
    reset_peak_memory,
)

MODEL_NAME = os.environ.get("ESM_MODEL_PATH", "facebook/esm2_t33_650M_UR50D")

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def is_main(rank):
    return rank == 0


def cuda_sync_all():
    torch.cuda.synchronize()
    dist.barrier()


def load_ds_config(stage, micro_batch):
    cfg_path = CONFIG_DIR / f"ds_zero{stage}.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["train_micro_batch_size_per_gpu"] = micro_batch
    return cfg


def bench_train_step(engine, tokenizer, global_batch, seq_len,
                     warmup, iters, rank, world, local_rank, stage):
    if global_batch % world != 0:
        return None
    per_rank = global_batch // world
    try:
        batch = make_random_protein_batch(
            per_rank, seq_len, tokenizer,
            device=f"cuda:{local_rank}", seed=42 + rank,
        )
        labels = batch["input_ids"].clone()

        def one_step():
            with cuda_timer(local_rank) as fwd:
                out = engine(**batch, labels=labels)
                torch.cuda.synchronize(local_rank)
            f = fwd()
            with cuda_timer(local_rank) as bwd:
                engine.backward(out.loss)
                torch.cuda.synchronize(local_rank)
            b = bwd()
            with cuda_timer(local_rank) as opt:
                engine.step()
                torch.cuda.synchronize(local_rank)
            o = opt()
            return f, b, o

        for _ in range(warmup):
            one_step()
        cuda_sync_all()
        reset_peak_memory(local_rank)

        f_total = b_total = o_total = 0.0
        for _ in range(iters):
            cuda_sync_all()
            f, b, o = one_step()
            dist.barrier()
            f_total += f
            b_total += b
            o_total += o
        f_avg = f_total / iters
        b_avg = b_total / iters
        o_avg = o_total / iters
        step_ms = f_avg + b_avg + o_avg
        gt = global_batch / (step_ms / 1000.0)
        return BenchResult(
            label=f"ds_zero{stage}_bf16",
            batch_size=global_batch,
            seq_len=seq_len,
            dtype="bf16",
            forward_ms=step_ms,
            throughput_seq_per_s=gt,
            throughput_tokens_per_s=gt * seq_len,
            peak_mem_gb=peak_memory_gb(local_rank),
            allocated_mem_gb=allocated_memory_gb(local_rank),
            extras={
                "per_rank_batch": per_rank,
                "world": world,
                "stage": stage,
                "fwd_ms": round(f_avg, 2),
                "bwd_ms": round(b_avg, 2),
                "opt_ms": round(o_avg, 2),
            },
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if is_main(rank):
            print(f"    ! OOM at global_batch={global_batch}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero-stage", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--output-dir", type=Path,
                    default=Path("./artifacts/benchmark"))
    ap.add_argument("--local_rank", type=int, default=-1)
    args = ap.parse_args()

    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)

    if is_main(rank):
        print(f"World: {world}  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"DeepSpeed: {deepspeed.__version__}  ZeRO stage: {args.zero_stage}")
        print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results = []

    for bs in args.batch_sizes:
        if bs % world != 0:
            continue
        per_rank = bs // world

        gc.collect()
        torch.cuda.empty_cache()
        reset_peak_memory(local_rank)

        if is_main(rank):
            print(f"  global_batch={bs:>4d} (per_rank={per_rank}) ...",
                  end=" ", flush=True)

        ds_cfg = load_ds_config(args.zero_stage, per_rank)
        model = AutoModelForMaskedLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16
        )
        model.train()
        # Use native AdamW to avoid DeepSpeed FusedAdam JIT (needs CUDA_HOME)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=model.parameters(),
            config=ds_cfg,
        )

        r = bench_train_step(
            engine, tokenizer, bs, args.seq_len,
            args.warmup, args.iters, rank, world, local_rank, args.zero_stage,
        )

        del engine, model
        gc.collect()
        torch.cuda.empty_cache()

        if r is None:
            if is_main(rank):
                print("OOM/skip", flush=True)
            continue
        if is_main(rank):
            e = r.extras
            print(f"step={r.forward_ms:.1f}ms (fwd {e['fwd_ms']} / bwd {e['bwd_ms']} / opt {e['opt_ms']}) "
                  f"{r.throughput_seq_per_s:.1f} seq/s  peak {r.peak_mem_gb:.2f} GB")
            results.append(r)

    if is_main(rank) and results:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        md_path = args.output_dir / f"benchmark_ds_zero{args.zero_stage}_{ts}.md"
        json_path = args.output_dir / f"benchmark_ds_zero{args.zero_stage}_{ts}.json"
        rows = [r.to_row() for r in results]
        extras_cols = ["fwd_ms", "bwd_ms", "opt_ms"]
        header = BenchResult.header() + extras_cols
        for i, r in enumerate(results):
            rows[i] += [r.extras.get(c, "-") for c in extras_cols]
        table = print_markdown_table(rows, header)
        gpu_name = torch.cuda.get_device_name(local_rank)
        md = f"# ESM-2 650M DeepSpeed ZeRO-{args.zero_stage} - {gpu_name}\n\n"
        md += f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        md += f"**Stage**: ZeRO-{args.zero_stage}\n"
        md += f"**Model**: `{MODEL_NAME}`\n"
        md += f"**World**: {world}\n"
        md += f"**SeqLen**: {args.seq_len}\n"
        md += f"**Warmup/Iters**: {args.warmup}/{args.iters}\n\n"
        md += "## Results\n\n" + table + "\n"
        md_path.write_text(md, encoding="utf-8")
        json_path.write_text(
            json.dumps([{**r.__dict__} for r in results], indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nReport: {md_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
