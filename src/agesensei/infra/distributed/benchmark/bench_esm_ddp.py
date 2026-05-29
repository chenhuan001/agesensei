"""ESM-2 650M 2-GPU DDP benchmark: inference + train step."""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def setup_dist():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def teardown_dist():
    dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def cuda_sync_all():
    torch.cuda.synchronize()
    dist.barrier()


def all_reduce_probe_ms(numel, device):
    t = torch.randn(numel, device=device)
    cuda_sync_all()
    with cuda_timer(device) as get_ms:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
    return get_ms()


def load_model(dtype, local_rank):
    gc.collect()
    torch.cuda.empty_cache()
    reset_peak_memory(local_rank)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype
    ).cuda(local_rank).eval()
    return model


def bench_infer(model_ddp, tokenizer, global_batch, seq_len,
                dtype_name, warmup, iters, rank, world, local_rank):
    if global_batch % world != 0:
        return None
    per_rank = global_batch // world
    try:
        batch = make_random_protein_batch(
            per_rank, seq_len, tokenizer,
            device=f"cuda:{local_rank}", seed=42 + rank,
        )
        with torch.no_grad():
            for _ in range(warmup):
                _ = model_ddp(**batch)
        cuda_sync_all()
        reset_peak_memory(local_rank)

        total_ms = 0.0
        with torch.no_grad():
            for _ in range(iters):
                cuda_sync_all()
                with cuda_timer(local_rank) as get_ms:
                    _ = model_ddp(**batch)
                    torch.cuda.synchronize(local_rank)
                dist.barrier()
                total_ms += get_ms()
        avg_ms = total_ms / iters
        gt = global_batch / (avg_ms / 1000.0)
        return BenchResult(
            label=f"ddp{world}_infer_{dtype_name}",
            batch_size=global_batch,
            seq_len=seq_len,
            dtype=dtype_name,
            forward_ms=avg_ms,
            throughput_seq_per_s=gt,
            throughput_tokens_per_s=gt * seq_len,
            peak_mem_gb=peak_memory_gb(local_rank),
            allocated_mem_gb=allocated_memory_gb(local_rank),
            extras={"per_rank_batch": per_rank, "world": world},
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if is_main(rank):
            print(f"    ! OOM at global_batch={global_batch}", flush=True)
        return None


def bench_train_step(model_ddp, tokenizer, global_batch, seq_len,
                     dtype_name, warmup, iters, rank, world, local_rank):
    if global_batch % world != 0:
        return None
    per_rank = global_batch // world
    try:
        params = [p for p in model_ddp.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-5)
        batch = make_random_protein_batch(
            per_rank, seq_len, tokenizer,
            device=f"cuda:{local_rank}", seed=42 + rank,
        )
        labels = batch["input_ids"].clone()

        def one_step():
            optimizer.zero_grad(set_to_none=True)
            with cuda_timer(local_rank) as fwd:
                out = model_ddp(**batch, labels=labels)
                torch.cuda.synchronize(local_rank)
            f = fwd()
            with cuda_timer(local_rank) as bwd:
                out.loss.backward()
                torch.cuda.synchronize(local_rank)
            b = bwd()
            with cuda_timer(local_rank) as opt:
                optimizer.step()
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
            label=f"ddp{world}_train_{dtype_name}",
            batch_size=global_batch,
            seq_len=seq_len,
            dtype=dtype_name,
            forward_ms=step_ms,
            throughput_seq_per_s=gt,
            throughput_tokens_per_s=gt * seq_len,
            peak_mem_gb=peak_memory_gb(local_rank),
            allocated_mem_gb=allocated_memory_gb(local_rank),
            extras={
                "per_rank_batch": per_rank,
                "world": world,
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
    ap.add_argument("--mode", choices=["infer", "train"], default="infer")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--dtypes", nargs="+", default=["fp32", "fp16", "bf16"],
                    choices=list(DTYPE_MAP.keys()))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("./artifacts/benchmark"))
    args = ap.parse_args()

    rank, world, local_rank = setup_dist()
    if is_main(rank):
        print(f"World: {world}  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"NCCL: {torch.cuda.nccl.version()}  Mode: {args.mode}")
        for size_m in [1, 10, 100]:
            ms = all_reduce_probe_ms(size_m * 1_000_000, local_rank)
            bw = (size_m * 4) / (ms / 1000.0) / 1024
            print(f"  all-reduce {size_m}M floats ({size_m*4}MB): "
                  f"{ms:.2f} ms  bus_bw={bw:.1f} GB/s")
        print()
    else:
        for size_m in [1, 10, 100]:
            all_reduce_probe_ms(size_m * 1_000_000, local_rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results = []
    for dtype_name in args.dtypes:
        if args.mode == "train" and dtype_name == "fp16":
            if is_main(rank):
                print("(skip fp16 train: needs GradScaler)")
            continue
        if is_main(rank):
            print(f"=== dtype={dtype_name} ===")
        dtype = DTYPE_MAP[dtype_name]
        model = load_model(dtype, local_rank)
        if args.mode == "train":
            model.train()
        model_ddp = DDP(
            model, device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        for bs in args.batch_sizes:
            if bs % world != 0:
                continue
            if is_main(rank):
                print(f"  global_batch={bs:>4d} (per_rank={bs//world}) ...",
                      end=" ", flush=True)
            if args.mode == "infer":
                r = bench_infer(model_ddp, tokenizer, bs, args.seq_len,
                                dtype_name, args.warmup, args.iters,
                                rank, world, local_rank)
            else:
                r = bench_train_step(model_ddp, tokenizer, bs, args.seq_len,
                                     dtype_name, args.warmup, args.iters,
                                     rank, world, local_rank)
            if r is None:
                if is_main(rank):
                    print("OOM/skip", flush=True)
                break
            if is_main(rank):
                if args.mode == "train":
                    e = r.extras
                    print(f"step={r.forward_ms:.1f}ms (fwd {e['fwd_ms']} / bwd {e['bwd_ms']} / opt {e['opt_ms']}) "
                          f"{r.throughput_seq_per_s:.1f} seq/s  peak {r.peak_mem_gb:.2f} GB")
                else:
                    print(f"{r.forward_ms:.1f}ms  {r.throughput_seq_per_s:.1f} seq/s  "
                          f"peak {r.peak_mem_gb:.2f} GB")
                results.append(r)
        del model, model_ddp
        gc.collect()
        torch.cuda.empty_cache()

    if is_main(rank) and results:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        md_path = args.output_dir / f"benchmark_ddp_{args.mode}_{ts}.md"
        json_path = args.output_dir / f"benchmark_ddp_{args.mode}_{ts}.json"
        rows = [r.to_row() for r in results]
        if args.mode == "train":
            extras_cols = ["fwd_ms", "bwd_ms", "opt_ms"]
            header = BenchResult.header() + extras_cols
            for i, r in enumerate(results):
                rows[i] += [r.extras.get(c, "-") for c in extras_cols]
        else:
            header = BenchResult.header()
        table = print_markdown_table(rows, header)
        gpu_name = torch.cuda.get_device_name(local_rank)
        md = f"# ESM-2 650M  DDP x {world} - {gpu_name}\n\n"
        md += f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        md += f"**Mode**: `{args.mode}`\n"
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
        print(f"JSON:   {json_path}")
    teardown_dist()


if __name__ == "__main__":
    main()
