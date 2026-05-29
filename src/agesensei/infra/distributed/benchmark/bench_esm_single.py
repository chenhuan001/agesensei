"""ESM-2 650M 单卡 baseline benchmark。

产出：benchmark_single.md

用法：
    # Default config
    python -m agesensei.infra.distributed.benchmark.bench_esm_single

    # 自定义
    python -m agesensei.infra.distributed.benchmark.bench_esm_single \
        --seq-len 512 --warmup 3 --iters 10 \
        --dtypes fp32 fp16 bf16 \
        --batch-sizes 1 2 4 8 16
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import sys
from pathlib import Path

import torch
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

import os as _os
MODEL_NAME = _os.environ.get("ESM_MODEL_PATH", "facebook/esm2_t33_650M_UR50D")

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def load_model(dtype: torch.dtype):
    """清空显存后重新加载模型。"""
    gc.collect()
    torch.cuda.empty_cache()
    reset_peak_memory()

    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    ).cuda().eval()
    return model


def bench_forward(
    model,
    tokenizer,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    dtype_name: str,
    warmup: int,
    iters: int,
) -> BenchResult | None:
    """测单次 batch forward。返回 None 表示 OOM。"""
    try:
        batch = make_random_protein_batch(batch_size, seq_len, tokenizer)

        # warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(**batch)
        torch.cuda.synchronize()
        reset_peak_memory()

        # 正式计时
        total_ms = 0.0
        with torch.no_grad():
            for _ in range(iters):
                with cuda_timer() as get_ms:
                    _ = model(**batch)
                total_ms += get_ms()

        avg_ms = total_ms / iters
        throughput_seq = batch_size / (avg_ms / 1000.0)
        throughput_tok = throughput_seq * seq_len

        return BenchResult(
            label=f"single_{dtype_name}",
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=dtype_name,
            forward_ms=avg_ms,
            throughput_seq_per_s=throughput_seq,
            throughput_tokens_per_s=throughput_tok,
            peak_mem_gb=peak_memory_gb(),
            allocated_mem_gb=allocated_memory_gb(),
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"    ! OOM at batch={batch_size}, stopping larger batches for this dtype.", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=512,
                    help="蛋白序列长度（ESM-2 训练时 1024，常用 512）")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--dtypes", nargs="+",
                    default=["fp32", "fp16", "bf16"],
                    choices=list(DTYPE_MAP.keys()))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("./artifacts/benchmark"))
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    gpu_name = torch.cuda.get_device_name(args.gpu)
    print(f"Device: GPU {args.gpu} ({gpu_name})")
    print(f"Model:  {MODEL_NAME}")
    print(f"SeqLen: {args.seq_len}, warmup={args.warmup}, iters={args.iters}")
    print("")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results: list[BenchResult] = []

    for dtype_name in args.dtypes:
        print(f"=== dtype={dtype_name} ===")
        dtype = DTYPE_MAP[dtype_name]
        model = load_model(dtype)
        for bs in args.batch_sizes:
            print(f"  batch={bs:>4d} ...", end=" ", flush=True)
            r = bench_forward(
                model, tokenizer,
                batch_size=bs,
                seq_len=args.seq_len,
                dtype=dtype,
                dtype_name=dtype_name,
                warmup=args.warmup,
                iters=args.iters,
            )
            if r is None:
                break
            print(f"{r.forward_ms:.1f} ms  {r.throughput_seq_per_s:.1f} seq/s  "
                  f"peak {r.peak_mem_gb:.2f} GB")
            results.append(r)
        # 释放当前 dtype 的模型
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("")

    # 写 markdown 报告
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    md_path = args.output_dir / f"benchmark_single_{ts}.md"
    json_path = args.output_dir / f"benchmark_single_{ts}.json"

    table = print_markdown_table(
        [r.to_row() for r in results],
        BenchResult.header(),
    )

    md = f"""# ESM-2 650M 单卡 Baseline · {gpu_name}

**日期**：{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
**模型**：`{MODEL_NAME}`
**GPU**：{gpu_name}
**序列长度**：{args.seq_len} aa
**Warmup / Iters**：{args.warmup} / {args.iters}

## 结果

{table}

## 关键观察（待人工填）

- fp16 vs fp32 显存节省：
- fp16 vs fp32 吞吐提升：
- bf16 vs fp16 差异：
- 能跑的最大 batch（fp16）：
- 峰值吞吐：

## 下一步（Day 2）

- 2 卡 DDP，同样 seq_len / batch 测试，看 all-reduce 开销
- 画通信拓扑图
"""

    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(
        json.dumps([r.__dict__ for r in results], indent=2),
        encoding="utf-8",
    )

    print(f"✅ Report: {md_path}")
    print(f"✅ JSON:   {json_path}")


if __name__ == "__main__":
    main()
