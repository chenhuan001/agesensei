"""Pre-build full-text cache for LitQA2 evaluation questions.

Downloads full papers via DOI → PMC/Unpaywall for all 50 evaluation questions,
so the actual evaluation run doesn't need to wait for downloads.
"""
import asyncio
import sys
sys.path.insert(0, "src")

from datasets import load_dataset
from agesensei.eval.lab_bench_adapter import (
    _retrieve_from_dois, _cache, RetrievedChunk
)


async def prebuild():
    ds = load_dataset("futurehouse/lab-bench", "LitQA2", split="train")

    # Take first 50 questions (same as eval)
    max_q = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    total_chunks = 0
    fulltext_count = 0
    abstract_only = 0

    for i, row in enumerate(ds):
        if i >= max_q:
            break

        sources = row.get("sources", [])
        if not sources:
            print(f"Q{i+1}: no sources, skipping", flush=True)
            continue

        doi = sources[0].replace("https://doi.org/", "").replace("https://dx.doi.org/", "").strip()

        # Check if already cached
        cached_pmc = _cache.get_pmc(f"unpaywall:{doi}")
        if cached_pmc is not None:
            n = sum(1 for v in cached_pmc.values() if len(v) > 50)
            print(f"Q{i+1}: DOI {doi} — cached ({n} sections)", flush=True)
            total_chunks += n
            fulltext_count += 1
            continue

        # Try to fetch
        try:
            chunks = await _retrieve_from_dois(sources)
            n = len(chunks)
            total_chunks += n

            if any("abstract-only" in c.source for c in chunks):
                abstract_only += 1
                print(f"Q{i+1}: DOI {doi} — abstract only ({n} chunks)", flush=True)
            elif n > 0:
                fulltext_count += 1
                print(f"Q{i+1}: DOI {doi} — full-text ({n} chunks)", flush=True)
            else:
                print(f"Q{i+1}: DOI {doi} — FAILED (0 chunks)", flush=True)
        except Exception as e:
            print(f"Q{i+1}: DOI {doi} — ERROR: {e}", flush=True)

        # Rate limit
        await asyncio.sleep(0.5)

    print(f"\n=== Cache Summary ===", flush=True)
    print(f"Total questions: {min(max_q, len(ds))}", flush=True)
    print(f"Full-text: {fulltext_count}", flush=True)
    print(f"Abstract-only: {abstract_only}", flush=True)
    print(f"Total chunks cached: {total_chunks}", flush=True)


if __name__ == "__main__":
    asyncio.run(prebuild())
