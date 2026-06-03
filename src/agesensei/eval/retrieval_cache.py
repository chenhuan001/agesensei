"""Local disk cache for PMC full-text and PubMed abstracts.

Avoids redundant NCBI/S2 API calls across evaluation runs.
Cache is stored as JSON files in a local directory, keyed by identifier.

Usage:
    cache = RetrievalCache("artifacts/cache")

    # Check cache first
    cached = cache.get_pmc(pmcid)
    if cached is None:
        sections = await fetch_full_text(pmcid)
        cache.put_pmc(pmcid, sections)

    # PubMed abstracts
    cached = cache.get_pubmed(pmid)
    if cached is None:
        paper = await fetch_paper(pmid)
        cache.put_pubmed(pmid, {"title": paper.title, "abstract": paper.abstract})
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


class RetrievalCache:
    """Disk-based cache for retrieval results.

    Structure:
        cache_dir/
            pmc/          # PMC full-text sections by PMCID
            pubmed/       # PubMed abstracts by PMID
            s2/           # Semantic Scholar results by query hash
            queries/      # PubMed search results by query hash
    """

    def __init__(self, cache_dir: str = "artifacts/cache"):
        self.cache_dir = Path(cache_dir)
        for subdir in ["pmc", "pubmed", "s2", "queries"]:
            (self.cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _hash_key(self, key: str) -> str:
        """Create a filesystem-safe hash for a cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    # --- PMC full-text ---

    def get_pmc(self, pmcid: str) -> dict[str, str] | None:
        """Get cached PMC full-text sections."""
        path = self.cache_dir / "pmc" / f"{pmcid}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return data.get("sections")
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def put_pmc(self, pmcid: str, sections: dict[str, str]) -> None:
        """Cache PMC full-text sections."""
        path = self.cache_dir / "pmc" / f"{pmcid}.json"
        path.write_text(json.dumps({
            "pmcid": pmcid,
            "sections": sections,
            "cached_at": time.time(),
        }, ensure_ascii=False))

    # --- PubMed abstracts ---

    def get_pubmed(self, pmid: str) -> dict[str, str] | None:
        """Get cached PubMed paper (title + abstract)."""
        path = self.cache_dir / "pubmed" / f"{pmid}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def put_pubmed(self, pmid: str, data: dict[str, str]) -> None:
        """Cache PubMed paper data."""
        path = self.cache_dir / "pubmed" / f"{pmid}.json"
        data["cached_at"] = time.time()
        path.write_text(json.dumps(data, ensure_ascii=False))

    # --- Semantic Scholar ---

    def get_s2(self, query: str) -> list[dict] | None:
        """Get cached S2 search results."""
        key = self._hash_key(query)
        path = self.cache_dir / "s2" / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return data.get("results")
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def put_s2(self, query: str, results: list[dict]) -> None:
        """Cache S2 search results."""
        key = self._hash_key(query)
        path = self.cache_dir / "s2" / f"{key}.json"
        path.write_text(json.dumps({
            "query": query,
            "results": results,
            "cached_at": time.time(),
        }, ensure_ascii=False))

    # --- PubMed search queries ---

    def get_query(self, query: str) -> list[str] | None:
        """Get cached PubMed search PMIDs."""
        key = self._hash_key(query)
        path = self.cache_dir / "queries" / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return data.get("pmids")
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def put_query(self, query: str, pmids: list[str]) -> None:
        """Cache PubMed search PMIDs."""
        key = self._hash_key(query)
        path = self.cache_dir / "queries" / f"{key}.json"
        path.write_text(json.dumps({
            "query": query,
            "pmids": pmids,
            "cached_at": time.time(),
        }, ensure_ascii=False))

    # --- Stats ---

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            subdir: len(list((self.cache_dir / subdir).glob("*.json")))
            for subdir in ["pmc", "pubmed", "s2", "queries"]
        }
