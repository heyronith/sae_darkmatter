"""Duplication counts via the infini-gram HTTP API (text queries)."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests

API_URL = "https://api.infini-gram.io/"
# Full Pile-train index (Llama-2 tokenizer). No public deduped-Pile index found.
PRIMARY_INDEX = "v4_piletrain_llama"
# Documented absence: no deduped-Pile train index on the public API as of P1.3.
DEDUPE_INDEX: str | None = None

_throttle_lock = threading.Lock()
_last_request_mono = 0.0


def _cache_key(index: str, query: str) -> str:
    h = hashlib.sha256(f"{index}\0{query}".encode()).hexdigest()
    return h


def count_ngram(
    query: str,
    *,
    index: str = PRIMARY_INDEX,
    cache_dir: str | Path,
    timeout_s: float = 60.0,
    min_interval_s: float = 0.05,
    _last_call: list[float] | None = None,
) -> dict[str, Any]:
    """Count occurrences of ``query`` text in an infini-gram index.

    Caches JSON responses under ``cache_dir`` keyed by sha256(index, query).
    Throttles outbound requests to about one per ``min_interval_s``.
    """
    del _last_call  # kept for call-site compatibility; module lock is used instead
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = _cache_key(index, query)
    path = cache / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())

    payload = {"index": index, "query_type": "count", "query": query}
    max_attempts = 8
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        global _last_request_mono
        with _throttle_lock:
            elapsed = time.monotonic() - _last_request_mono
            if elapsed < min_interval_s:
                time.sleep(min_interval_s - elapsed)
            _last_request_mono = time.monotonic()

        try:
            resp = requests.post(API_URL, json=payload, timeout=timeout_s)
            if resp.status_code in {403, 429, 500, 502, 503, 504}:
                wait = min(120.0, (2**attempt) * 2.0)
                time.sleep(wait)
                last_err = requests.HTTPError(
                    f"{resp.status_code} for {API_URL} (attempt {attempt + 1})",
                    response=resp,
                )
                continue
            resp.raise_for_status()
            data = resp.json()
            data["_meta"] = {"index": index, "query": query}
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(path)
            return data
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_err = exc
            time.sleep(min(120.0, (2**attempt) * 2.0))
    assert last_err is not None
    raise last_err


def stratum_for_k(k: int) -> str:
    if k <= 5:
        return "low"
    if k <= 31:
        return "mid"
    return "high"


def summarize_counts(ks: list[int]) -> dict[str, Any]:
    hist: dict[str, int] = {}
    strata = {"low": 0, "mid": 0, "high": 0, "k_le_2": 0}
    for k in ks:
        hist[str(k)] = hist.get(str(k), 0) + 1
        strata[stratum_for_k(k)] += 1
        if k <= 2:
            strata["k_le_2"] += 1
    return {"histogram_k": hist, "strata": strata, "n": len(ks)}
