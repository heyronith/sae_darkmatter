"""Duplication counts: precomputed NeoX 32-gram counts + token-aligned infini-gram."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import requests
from huggingface_hub import dataset_info as hf_dataset_info
from huggingface_hub import hf_hub_download

API_URL = "https://api.infini-gram.io/"
# Full Pile-train index (Llama-2 tokenizer). No public deduped-Pile index on the API.
PRIMARY_INDEX = "v4_piletrain_llama"
DEDUPE_INDEX: str | None = None

# Precomputed exact 32-gram duplicate counts (GPT-NeoX tokens) from Prashanth et al.
DEDUPED_COUNTS_DS = "usvsnsp/deduped-num-duplicates"
DEDUPED_COUNTS_FILE = "data/train-00000-of-00001-a51cc17920038302.parquet"
DUPED_COUNTS_DS = "usvsnsp/duped-num-duplicates"
DUPED_COUNTS_FILE = "data/train-00000-of-00001-0af281fb929b7d87.parquet"

GPT_NEOX_EOS_ID = 0
PROMPT_LEN = 32
CONT_LEN = 32
LLAMA_EDGE_DROP = 2
LLAMA_MIN_QUERY_LEN = 16
TRIVIAL_DISTINCT_TOKENS = 8
TRIVIAL_WHITESPACE_FRAC = 0.5

_throttle_lock = threading.Lock()
_last_request_mono = 0.0


def precomputed_dataset_meta() -> dict[str, Any]:
    """Provenance for the released per-sequence duplicate-count tables."""
    return {
        "deduped": {
            "dataset_id": DEDUPED_COUNTS_DS,
            "revision": hf_dataset_info(DEDUPED_COUNTS_DS).sha,
            "parquet": DEDUPED_COUNTS_FILE,
            "definition": (
                "Exact duplicate count of each 32-token (GPT-NeoX) training window "
                "in the near-deduplicated Pile; Index == Pile sequence id. "
                "Released with EleutherAI/semantic-memorization "
                "(commit 84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2)."
            ),
            "corpus": "Pile near-deduplicated (Pythia-deduped training corpus)",
            "window_tokens": 32,
            "match_type": "exact",
        },
        "duped": {
            "dataset_id": DUPED_COUNTS_DS,
            "revision": hf_dataset_info(DUPED_COUNTS_DS).sha,
            "parquet": DUPED_COUNTS_FILE,
            "definition": (
                "Same 32-token exact counts on the standard (non-deduped) Pile."
            ),
            "corpus": "Pile (full / duped)",
            "window_tokens": 32,
            "match_type": "exact",
        },
    }


def load_precomputed_counts(corpus: str = "deduped") -> np.ndarray:
    """Return Counts array indexed by Pile sequence id (length ~146.4M)."""
    if corpus == "deduped":
        ds, fname = DEDUPED_COUNTS_DS, DEDUPED_COUNTS_FILE
    elif corpus == "duped":
        ds, fname = DUPED_COUNTS_DS, DUPED_COUNTS_FILE
    else:
        raise ValueError(corpus)
    path = hf_hub_download(ds, fname, repo_type="dataset")
    table = pq.read_table(path, columns=["Counts"])
    return np.asarray(table.column("Counts"), dtype=np.int64)


def _cache_key_ids(index: str, query_ids: list[int]) -> str:
    raw = f"{index}\0" + ",".join(map(str, query_ids))
    return hashlib.sha256(raw.encode()).hexdigest()


def count_ngram_ids(
    query_ids: list[int],
    *,
    index: str = PRIMARY_INDEX,
    cache_dir: str | Path,
    timeout_s: float = 60.0,
    min_interval_s: float = 0.2,
) -> dict[str, Any]:
    """Count an n-gram given Llama-2 token ids (``query_ids`` API field)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = _cache_key_ids(index, query_ids)
    path = cache / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())

    payload: dict[str, Any] = {
        "index": index,
        "query_type": "count",
        "query_ids": query_ids,
    }
    max_attempts = 12
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
                wait = min(300.0, (2**attempt) * 3.0)
                print(
                    f"infini-gram {resp.status_code}; sleeping {wait:.0f}s "
                    f"(attempt {attempt + 1}/{max_attempts})",
                    flush=True,
                )
                time.sleep(wait)
                last_err = requests.HTTPError(
                    f"{resp.status_code} for {API_URL} (attempt {attempt + 1})",
                    response=resp,
                )
                continue
            resp.raise_for_status()
            data = resp.json()
            data["_meta"] = {"index": index, "query_ids": query_ids}
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(path)
            return data
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_err = exc
            time.sleep(min(300.0, (2**attempt) * 3.0))
    assert last_err is not None
    raise last_err


def count_ngram(
    query: str,
    *,
    index: str = PRIMARY_INDEX,
    cache_dir: str | Path,
    timeout_s: float = 60.0,
    min_interval_s: float = 0.05,
    _last_call: list[float] | None = None,
) -> dict[str, Any]:
    """Legacy text-query count (kept for P1.3 cache compatibility)."""
    del _last_call
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{index}\0{query}".encode()).hexdigest()
    path = cache / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    # Reuse id path after tokenization is preferred; text path still throttled.
    payload = {"index": index, "query_type": "count", "query": query}
    global _last_request_mono
    with _throttle_lock:
        elapsed = time.monotonic() - _last_request_mono
        if elapsed < min_interval_s:
            time.sleep(min_interval_s - elapsed)
        _last_request_mono = time.monotonic()
    resp = requests.post(API_URL, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    data["_meta"] = {"index": index, "query": query}
    path.write_text(json.dumps(data))
    return data


def neox_token_char_spans(neox_tokenizer: Any, token_ids: list[int]) -> list[tuple[int, int]]:
    """Character spans of each NeoX token inside ``decode(token_ids)``.

    Uses progressive prefix decoding so spans are consistent with full decode.
    """
    spans: list[tuple[int, int]] = []
    prev_text = ""
    for i in range(len(token_ids)):
        curr_text = neox_tokenizer.decode(token_ids[: i + 1])
        # Prefer prefix alignment; fall back to append length if decode is non-monotonic.
        if curr_text.startswith(prev_text):
            start, end = len(prev_text), len(curr_text)
        else:
            # Rare non-prefix case: place new content at end by length delta.
            delta = max(0, len(curr_text) - len(prev_text))
            start, end = len(prev_text), len(prev_text) + delta
            if end > len(curr_text):
                start, end = max(0, len(curr_text) - delta), len(curr_text)
        spans.append((start, end))
        prev_text = curr_text
    return spans


def continuation_char_span(
    neox_tokenizer: Any,
    token_ids: list[int],
    *,
    prompt_len: int = PROMPT_LEN,
) -> tuple[str, int, int]:
    """Return ``(full_text, cont_start, cont_end)`` for the continuation region."""
    if len(token_ids) < prompt_len:
        raise ValueError("token_ids shorter than prompt_len")
    spans = neox_token_char_spans(neox_tokenizer, token_ids)
    full_text = neox_tokenizer.decode(token_ids)
    cont_start = spans[prompt_len][0]
    cont_end = spans[-1][1]
    return full_text, cont_start, cont_end


def longest_eos_free_continuation(
    token_ids: list[int],
    *,
    prompt_len: int = PROMPT_LEN,
    eos_id: int = GPT_NEOX_EOS_ID,
) -> tuple[list[int], bool, int]:
    """Return ``(cont_tokens, crosses_doc_boundary, usable_len)``.

    If EOS appears in the continuation, keep the longest contiguous EOS-free
    sub-span of the continuation (may be empty).
    """
    cont = list(token_ids[prompt_len : prompt_len + CONT_LEN])
    if eos_id not in cont:
        return cont, False, len(cont)

    # Split on EOS; take longest run of non-EOS tokens.
    best: list[int] = []
    cur: list[int] = []
    for tid in cont:
        if tid == eos_id:
            if len(cur) > len(best):
                best = cur
            cur = []
        else:
            cur.append(tid)
    if len(cur) > len(best):
        best = cur
    return best, True, len(best)


def llama_query_ids_for_continuation(
    neox_tokenizer: Any,
    llama_tokenizer: Any,
    token_ids: list[int],
    *,
    prompt_len: int = PROMPT_LEN,
    edge_drop: int = LLAMA_EDGE_DROP,
    min_len: int = LLAMA_MIN_QUERY_LEN,
) -> dict[str, Any]:
    """Build token-aligned Llama ``query_ids`` for the NeoX continuation span.

    Decodes the full window, retokenizes with Llama-2, keeps Llama tokens whose
    character spans lie entirely inside the continuation span, then drops
    ``edge_drop`` tokens at each end.
    """
    cont_tokens, crosses, usable_len = longest_eos_free_continuation(
        token_ids, prompt_len=prompt_len
    )
    # Rebuild a synthetic window: original prompt + usable continuation (no EOS).
    prompt = list(token_ids[:prompt_len])
    synthetic = prompt + cont_tokens
    full_text, cont_start, cont_end = continuation_char_span(
        neox_tokenizer, synthetic, prompt_len=prompt_len
    )

    enc = llama_tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])
    kept = [
        tid
        for tid, (s, e) in zip(ids, offsets, strict=True)
        if s >= cont_start and e <= cont_end and e > s
    ]
    too_short_before_drop = len(kept) < (min_len + 2 * edge_drop)
    if len(kept) > 2 * edge_drop:
        kept = kept[edge_drop : len(kept) - edge_drop]
    else:
        kept = []
    too_short = len(kept) < min_len
    return {
        "query_ids": kept,
        "query_len_tokens": len(kept),
        "crosses_doc_boundary": crosses,
        "usable_cont_len": usable_len,
        "too_short": too_short or too_short_before_drop,
        "cont_char_span": [cont_start, cont_end],
        "full_text": full_text,
    }


def is_trivial_sequence(token_ids: list[int], neox_tokenizer: Any) -> dict[str, Any]:
    """Flag sequences with few distinct tokens or mostly whitespace."""
    cont = token_ids[PROMPT_LEN : PROMPT_LEN + CONT_LEN]
    distinct = len(set(cont))
    pieces = [neox_tokenizer.decode([t]) for t in cont]
    ws = sum(1 for p in pieces if p.strip() == "")
    frac_ws = ws / max(1, len(cont))
    trivial = distinct < TRIVIAL_DISTINCT_TOKENS or frac_ws > TRIVIAL_WHITESPACE_FRAC
    return {
        "trivial": trivial,
        "n_distinct_tokens": distinct,
        "whitespace_token_frac": frac_ws,
    }


def stratum_for_k(k: int) -> str:
    """Corrected strata: unmatched (0), low 1–5, mid 6–31, high ≥32."""
    if k <= 0:
        return "unmatched"
    if k <= 5:
        return "low"
    if k <= 31:
        return "mid"
    return "high"


def summarize_counts(ks: list[int]) -> dict[str, Any]:
    hist: dict[str, int] = {}
    strata = {"unmatched": 0, "low": 0, "mid": 0, "high": 0, "k_le_2": 0}
    for k in ks:
        hist[str(k)] = hist.get(str(k), 0) + 1
        strata[stratum_for_k(k)] += 1
        if 1 <= k <= 2:
            strata["k_le_2"] += 1
    return {"histogram_k": hist, "strata": strata, "n": len(ks)}
