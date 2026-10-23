"""Control-set construction helpers for P1.6 (C1 matching, C3 copying)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np
from scipy import stats

from saedm.data.memorized import CONT_LEN, PROMPT_LEN, SEQ_LEN

# NeoX EOD / EOS used as a short separator between copies.
DEFAULT_SEPARATOR_IDS = (0,)


def canonical_sha256(rows: list[dict]) -> str:
    """Stable hash of a list of dict rows (sorted keys, compact JSON)."""
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def freeze_split(
    rows: list[dict],
    *,
    seed: int = 0,
    frac_a: float = 0.5,
) -> tuple[list[dict], list[dict], str, str]:
    """Shuffle then 50/50 split; return (A, B, hash_A, hash_B)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    n_a = int(round(len(rows) * frac_a))
    a = [rows[int(i)] for i in order[:n_a]]
    b = [rows[int(i)] for i in order[n_a:]]
    return a, b, canonical_sha256(a), canonical_sha256(b)


def coarse_pile_subset(text: str) -> str:
    """Heuristic Pile-subset proxy when mmap windows lack domain labels.

    Buckets approximate The Pile's major families so C1 can be matched on
    subset *distribution* as well as loss. Not a gold domain label.
    """
    t = text or ""
    lower = t.lower()
    if re.search(
        r"\b(def |class |import |#include|function\s*\(|console\.log|=>)\b", t
    ) or re.search(r"[{};]\s*\n", t):
        return "code"
    if re.search(
        r"\b(abstract|doi:|arxiv|pubmed|et al\.|hypothesis|methods)\b", lower
    ):
        return "academic"
    if re.search(r"\b(wikipedia|references|external links)\b", lower):
        return "wiki"
    if re.search(r"<[^>]+>|&[a-z]+;|http://|https://", lower):
        return "web"
    return "other"


def build_c3_sequence(
    rng: np.random.Generator,
    *,
    vocab_size: int = 50_304,
    prompt_len: int = PROMPT_LEN,
    cont_len: int = CONT_LEN,
    separator_ids: tuple[int, ...] = DEFAULT_SEPARATOR_IDS,
    min_token_id: int = 1,
) -> dict[str, Any]:
    """In-context copy control: random string, separator, same string again.

    Spec: 32-token high-entropy string, a separator, then the same string;
    measured tokens = the second copy. Stored ``tokens`` (length 64) places the
    first copy in the prompt half and the second copy in the continuation half
    so the window matches memorized-evals (32+32). ``separator_ids`` and
    ``icl_prefix`` (= first copy + separator) are kept for generation probes
    that feed the separator-delimited form.
    """
    if prompt_len != cont_len:
        raise ValueError("C3 expects prompt_len == cont_len for copy symmetry")
    payload = rng.integers(min_token_id, vocab_size, size=prompt_len, dtype=np.int64)
    sep = list(separator_ids)
    first = payload.tolist()
    second = list(first)
    tokens = first + second
    assert len(tokens) == SEQ_LEN
    return {
        "tokens": tokens,
        "payload": first,
        "separator_ids": sep,
        "icl_prefix": first + sep,
        "pile_subset": "synthetic_random",
        "control": "C3",
    }


def build_c3_batch(n: int, seed: int = 0, **kwargs: Any) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = build_c3_sequence(rng, **kwargs)
        row["seq_i"] = i
        rows.append(row)
    return rows


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic on flattened arrays."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(stats.ks_2samp(a, b, method="auto").statistic)


def match_by_loss_and_subset(
    memorized: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    n: int,
    seed: int = 0,
    loss_key: str = "token_losses",
    subset_key: str = "pile_subset",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedy 1-NN match on mean loss within subset; report KS quality.

    Each memorized row needs ``loss_key`` (length-``CONT_LEN`` list) and
    ``subset_key``. Candidates likewise. Returns up to ``n`` matched controls
    (or fewer if the candidate pool is exhausted).
    """
    rng = np.random.default_rng(seed)
    mem_order = rng.permutation(len(memorized))
    # Index candidates by subset.
    by_sub: dict[str, list[int]] = {}
    for i, c in enumerate(candidates):
        by_sub.setdefault(str(c[subset_key]), []).append(i)
    used: set[int] = set()
    matched: list[dict[str, Any]] = []
    pair_diffs: list[float] = []

    for mi in mem_order:
        if len(matched) >= n:
            break
        m = memorized[int(mi)]
        sub = str(m[subset_key])
        m_loss = np.asarray(m[loss_key], dtype=np.float64)
        m_mean = float(m_loss.mean())
        pool = [j for j in by_sub.get(sub, []) if j not in used]
        if not pool:
            # Fall back to any unused candidate.
            pool = [j for j in range(len(candidates)) if j not in used]
        if not pool:
            break
        best_j = None
        best_d = float("inf")
        for j in pool:
            c_loss = np.asarray(candidates[j][loss_key], dtype=np.float64)
            d = abs(float(c_loss.mean()) - m_mean)
            if d < best_d:
                best_d = d
                best_j = j
        assert best_j is not None
        used.add(best_j)
        c = candidates[best_j]
        pair_diffs.append(best_d)
        matched.append(
            {
                **{k: c[k] for k in c if k != loss_key},
                "token_losses": list(c[loss_key]),
                "matched_memorized_i": int(mi),
                "mean_loss_diff": best_d,
                "pile_subset": c[subset_key],
                "control": "C1",
            }
        )

    if matched:
        mem_flat = np.concatenate(
            [
                np.asarray(memorized[int(i)][loss_key], dtype=np.float64)
                for i in mem_order[: len(matched)]
            ]
        )
        ctrl_flat = np.concatenate(
            [np.asarray(r["token_losses"], dtype=np.float64) for r in matched]
        )
    else:
        mem_flat = np.array([])
        ctrl_flat = np.array([])
    # Subset distribution TV distance.
    def _hist(rows: list[dict]) -> dict[str, float]:
        from collections import Counter

        c = Counter(str(r[subset_key]) for r in rows)
        tot = sum(c.values()) or 1
        return {k: v / tot for k, v in c.items()}

    mem_h = _hist([memorized[int(i)] for i in mem_order[: len(matched)]])
    ctrl_h = _hist(matched)
    keys = sorted(set(mem_h) | set(ctrl_h))
    tv = 0.5 * sum(abs(mem_h.get(k, 0.0) - ctrl_h.get(k, 0.0)) for k in keys)

    quality = {
        "n_matched": len(matched),
        "ks_token_loss": ks_statistic(mem_flat, ctrl_flat),
        "mean_abs_mean_loss_diff": float(np.mean(pair_diffs)) if pair_diffs else None,
        "subset_tv_distance": tv,
        "memorized_subset_hist": mem_h,
        "control_subset_hist": ctrl_h,
    }
    return matched, quality


def sample_rows(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Sample up to ``n`` rows without replacement (seeded)."""
    if len(rows) <= n:
        return list(rows)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=n, replace=False)
    return [rows[int(i)] for i in sorted(idx.tolist())]
