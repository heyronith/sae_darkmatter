"""Tests for P1.6 matching and C3 construction."""

from __future__ import annotations

import numpy as np

from saedm.data.controls import (
    build_c3_batch,
    build_c3_sequence,
    canonical_sha256,
    coarse_pile_subset,
    freeze_split,
    ks_statistic,
    match_by_loss_and_subset,
)
from saedm.data.memorized import SEQ_LEN


def test_c3_second_copy_equals_first() -> None:
    rng = np.random.default_rng(0)
    row = build_c3_sequence(rng)
    assert len(row["tokens"]) == SEQ_LEN
    assert row["tokens"][:32] == row["tokens"][32:]
    assert row["tokens"][:32] == row["payload"]
    assert row["icl_prefix"][:32] == row["payload"]
    assert row["separator_ids"] == [0]


def test_c3_batch_deterministic() -> None:
    a = build_c3_batch(5, seed=0)
    b = build_c3_batch(5, seed=0)
    assert [r["tokens"] for r in a] == [r["tokens"] for r in b]
    c = build_c3_batch(5, seed=1)
    assert [r["tokens"] for r in a] != [r["tokens"] for r in c]


def test_coarse_pile_subset_buckets() -> None:
    assert coarse_pile_subset("def foo(x):\n    return x") == "code"
    assert coarse_pile_subset("We report a randomized trial. Abstract: DOI: 10.1") == "academic"
    assert coarse_pile_subset("See https://example.com for details") == "web"
    assert coarse_pile_subset("plain prose without markers") == "other"


def test_match_loss_and_subset_ks() -> None:
    rng = np.random.default_rng(0)
    memorized = []
    for i in range(20):
        sub = "code" if i < 10 else "other"
        base = 0.5 if sub == "code" else 1.5
        memorized.append(
            {
                "pile_subset": sub,
                "token_losses": (base + 0.05 * rng.normal(size=32)).tolist(),
                "tokens": list(range(64)),
            }
        )
    candidates = []
    for i in range(80):
        sub = "code" if i < 40 else "other"
        base = 0.5 if sub == "code" else 1.5
        candidates.append(
            {
                "pile_subset": sub,
                "token_losses": (base + 0.05 * rng.normal(size=32)).tolist(),
                "tokens": list(range(100, 164)),
                "pile_index": i,
            }
        )
    matched, quality = match_by_loss_and_subset(
        memorized, candidates, n=20, seed=0
    )
    assert len(matched) == 20
    assert quality["ks_token_loss"] < 0.25
    assert quality["subset_tv_distance"] < 0.05


def test_freeze_split_hashes() -> None:
    rows = [{"i": i, "tokens": [i]} for i in range(10)]
    a, b, ha, hb = freeze_split(rows, seed=0)
    assert len(a) + len(b) == 10
    assert ha == canonical_sha256(a)
    assert hb == canonical_sha256(b)
    assert ha != hb


def test_ks_identical() -> None:
    x = np.array([1.0, 2.0, 3.0])
    assert ks_statistic(x, x) == 0.0
