"""Small shared statistics helpers."""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((centre - margin) / denom, (centre + margin) / denom)


def bootstrap_median_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    paired: bool = False,
) -> dict[str, float]:
    """Bootstrap CI for median(a) - median(b).

    Here ``a`` is typically the predicted-interval drop and ``b`` the control
    drop, so a positive difference supports the mapping. If ``paired``, resample
    shared indices (requires ``len(a) == len(b)``).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return {
            "median_a": float("nan"),
            "median_b": float("nan"),
            "diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    if paired and len(a) != len(b):
        raise ValueError("paired bootstrap requires equal-length arrays")
    rng = np.random.default_rng(seed)
    med_a = float(np.median(a))
    med_b = float(np.median(b))
    diffs = np.empty(n_boot, dtype=np.float64)
    n = len(a)
    for i in range(n_boot):
        if paired:
            idx = rng.integers(0, n, size=n)
            diffs[i] = float(np.median(a[idx]) - np.median(b[idx]))
        else:
            sa = a[rng.integers(0, len(a), size=len(a))]
            sb = b[rng.integers(0, len(b), size=len(b))]
            diffs[i] = float(np.median(sa) - np.median(sb))
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return {
        "median_a": med_a,
        "median_b": med_b,
        "diff": med_a - med_b,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": float(n_boot),
        "paired": float(paired),
    }
