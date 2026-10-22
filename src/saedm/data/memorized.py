"""EleutherAI Pythia memorization evaluation dataset helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, dataset_info as hf_dataset_info

DATASET_ID = "EleutherAI/pythia-memorized-evals"
# Final-checkpoint memorized set for Pythia-410M-deduped (step 143000).
SPLIT = "deduped.410m"
PARQUET_NAME = "data/deduped.410m-00000-of-00001-4bb233eb8d735ee8.parquet"
PROMPT_LEN = 32
CONT_LEN = 32
SEQ_LEN = PROMPT_LEN + CONT_LEN


def dataset_revision() -> str:
    return hf_dataset_info(DATASET_ID).sha or "unknown"


def load_memorized_table(
    cache_dir: str | Path | None = None,
) -> tuple[pa.Table, dict[str, Any]]:
    """Load the full memorized table for ``deduped.410m``.

    Every row is a memorized sequence (exact 32+32 greedy match); there is no
    score column — membership implies score == 1.
    """
    path = hf_hub_download(
        DATASET_ID,
        PARQUET_NAME,
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    table = pq.read_table(path)
    meta = {
        "dataset_id": DATASET_ID,
        "split": SPLIT,
        "parquet": PARQUET_NAME,
        "revision": dataset_revision(),
        "local_path": path,
        "n_rows": table.num_rows,
        "columns": table.column_names,
        "memorization_definition": (
            "32-token prompt, greedy 32-token continuation, exact match "
            "(Biderman et al. 2023); final checkpoint step143000"
        ),
    }
    return table, meta


def sample_rows(
    table: pa.Table,
    n: int,
    seed: int = 0,
) -> tuple[np.ndarray, pa.Table]:
    """Draw a fixed random sample of row indices (without replacement)."""
    n_rows = table.num_rows
    if n > n_rows:
        raise ValueError(f"Cannot sample {n} from {n_rows} rows")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_rows, size=n, replace=False))
    return indices, table.take(pa.array(indices))


def save_sample(
    out_dir: str | Path,
    indices: np.ndarray,
    sample: pa.Table,
) -> dict[str, str]:
    """Persist sample indices + tokens under ``out_dir`` (gitignored data/)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx_path = out / "sample_indices.npy"
    pq_path = out / "sample.parquet"
    np.save(idx_path, indices)
    pq.write_table(sample, pq_path)
    return {
        "sample_indices": str(idx_path),
        "sample_parquet": str(pq_path),
    }


def continuation_tokens(tokens: list[int]) -> list[int]:
    if len(tokens) < SEQ_LEN:
        raise ValueError(f"Expected >= {SEQ_LEN} tokens, got {len(tokens)}")
    return list(tokens[PROMPT_LEN:SEQ_LEN])
