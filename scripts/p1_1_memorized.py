#!/usr/bin/env python3
"""P1.1 — load Pythia-410M-deduped memorized set and draw a 20k sample."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saedm.data.memorized import (  # noqa: E402
    DATASET_ID,
    SPLIT,
    load_memorized_table,
    sample_rows,
    save_sample,
)
from saedm.manifest import write_manifest  # noqa: E402
from saedm.paths import data_dir, results_dir  # noqa: E402
from saedm.seeding import set_seed  # noqa: E402

SAMPLE_SIZE = 20_000
SEED = 0
TASK_ID = "P1.1"


def _push_to_modal_volume(local_dir: Path) -> str:
    """Copy sample artifacts onto the saedm-artifacts Modal volume."""
    remote = f"/P1.1/{local_dir.name}"
    cmd = [
        "modal",
        "volume",
        "put",
        "saedm-artifacts",
        str(local_dir),
        remote,
        "--force",
    ]
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        return f"modal:saedm-artifacts:{remote}"
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"modal_upload_failed: {exc}"


def main() -> None:
    started_at = datetime.now(UTC)
    set_seed(SEED)

    table, meta = load_memorized_table()
    total = meta["n_rows"]
    print(f"Loaded {DATASET_ID} split={SPLIT} n={total} revision={meta['revision']}")

    indices, sample = sample_rows(table, SAMPLE_SIZE, seed=SEED)
    sample_dir = data_dir() / "P1.1"
    paths = save_sample(sample_dir, indices, sample)
    print(f"Saved sample to {sample_dir}")

    remote_path = _push_to_modal_volume(sample_dir)
    print(f"Remote copy: {remote_path}")

    out_dir = results_dir() / TASK_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_id": DATASET_ID,
        "split": SPLIT,
        "revision": meta["revision"],
        "total_memorized_count": total,
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "row_schema": {
            "index": "int — Pile training-sequence index (0-based)",
            "tokens": "list[int] — first 64 tokens (32 prompt + 32 continuation)",
            "__index_level_0__": "int — pandas export artifact (same as index)",
        },
        "memorization_definition": meta["memorization_definition"],
        "note": (
            "All rows in this split are memorized (score == 1 by construction); "
            "there is no separate score column."
        ),
        "sample_paths": {
            "local": {k: str(Path(v).relative_to(REPO_ROOT)) for k, v in paths.items()},
            "remote": remote_path,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    ended_at = datetime.now(UTC)
    write_manifest(
        task_id=TASK_ID,
        config={
            "task_id": TASK_ID,
            "dataset_id": DATASET_ID,
            "split": SPLIT,
            "sample_size": SAMPLE_SIZE,
            "seed": SEED,
        },
        seed=SEED,
        out_dir=out_dir,
        outputs=[
            str(summary_path.relative_to(REPO_ROOT)),
        ],
        started_at=started_at,
        ended_at=ended_at,
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
