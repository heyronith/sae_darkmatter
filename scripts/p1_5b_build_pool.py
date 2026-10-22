#!/usr/bin/env python3
"""P1.5b CPU — build powered-mapping candidate pool from full memorized split."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saedm.data.dupcount import (  # noqa: E402
    DEDUPED_COUNTS_DS,
    is_trivial_sequence,
    load_precomputed_counts,
    precomputed_dataset_meta,
)
from saedm.data.memorized import load_memorized_table  # noqa: E402
from saedm.data.steps import (  # noqa: E402
    MIN_STEP_POWERED,
    epoch1_index_cutoff,
    powered_brackets,
    predicted_step,
)
from saedm.data.templates import (  # noqa: E402
    HUFFMAN_TEMPLATE_THRESHOLD,
    SEMANTIC_MEM_COMMIT,
    classify_template,
)
from saedm.manifest import write_manifest  # noqa: E402
from saedm.paths import data_dir, results_dir  # noqa: E402
from saedm.seeding import set_seed  # noqa: E402

TASK_ID = "P1.5b"
SEED = 0
NEOX_NAME = "EleutherAI/pythia-410m-deduped"
NEOX_REVISION = "step143000"
MAX_SAMPLE = 20_000


def main() -> None:
    started_at = datetime.now(UTC)
    set_seed(SEED)
    out_dir = results_dir() / TASK_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = data_dir() / "P1.5b"
    pool_dir.mkdir(parents=True, exist_ok=True)

    epoch_meta = epoch1_index_cutoff()
    cutoff = int(epoch_meta["index_cutoff"])
    print("Epoch-1 cutoff:", epoch_meta["derivation"])

    print("Loading memorized table + deduped Counts...")
    table, mem_meta = load_memorized_table()
    counts = load_precomputed_counts("deduped")
    counts_meta = precomputed_dataset_meta()["deduped"]
    assert counts_meta["dataset_id"] == DEDUPED_COUNTS_DS

    indices = np.asarray(table.column("index").to_pylist(), dtype=np.int64)
    tokens_col = table.column("tokens")
    n_full = len(indices)
    print(f"Full memorized split: {n_full}")

    k_primary = counts[indices]
    mask_k1 = k_primary == 1
    n_k1 = int(mask_k1.sum())
    print(f"k_primary==1: {n_k1}")

    mask_e1 = mask_k1 & (indices < cutoff)
    n_e1 = int(mask_e1.sum())
    print(f"epoch-1 (index < {cutoff}): {n_e1}")

    steps = indices // 1024
    mask_late = mask_e1 & (steps >= MIN_STEP_POWERED)
    n_late = int(mask_late.sum())
    print(f"s >= {MIN_STEP_POWERED}: {n_late}")

    tok = AutoTokenizer.from_pretrained(NEOX_NAME, revision=NEOX_REVISION)
    cand_idx = np.flatnonzero(mask_late)
    kept_rows: list[dict] = []
    n_trivial = 0
    n_template = 0
    for j, row_i in enumerate(cand_idx):
        tokens = tokens_col[int(row_i)].as_py()
        tr = is_trivial_sequence(tokens, tok)
        if tr["trivial"]:
            n_trivial += 1
            continue
        text = tok.decode(tokens[:64])
        flags = classify_template(text)
        if flags["is_template"]:
            n_template += 1
            continue
        pile_index = int(indices[int(row_i)])
        step = int(predicted_step(pile_index))
        br = powered_brackets(step)
        kept_rows.append(
            {
                "memorized_row": int(row_i),
                "pile_index": pile_index,
                "k_primary": 1,
                "predicted_step": step,
                "tokens": list(tokens[:64]),
                **{k: br[k] for k in ("c_sel", "c_pl", "c_before", "c_after")},
            }
        )
        if (j + 1) % 5000 == 0:
            print(f"  filtered {j + 1}/{len(cand_idx)}; kept {len(kept_rows)}", flush=True)

    n_filtered = len(kept_rows)
    print(f"after trivial+template: {n_filtered} (trivial={n_trivial}, template={n_template})")

    # Spread sample over steps.
    kept_rows.sort(key=lambda r: r["predicted_step"])
    if n_filtered <= MAX_SAMPLE:
        sample = kept_rows
    else:
        idxs = np.linspace(0, n_filtered - 1, num=MAX_SAMPLE, dtype=int)
        seen: set[int] = set()
        sample = []
        for i in idxs:
            ii = int(i)
            if ii not in seen:
                sample.append(kept_rows[ii])
                seen.add(ii)
        rng = np.random.default_rng(SEED)
        while len(sample) < MAX_SAMPLE:
            j = int(rng.integers(0, n_filtered))
            if j not in seen:
                sample.append(kept_rows[j])
                seen.add(j)
    sample.sort(key=lambda r: r["predicted_step"])
    for i, row in enumerate(sample):
        row["sample_i"] = i

    pool_path = pool_dir / "candidates.jsonl"
    with pool_path.open("w") as f:
        for row in sample:
            f.write(json.dumps(row) + "\n")

    pool_summary = {
        "seed": SEED,
        "epoch1": epoch_meta,
        "k_primary_dataset": counts_meta,
        "memorized_meta": {
            k: mem_meta[k] for k in ("dataset_id", "split", "revision", "n_rows")
        },
        "filter_counts": {
            "n_full_memorized": n_full,
            "n_k_primary_1": n_k1,
            "n_epoch1": n_e1,
            "n_step_ge_10000": n_late,
            "n_trivial_removed": n_trivial,
            "n_template_removed": n_template,
            "n_candidates_after_filters": n_filtered,
            "n_sampled": len(sample),
            "max_sample": MAX_SAMPLE,
        },
        "template_commit": SEMANTIC_MEM_COMMIT,
        "huffman_threshold": HUFFMAN_TEMPLATE_THRESHOLD,
        "min_step": MIN_STEP_POWERED,
        "step_range_sampled": {
            "min": sample[0]["predicted_step"] if sample else None,
            "max": sample[-1]["predicted_step"] if sample else None,
        },
        "candidates_path": str(pool_path.relative_to(REPO_ROOT)),
    }
    # Write into results as pool_summary (GPU job will merge into summary.json).
    (out_dir / "pool_summary.json").write_text(json.dumps(pool_summary, indent=2) + "\n")
    (out_dir / "notes_pool.md").write_text(
        f"""# P1.5b pool notes (CPU)

## Epoch-1 cutoff

{epoch_meta["derivation"]}

Pythia (Biderman et al.): near-deduplicated Pile ≈ 207B tokens; training mmap
uses packed length-2049 windows. One epoch ≈ {epoch_meta["n_sequences_one_epoch"]}
sequences. We keep Indices `< 0.95 × that` (= {cutoff}) to stay clear of the
ill-defined second-epoch packing region (pythia#144).

## Filters

k_primary==1 from `{DEDUPED_COUNTS_DS}` @ `{counts_meta["revision"][:12]}…`
(same revision as P1.3c). Then s≥{MIN_STEP_POWERED}, trivial + template
(`{SEMANTIC_MEM_COMMIT}`). Sampled {len(sample)}/{n_filtered} spread over steps.
"""
    )

    ended_at = datetime.now(UTC)
    write_manifest(
        task_id=TASK_ID,
        config={
            "task_id": TASK_ID,
            "stage": "cpu_pool",
            "seed": SEED,
            "max_sample": MAX_SAMPLE,
            "epoch1_cutoff": cutoff,
        },
        seed=SEED,
        out_dir=out_dir,
        outputs=[
            str((out_dir / "pool_summary.json").relative_to(REPO_ROOT)),
            str((out_dir / "notes_pool.md").relative_to(REPO_ROOT)),
        ],
        started_at=started_at,
        ended_at=ended_at,
    )
    print(json.dumps(pool_summary["filter_counts"], indent=2))


if __name__ == "__main__":
    main()
