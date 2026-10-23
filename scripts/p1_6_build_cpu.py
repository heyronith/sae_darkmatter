#!/usr/bin/env python3
"""P1.6 CPU — near_dup/mid/high pools, exposure-lift, C3 (no GPU)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saedm.data.controls import (  # noqa: E402
    build_c3_batch,
    canonical_sha256,
    sample_rows,
)
from saedm.data.dupcount import (  # noqa: E402
    load_precomputed_counts,
    stratum_recollection,
)
from saedm.data.steps import epoch1_index_cutoff  # noqa: E402
from saedm.data.templates import (  # noqa: E402
    HUFFMAN_TEMPLATE_THRESHOLD,
    SEMANTIC_MEM_COMMIT,
    classify_template,
)
from saedm.manifest import write_manifest  # noqa: E402
from saedm.paths import data_dir, results_dir  # noqa: E402
from saedm.seeding import set_seed  # noqa: E402

TASK_ID = "P1.6"
SEED = 0
N_PER = 1_000
NEAR_DUP_THRESH = 32
TOKENIZER_NAME = "EleutherAI/pythia-410m-deduped"
TOKENIZER_REVISION = "step143000"


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    h = canonical_sha256(rows)
    path.with_suffix(path.suffix + ".sha256").write_text(h + "\n")
    return h


def main() -> None:
    started_at = datetime.now(UTC)
    set_seed(SEED)
    out_dir = results_dir() / TASK_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = data_dir() / "P1.6"
    pool_dir.mkdir(parents=True, exist_ok=True)

    # --- Rebuild P1.4d-style pools from P1.1 sample + P1.3c counts ---
    sample_path = data_dir() / "P1.1" / "sample.parquet"
    counts_path = data_dir() / "P1.3c" / "counts.jsonl"
    if not sample_path.exists() or not counts_path.exists():
        raise FileNotFoundError("Need data/P1.1/sample.parquet and data/P1.3c/counts.jsonl")

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION)
    table = pq.read_table(sample_path)
    counts = [
        json.loads(line) for line in counts_path.read_text().splitlines() if line.strip()
    ]
    assert len(counts) == table.num_rows

    pools: dict[str, list[dict]] = {"near_dup": [], "mid": [], "high": []}
    n_trivial = 0
    n_template = 0
    for i, crow in enumerate(counts):
        if crow.get("trivial"):
            n_trivial += 1
            continue
        kp = int(crow["k_primary"])
        ki = int(crow["k_infgram"])
        tokens = table.column("tokens")[i].as_py()
        text = tok.decode(tokens[:64])
        flags = classify_template(text)
        if flags["is_template"]:
            n_template += 1
            continue
        refined = stratum_recollection(kp, ki, near_dup_threshold=NEAR_DUP_THRESH)
        if refined not in pools:
            continue
        pools[refined].append(
            {
                "sample_row": i,
                "pile_index": int(crow["pile_index"]),
                "k_primary": kp,
                "k_infgram": ki,
                "tokens": list(tokens[:64]),
                "stratum": refined,
            }
        )

    sampled = {}
    hashes = {}
    for name, rows in pools.items():
        take = sample_rows(rows, N_PER, seed=SEED)
        take = sorted(take, key=lambda r: r["pile_index"])
        h = _write_jsonl(pool_dir / f"{name}.jsonl", take)
        sampled[name] = take
        hashes[name] = {"n_pool": len(rows), "n_sampled": len(take), "sha256": h}
        print(f"{name}: pool={len(rows)} sampled={len(take)}")

    # --- Exposure lift: all k_primary==1 in epoch-1 vs memorized ---
    epoch = epoch1_index_cutoff()
    cutoff = int(epoch["index_cutoff"])
    print("Loading full deduped Counts for exposure lift...")
    counts_arr = load_precomputed_counts("deduped")
    n_all = min(len(counts_arr), cutoff)
    k_e1 = counts_arr[:n_all]
    n_k1_all = int((k_e1 == 1).sum())

    from saedm.data.memorized import load_memorized_table

    mem_table, mem_meta = load_memorized_table()
    mem_idx = np.asarray(mem_table.column("index").to_pylist(), dtype=np.int64)
    mem_k = counts_arr[mem_idx]
    mem_k1_e1 = int(((mem_k == 1) & (mem_idx < cutoff)).sum())
    rate = mem_k1_e1 / n_k1_all if n_k1_all else float("nan")
    p15c_rate = 2e-5
    exposure = {
        "epoch1": epoch,
        "n_indices_epoch1": n_all,
        "n_k_primary_1_epoch1": n_k1_all,
        "n_memorized_k_primary_1_epoch1": mem_k1_e1,
        "memorized_k1_over_all_k1": rate,
        "p15c_never_trained_exact_match_rate": p15c_rate,
        "lift_vs_p15c": (rate / p15c_rate) if p15c_rate else None,
        "memorized_meta": {
            k: mem_meta[k] for k in ("dataset_id", "split", "revision", "n_rows")
        },
    }
    print(
        f"Exposure: memorized_k1_e1={mem_k1_e1} / all_k1_e1={n_k1_all} "
        f"= {rate:.6g} (lift vs P1.5c {p15c_rate}: {rate / p15c_rate:.1f}x)"
    )

    # --- C3 ---
    c3 = build_c3_batch(N_PER, seed=SEED)
    c3_hash = _write_jsonl(pool_dir / "C3.jsonl", c3)

    cpu_summary = {
        "seed": SEED,
        "n_per_stratum": N_PER,
        "template_commit": SEMANTIC_MEM_COMMIT,
        "huffman_threshold": HUFFMAN_TEMPLATE_THRESHOLD,
        "near_dup_threshold": NEAR_DUP_THRESH,
        "n_trivial_excluded": n_trivial,
        "n_template_removed": n_template,
        "pools": hashes,
        "exposure_lift": exposure,
        "C3": {"n": len(c3), "sha256": c3_hash},
        "note": (
            "reconstruction / C1 / C2 / splits require GPU "
            "(modal_jobs/p1_6_strata_controls.py)"
        ),
    }
    (out_dir / "cpu_summary.json").write_text(json.dumps(cpu_summary, indent=2) + "\n")
    (out_dir / "notes_cpu.md").write_text(
        f"""# P1.6 CPU notes

## Strata (near_dup / mid / high)

Same filters as P1.4d on the P1.1 20k sample: drop trivial + template
(`{SEMANTIC_MEM_COMMIT}`, Huffman ≤ {HUFFMAN_TEMPLATE_THRESHOLD}), then
`stratum_recollection` with k_infgram threshold {NEAR_DUP_THRESH}. Sample
{N_PER}/stratum (seed {SEED}).

## Exposure lift

All deduped training Indices with `k_primary==1` and Index < epoch-1 cutoff
{cutoff} (= {epoch['derivation']}), versus memorized sequences with the same
filter. Compare the ratio to P1.5c never-trained exact-match rate {p15c_rate}.

## C3

High-entropy 32-token string, separator id 0, same string again; stored as
64-token prompt+continuation (= first copy | second copy).
"""
    )

    ended_at = datetime.now(UTC)
    write_manifest(
        task_id=TASK_ID,
        config={"task_id": TASK_ID, "phase": "cpu", "seed": SEED, "n_per": N_PER},
        seed=SEED,
        out_dir=out_dir,
        outputs=[
            "results/P1.6/cpu_summary.json",
            "results/P1.6/notes_cpu.md",
        ],
        started_at=started_at,
        ended_at=ended_at,
    )
    print(json.dumps({"pools": hashes, "exposure_lift": exposure}, indent=2))


if __name__ == "__main__":
    main()
