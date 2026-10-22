#!/usr/bin/env python3
"""P1.4d — split low stratum into recollection vs near-duplicate."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saedm.data.dupcount import stratum_for_k, stratum_recollection  # noqa: E402
from saedm.data.memorized import continuation_tokens  # noqa: E402
from saedm.data.templates import (  # noqa: E402
    HUFFMAN_TEMPLATE_THRESHOLD,
    SEMANTIC_MEM_COMMIT,
    classify_template,
)
from saedm.manifest import write_manifest  # noqa: E402
from saedm.paths import data_dir, results_dir  # noqa: E402
from saedm.seeding import set_seed  # noqa: E402
from saedm.stats import wilson_interval  # noqa: E402

TASK_ID = "P1.4d"
SEED = 0
TOKENIZER_NAME = "EleutherAI/pythia-410m-deduped"
TOKENIZER_REVISION = "step143000"
N_EXAMPLES = 30
DEFAULT_THRESH = 32
SENSITIVITY_THRESHOLDS = (8, 32, 128)


def _canonical_hash(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _scaled(count: int, sample_size: int, total: int) -> dict:
    lo, hi = wilson_interval(count, sample_size)
    return {
        "sample_count": count,
        "point": count / sample_size * total,
        "ci95_low": lo * total,
        "ci95_high": hi * total,
        "method": "wilson_proportion * total_memorized",
    }


def main() -> None:
    started_at = datetime.now(UTC)
    set_seed(SEED)

    sample_path = data_dir() / "P1.1" / "sample.parquet"
    counts_path = data_dir() / "P1.3c" / "counts.jsonl"
    summary_p11 = json.loads((results_dir() / "P1.1" / "summary.json").read_text())
    total_memorized = int(summary_p11["total_memorized_count"])
    sample_size = int(summary_p11["sample_size"])
    if not sample_path.exists() or not counts_path.exists():
        raise FileNotFoundError("Need P1.1 sample and P1.3c counts.jsonl")

    out_dir = results_dir() / TASK_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notes.md").write_text(
        f"""# P1.4d notes

Exact NeoX 32-gram `k_primary` under-calls near-duplicates (e.g. LaTeX
preambles with k_primary=2 but k_infgram≈5.8M). Split the former low stratum
**after** the unchanged template filter (`{SEMANTIC_MEM_COMMIT}`, Huffman ≤
{HUFFMAN_TEMPLATE_THRESHOLD}):

- **recollection**: k_primary ∈ [1,5] and k_infgram ≤ {DEFAULT_THRESH}
- **near_dup**: k_primary ∈ [1,5] and k_infgram > {DEFAULT_THRESH}
- mid / high unchanged (k_primary bands)

`k_infgram` reused from P1.3b/c cache (no new API queries).
"""
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION)
    table = pq.read_table(sample_path)
    counts = [json.loads(line) for line in counts_path.read_text().splitlines() if line.strip()]
    assert len(counts) == table.num_rows

    # After trivial exclusion + template filter, assign refined strata.
    strata_counts = {
        "recollection": 0,
        "near_dup": 0,
        "mid": 0,
        "high": 0,
        "unmatched": 0,
    }
    template_removed = 0
    trivial_n = 0
    recollection_rows: list[dict] = []
    near_dup_rows: list[dict] = []
    survivors_by_stratum: dict[str, list[dict]] = {
        "recollection": [],
        "near_dup": [],
        "mid": [],
        "high": [],
    }
    # Sensitivity: recollection counts at alternate thresholds (same survivors pool
    # of non-trivial, non-template, k_primary<=5).
    low_survivors: list[dict] = []

    for i, crow in enumerate(counts):
        if crow.get("trivial"):
            trivial_n += 1
            continue
        kp = int(crow["k_primary"])
        ki = int(crow["k_infgram"])
        tokens = table.column("tokens")[i].as_py()
        full_text = tokenizer.decode(tokens[:64])
        flags = classify_template(full_text)
        if flags["is_template"]:
            template_removed += 1
            continue

        refined = stratum_recollection(kp, ki, near_dup_threshold=DEFAULT_THRESH)
        strata_counts[refined] = strata_counts.get(refined, 0) + 1

        row = {
            "sample_row": i,
            "pile_index": int(crow["pile_index"]),
            "k_primary": kp,
            "k_infgram": ki,
            "tokens": list(tokens[:64]),
            "stratum": refined,
            "full_text": full_text,
            "continuation_text": tokenizer.decode(continuation_tokens(tokens)),
            **{k: flags[k] for k in (
                "is_incrementing",
                "is_repeating",
                "compressible",
                "huffman_code_length",
            )},
        }
        if refined in survivors_by_stratum:
            survivors_by_stratum[refined].append(row)
        if refined == "recollection":
            recollection_rows.append(row)
        elif refined == "near_dup":
            near_dup_rows.append(row)
        if stratum_for_k(kp) == "low":
            low_survivors.append(row)

    # Sensitivity table
    sensitivity = {}
    for thr in SENSITIVITY_THRESHOLDS:
        n_rec = sum(
            1
            for r in low_survivors
            if stratum_recollection(r["k_primary"], r["k_infgram"], near_dup_threshold=thr)
            == "recollection"
        )
        n_nd = len(low_survivors) - n_rec
        sensitivity[str(thr)] = {
            "recollection": n_rec,
            "near_dup": n_nd,
            "scaled_recollection": _scaled(n_rec, sample_size, total_memorized),
        }

    # k_primary histogram for recollection (1..5)
    hist = {str(k): 0 for k in range(1, 6)}
    for r in recollection_rows:
        hist[str(r["k_primary"])] += 1

    # Freeze lists (tokens + ks) for downstream GPU tasks
    def _freeze(rows: list[dict]) -> list[dict]:
        return [
            {
                "sample_row": r["sample_row"],
                "pile_index": r["pile_index"],
                "k_primary": r["k_primary"],
                "k_infgram": r["k_infgram"],
                "tokens": r["tokens"],
            }
            for r in sorted(rows, key=lambda x: x["sample_row"])
        ]

    frozen_rec = _freeze(recollection_rows)
    frozen_nd = _freeze(near_dup_rows)
    freeze_dir = data_dir() / "P1.4d"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    rec_path = freeze_dir / "recollection.jsonl"
    nd_path = freeze_dir / "near_dup.jsonl"
    with rec_path.open("w") as f:
        for r in frozen_rec:
            f.write(json.dumps(r) + "\n")
    with nd_path.open("w") as f:
        for r in frozen_nd:
            f.write(json.dumps(r) + "\n")
    rec_hash = _canonical_hash(frozen_rec)
    nd_hash = _canonical_hash(frozen_nd)
    (freeze_dir / "recollection.sha256").write_text(rec_hash + "\n")
    (freeze_dir / "near_dup.sha256").write_text(nd_hash + "\n")

    summary = {
        "template_commit": SEMANTIC_MEM_COMMIT,
        "huffman_threshold": HUFFMAN_TEMPLATE_THRESHOLD,
        "near_dup_threshold_default": DEFAULT_THRESH,
        "total_memorized": total_memorized,
        "sample_size": sample_size,
        "n_trivial_excluded": trivial_n,
        "n_template_removed": template_removed,
        "strata_after_template": strata_counts,
        "recollection_k_primary_histogram": hist,
        "scaled_estimates": {
            "recollection": _scaled(len(recollection_rows), sample_size, total_memorized),
            "near_dup": _scaled(len(near_dup_rows), sample_size, total_memorized),
            "mid": _scaled(strata_counts["mid"], sample_size, total_memorized),
            "high": _scaled(strata_counts["high"], sample_size, total_memorized),
        },
        "sensitivity_k_infgram_threshold": sensitivity,
        "frozen": {
            "recollection_n": len(frozen_rec),
            "recollection_sha256": rec_hash,
            "near_dup_n": len(frozen_nd),
            "near_dup_sha256": nd_hash,
            "paths": {
                "recollection": str(rec_path.relative_to(REPO_ROOT)),
                "near_dup": str(nd_path.relative_to(REPO_ROOT)),
            },
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    # Examples
    rng = random.Random(SEED)
    lines = ["# P1.4d examples\n"]
    for label, rows in (("recollection", recollection_rows), ("near_dup", near_dup_rows)):
        pick = rows if len(rows) <= N_EXAMPLES else rng.sample(rows, N_EXAMPLES)
        lines.append(f"\n## {label} ({len(pick)} of {len(rows)})\n")
        for j, row in enumerate(pick, 1):
            lines.append(
                f"### {j}. pile_index={row['pile_index']} "
                f"k_primary={row['k_primary']} k_infgram={row['k_infgram']}\n"
            )
            lines.append("```\n" + row["full_text"] + "\n```\n")
    examples_path = out_dir / "examples.md"
    examples_path.write_text("\n".join(lines))

    ended_at = datetime.now(UTC)
    write_manifest(
        task_id=TASK_ID,
        config={
            "task_id": TASK_ID,
            "seed": SEED,
            "near_dup_threshold": DEFAULT_THRESH,
            "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
            "template_commit": SEMANTIC_MEM_COMMIT,
            "recollection_sha256": rec_hash,
            "near_dup_sha256": nd_hash,
        },
        seed=SEED,
        out_dir=out_dir,
        outputs=[
            str(summary_path.relative_to(REPO_ROOT)),
            str(examples_path.relative_to(REPO_ROOT)),
            str((out_dir / "notes.md").relative_to(REPO_ROOT)),
        ],
        started_at=started_at,
        ended_at=ended_at,
    )
    print(json.dumps(strata_counts, indent=2))
    print("recollection scaled:", json.dumps(summary["scaled_estimates"]["recollection"], indent=2))
    print(f"frozen recollection sha256={rec_hash[:16]}… n={len(frozen_rec)}")


if __name__ == "__main__":
    main()
