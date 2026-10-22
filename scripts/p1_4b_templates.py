#!/usr/bin/env python3
"""P1.4b — template filter + Gate G1 on corrected P1.3b strata."""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saedm.data.dupcount import stratum_for_k  # noqa: E402
from saedm.data.memorized import continuation_tokens  # noqa: E402
from saedm.data.templates import (  # noqa: E402
    HUFFMAN_TEMPLATE_THRESHOLD,
    SEMANTIC_MEM_COMMIT,
    SEMANTIC_MEM_REPO,
    classify_template,
)
from saedm.manifest import write_manifest  # noqa: E402
from saedm.paths import data_dir, results_dir  # noqa: E402
from saedm.seeding import set_seed  # noqa: E402

TASK_ID = "P1.4b"
SEED = 0
TOKENIZER_NAME = "EleutherAI/pythia-410m-deduped"
TOKENIZER_REVISION = "step143000"
N_EXAMPLES = 30
ANALYSIS_STRATA = ("low", "mid", "high")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((centre - margin) / denom, (centre + margin) / denom)


def main() -> None:
    started_at = datetime.now(UTC)
    set_seed(SEED)

    sample_path = data_dir() / "P1.1" / "sample.parquet"
    counts_path = data_dir() / "P1.3b" / "counts.jsonl"
    summary_p11 = json.loads((results_dir() / "P1.1" / "summary.json").read_text())
    total_memorized = int(summary_p11["total_memorized_count"])
    sample_size = int(summary_p11["sample_size"])

    if not sample_path.exists() or not counts_path.exists():
        raise FileNotFoundError("Need P1.1 sample and P1.3b counts.jsonl")

    out_dir = results_dir() / TASK_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notes.md").write_text(
        f"""# P1.4b notes

- Same template heuristics as P1.4 (unchanged thresholds / commit).
- Source: [{SEMANTIC_MEM_REPO}]({SEMANTIC_MEM_REPO}) commit `{SEMANTIC_MEM_COMMIT}`.
- Flag = incrementing OR repeating OR (Huffman average code length ≤ {HUFFMAN_TEMPLATE_THRESHOLD}).
- Strata from **P1.3b** `k_primary` (near-deduplicated Pile 32-gram exact counts).
- Unmatched (`k_primary==0`) and trivial sequences are excluded from stratum
  before/after tallies and from the G1 scaled estimate.
"""
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION)
    table = pq.read_table(sample_path)
    counts = [json.loads(line) for line in counts_path.read_text().splitlines() if line.strip()]
    assert len(counts) == table.num_rows

    per_stratum = {
        s: {
            "before": 0,
            "after": 0,
            "removed_incrementing": 0,
            "removed_repeating": 0,
            "removed_compressible": 0,
            "removed_any_template": 0,
        }
        for s in ANALYSIS_STRATA
    }
    survivors_low: list[dict] = []
    n_unmatched = 0
    n_trivial = 0
    n_excluded = 0

    for i, crow in enumerate(counts):
        kp = int(crow["k_primary"])
        stratum = crow.get("stratum") or stratum_for_k(kp)
        assert stratum == stratum_for_k(kp)

        if crow.get("trivial"):
            n_trivial += 1
            n_excluded += 1
            continue
        if stratum == "unmatched":
            n_unmatched += 1
            n_excluded += 1
            continue

        tokens = table.column("tokens")[i].as_py()
        full_text = tokenizer.decode(tokens[:64])
        cont_text = tokenizer.decode(continuation_tokens(tokens))
        flags = classify_template(full_text)
        bucket = per_stratum[stratum]
        bucket["before"] += 1
        if flags["is_incrementing"]:
            bucket["removed_incrementing"] += 1
        if flags["is_repeating"]:
            bucket["removed_repeating"] += 1
        if flags["compressible"]:
            bucket["removed_compressible"] += 1
        if flags["is_template"]:
            bucket["removed_any_template"] += 1
        else:
            bucket["after"] += 1
            if stratum == "low":
                survivors_low.append(
                    {
                        "sample_row": i,
                        "pile_index": crow["pile_index"],
                        "k": kp,
                        "full_text": full_text,
                        "continuation_text": cont_text,
                        **flags,
                    }
                )

    low_surv = per_stratum["low"]["after"]
    p_hat = low_surv / sample_size
    lo, hi = wilson_interval(low_surv, sample_size)
    est = p_hat * total_memorized
    est_lo, est_hi = lo * total_memorized, hi * total_memorized
    verdict = "upper_bound_below_200" if est_hi < 200 else "continue_to_P1.5"

    g1 = {
        "source_commit_heuristics": SEMANTIC_MEM_COMMIT,
        "huffman_threshold": HUFFMAN_TEMPLATE_THRESHOLD,
        "k_source": "P1.3b k_primary (usvsnsp/deduped-num-duplicates)",
        "total_memorized": total_memorized,
        "sample_size": sample_size,
        "n_unmatched": n_unmatched,
        "n_trivial": n_trivial,
        "n_excluded_from_strata": n_excluded,
        "per_stratum": per_stratum,
        "low_stratum_survivors_in_sample": low_surv,
        "scaled_low_survivor_estimate": {
            "point": est,
            "ci95_low": est_lo,
            "ci95_high": est_hi,
            "method": "wilson_proportion * total_memorized",
        },
        "verdict": verdict,
    }
    g1_path = out_dir / "g1_early.json"
    g1_path.write_text(json.dumps(g1, indent=2) + "\n")

    rng = random.Random(SEED)
    pick = (
        survivors_low
        if len(survivors_low) <= N_EXAMPLES
        else rng.sample(survivors_low, N_EXAMPLES)
    )
    lines = [
        "# Surviving low-stratum examples (after template filter)\n",
        f"Showing {len(pick)} of {len(survivors_low)} survivors.\n",
        "k = k_primary (near-deduplicated Pile 32-gram exact count).\n",
    ]
    for j, row in enumerate(pick, 1):
        lines.append(f"## {j}. pile_index={row['pile_index']} k={row['k']}\n")
        lines.append("```\n" + row["full_text"] + "\n```\n")
    examples_path = out_dir / "examples.md"
    examples_path.write_text("\n".join(lines))

    ended_at = datetime.now(UTC)
    write_manifest(
        task_id=TASK_ID,
        config={
            "task_id": TASK_ID,
            "huffman_threshold": HUFFMAN_TEMPLATE_THRESHOLD,
            "heuristics_commit": SEMANTIC_MEM_COMMIT,
            "seed": SEED,
            "counts_source": "P1.3b",
        },
        seed=SEED,
        out_dir=out_dir,
        outputs=[
            str(g1_path.relative_to(REPO_ROOT)),
            str(examples_path.relative_to(REPO_ROOT)),
            str((out_dir / "notes.md").relative_to(REPO_ROOT)),
        ],
        started_at=started_at,
        ended_at=ended_at,
    )
    print(json.dumps(g1["scaled_low_survivor_estimate"], indent=2))
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
