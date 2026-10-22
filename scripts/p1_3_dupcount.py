#!/usr/bin/env python3
"""P1.3 — infini-gram duplication counts for the P1.1 sample."""

from __future__ import annotations

import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saedm.data.dupcount import (  # noqa: E402
    DEDUPE_INDEX,
    PRIMARY_INDEX,
    count_ngram,
    stratum_for_k,
    summarize_counts,
)
from saedm.data.memorized import continuation_tokens  # noqa: E402
from saedm.manifest import write_manifest  # noqa: E402
from saedm.paths import data_dir, results_dir  # noqa: E402
from saedm.seeding import set_seed  # noqa: E402

TASK_ID = "P1.3"
SEED = 0
TOKENIZER_NAME = "EleutherAI/pythia-410m-deduped"
TOKENIZER_REVISION = "step143000"
EXAMPLE_N = 200
WORKERS = 1
MIN_INTERVAL_S = 0.35


def main() -> None:
    started_at = datetime.now(UTC)
    set_seed(SEED)

    sample_path = data_dir() / "P1.1" / "sample.parquet"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing {sample_path}; run scripts/p1_1_memorized.py first")

    notes_dir = results_dir() / TASK_ID
    notes_dir.mkdir(parents=True, exist_ok=True)
    notes = {
        "primary_index": PRIMARY_INDEX,
        "primary_index_tokenizer": "Llama-2 (API tokenizes query text)",
        "deduped_pile_index": DEDUPE_INDEX,
        "deduped_note": (
            "No public deduped-Pile train index was listed on "
            "https://infini-gram.readthedocs.io/en/latest/api.html at task time. "
            "Only full-Pile count (v4_piletrain_llama) is recorded as k."
        ),
        "query": "decoded 32-token continuation text",
        "rate_limit": f"{WORKERS} workers, shared min_interval_s={MIN_INTERVAL_S}; disk cache",
    }
    (notes_dir / "notes.md").write_text(
        "# P1.3 notes\n\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in notes.items())
        + "\n"
    )

    table = pq.read_table(sample_path)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION)
    cache_dir = data_dir() / "P1.3" / "infini_gram_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    n = table.num_rows
    jobs = []
    for i in range(n):
        idx = int(table.column("index")[i].as_py())
        tokens = table.column("tokens")[i].as_py()
        cont = continuation_tokens(tokens)
        text = tokenizer.decode(cont)
        jobs.append((i, idx, text))

    rows_out: list[dict | None] = [None] * n
    done_lock = threading.Lock()
    done = [0]

    def _one(job: tuple[int, int, str]) -> None:
        i, idx, text = job
        result = count_ngram(
            text,
            index=PRIMARY_INDEX,
            cache_dir=cache_dir,
            min_interval_s=MIN_INTERVAL_S,
        )
        k = int(result.get("count", 0))
        rows_out[i] = {
            "sample_row": i,
            "pile_index": idx,
            "k_full_pile": k,
            "stratum": stratum_for_k(k),
            "continuation_text": text,
            "approx": bool(result.get("approx", False)),
        }
        with done_lock:
            done[0] += 1
            cur = done[0]
        if cur % 500 == 0 or cur == n:
            print(f"[{cur}/{n}] last_k={k}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(_one, job) for job in jobs]
        for fut in as_completed(futs):
            fut.result()

    assert all(r is not None for r in rows_out)
    rows_out_list = [r for r in rows_out if r is not None]
    ks = [int(r["k_full_pile"]) for r in rows_out_list]

    summary = summarize_counts(ks)
    summary.update(
        {
            "index": PRIMARY_INDEX,
            "deduped_index": DEDUPE_INDEX,
            "tokenizer_for_decode": {
                "name": TOKENIZER_NAME,
                "revision": TOKENIZER_REVISION,
            },
            "notes": notes,
        }
    )

    out_dir = results_dir() / TASK_ID
    summary_path = out_dir / "counts_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    # Persist full per-row counts under data/ (gitignored); examples CSV in results/.
    full_path = data_dir() / "P1.3" / "counts.jsonl"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with full_path.open("w") as f:
        for row in rows_out_list:
            f.write(json.dumps(row) + "\n")

    csv_path = out_dir / "examples_200.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_row",
                "pile_index",
                "k_full_pile",
                "stratum",
                "continuation_text",
            ],
        )
        writer.writeheader()
        for row in rows_out_list[:EXAMPLE_N]:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    ended_at = datetime.now(UTC)
    write_manifest(
        task_id=TASK_ID,
        config={
            "task_id": TASK_ID,
            "sample_path": str(sample_path.relative_to(REPO_ROOT)),
            "index": PRIMARY_INDEX,
            "seed": SEED,
        },
        seed=SEED,
        out_dir=out_dir,
        outputs=[
            str(summary_path.relative_to(REPO_ROOT)),
            str(csv_path.relative_to(REPO_ROOT)),
            str((out_dir / "notes.md").relative_to(REPO_ROOT)),
        ],
        started_at=started_at,
        ended_at=ended_at,
    )
    print(json.dumps(summary["strata"], indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
