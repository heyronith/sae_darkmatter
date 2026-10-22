#!/usr/bin/env python3
"""P1.3c — resolve which window k_primary counts; drop mis-joined duped column."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saedm.data.dupcount import (  # noqa: E402
    DEDUPED_COUNTS_DS,
    PRIMARY_INDEX,
    count_ngram_ids,
    is_trivial_sequence,
    llama_query_ids_for_continuation,
    llama_query_ids_for_token_span,
    load_precomputed_counts,
    precomputed_dataset_meta,
    stratum_for_k,
    summarize_counts,
)
from saedm.manifest import write_manifest  # noqa: E402
from saedm.paths import data_dir, results_dir  # noqa: E402
from saedm.seeding import set_seed  # noqa: E402

TASK_ID = "P1.3c"
SEED = 0
NEOX_NAME = "EleutherAI/pythia-410m-deduped"
NEOX_REVISION = "step143000"
LLAMA_NAME = "NousResearch/Llama-2-7b-hf"
WINDOW_TEST_N = 300
MIN_INTERVAL_S = 1.0
SEMANTIC_MEM_COMMIT = "84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2"

WINDOWS = {
    "prompt": (0, 32),
    "continuation": (32, 64),
    "full64": (0, 64),
}


def _notes_md(
    *,
    meta: dict,
    window_test: dict,
    decision: str,
    epoch_notes: str,
) -> str:
    w = window_test["windows"]
    table_rows = "\n".join(
        f"| {name} | {span} | {w[name]['violation_rate']:.4f} | "
        f"{w[name]['spearman']:.4f} |"
        for name, span in (
            ("prompt", "[0, 32)"),
            ("continuation", "[32, 64)"),
            ("full64", "[0, 64)"),
        )
    )
    return f"""# P1.3c notes

## Step 1 — Definitions from EleutherAI/semantic-memorization @{SEMANTIC_MEM_COMMIT}

### `usvsnsp/deduped-num-duplicates` and `usvsnsp/duped-num-duplicates`

Pipeline (README.md:48–54):
1. `working_dirs/orz/sequence_duplication/save_sequence_hashes.py` — hash every
   32-gram in each 2049-token training sequence (`for i in range(2049 - ngrams)`,
   lines 101–103); `Index` = sequence id in that corpus’s preshuffled mmap;
   `Offset` = start token index of the 32-gram.
2. `save_zero_offset_duplicate_hashes.py` — keep hashes with
   `Offset == offset_start_idx` (default **32**, lines 109–112): comment says
   “start token idx would be 33rd token, zero indexed” i.e. tokens **S[32:64]**.
3. `save_true_duplicate_counts.cpp` lines 19–21:
   `NGRAMS = 32; OFFSET_START_IDX = 32;`
   `// We only calculate number of duplicates of continuation, S[32:64]`
   Exact compare loads `pos = (Index, OFFSET_START_IDX)` (line 72).
4. `save_true_duplicates.py` lines 19–22 — emit parquet with columns
   `Index` ∈ `[0, 1024*143000)` and `Counts` (default 1 if missing).

Corpus (cpp header lines 8–14):
- **deduped** table: `EleutherAI/pile-deduped-pythia-preshuffled`
- **duped/standard** table: `EleutherAI/pile-standard-pythia-preshuffled`
- `Index` is **corpus-specific** (same integer range, **different** sequence
  orderings). Joining the duped table on a deduped Index is incorrect.

### `usvsnsp/semantic-filters` → `sequence_duplicates`

- `filters/highly_duplicated_filter.py` lines 9–28: left-join precomputed
  `SEQUENCE_FREQUENCIES` on `sequence_id`, alias `frequency` →
  `sequence_duplicates`. Docstring (lines 18–19) says “64-gram” but the loaded
  feature is **not** from `*-num-duplicates` directly.
- `calculate_metrics.py` line 218: loads
  `usvsnsp/generation-num-duplicates` split=`scheme` (`deduped`/`duped`),
  maps `counts` → `frequency`. That HF dataset is the generation/continuation
  frequency table used as `sequence_duplicates` in the Spark pipeline.

**Conclusion from source:** `k_primary` (= deduped `Counts[Index]`) is the
exact NeoX **32-gram frequency of the continuation S[32:64]** in the
**near-deduplicated** Pile training mmap.

## Step 2 — Empirical window test

Same token-aligned Llama `query_ids` method as P1.3b (edge_drop=2, min≥16),
index `{PRIMARY_INDEX}`. Subsample n={window_test.get('n')}.

| window | NeoX span | violation rate (k_infgram < k_primary) | Spearman vs k_primary |
|---|---|---:|---:|
{table_rows}

**Empirical conclusion:** `{window_test.get('best_window')}`
(lowest violation rate; Spearman as tie-break).

Source + empirical agreement: **{decision}**

Note: even for the true window, `k_infgram < k_primary` can still occur because
infini-gram uses a **different tokenizer** (Llama-2) and a shortened interior
span on the **full** Pile — so violation rate need not be exactly 0.

## Step 3 — Final k

- **Primary:** `k_primary` from `{DEDUPED_COUNTS_DS}` (revision
  `{meta['deduped']['revision']}`), continuation 32-gram, deduped Pile Index.
- **Removed:** `k_full_pile_precomputed`. Duped and deduped mmaps have
  independent Index orderings; no public Index mapping exists, so the join used
  in P1.3b was invalid (and produced many rows with
  `k_full_pile_precomputed < k_primary`).
- **Full-Pile proxy:** token-aligned `k_infgram` on `{PRIMARY_INDEX}` only.

## Step 4 — Epoch exposure

{epoch_notes}

## Dataset meta

```json
{json.dumps(meta['deduped'], indent=2)}
```
"""


def _epoch_notes() -> str:
    return """\
Pythia README: deduped models train for ≈300B tokens on a 207B-token
near-deduplicated Pile ⇒ **≈1.5 epochs**
(https://github.com/EleutherAI/pythia). Each training step uses batch of 1024
sequences of length 2049 from the **preshuffled** mmap
(`EleutherAI/pile-deduped-pythia-preshuffled`); `dataset[(step*1024)+j]` is the
window at that step (EleutherAI/pythia#123).

Whether the second half-epoch simply re-reads Indices 0… from the start in the
same packing is **not verified**. EleutherAI/pythia#144 (StellaAthena,
2025-05-23): document concatenation happens *before* packing, so tokens from the
“extra” ~0.5 epoch can appear at **different positions** within 2049-token
windows — Index-based `seen_epochs_of_self ∈ {{1,2}}` is therefore unreliable.

We record `exposure_lower_bound = k_primary` (exact continuation duplicates in
the deduped mmap) and do **not** multiply by a guessed epoch factor. Duplicates’
own epoch / packing positions are unknown."""


def _infgram_span(
    neox_tok,
    llama_tok,
    tokens: list[int],
    tok_start: int,
    tok_end: int,
    cache_dir: Path,
    *,
    use_continuation_eos_trim: bool,
) -> dict:
    if use_continuation_eos_trim:
        align = llama_query_ids_for_continuation(neox_tok, llama_tok, tokens)
    else:
        align = llama_query_ids_for_token_span(
            neox_tok, llama_tok, tokens, tok_start=tok_start, tok_end=tok_end
        )
    out = {
        "query_len_tokens": align["query_len_tokens"],
        "too_short": align["too_short"],
    }
    if align["too_short"] or not align["query_ids"]:
        out["k_infgram"] = 0
        out["infgram_skipped"] = True
        return out
    result = count_ngram_ids(
        align["query_ids"],
        index=PRIMARY_INDEX,
        cache_dir=cache_dir,
        min_interval_s=MIN_INTERVAL_S,
    )
    out["k_infgram"] = int(result.get("count", 0))
    out["infgram_skipped"] = False
    return out


def run_window_test(
    sample_table,
    k_primary: np.ndarray,
    neox_tok,
    llama_tok,
    cache_dir: Path,
    rng: np.random.Generator,
) -> dict:
    n = sample_table.num_rows
    idx = np.sort(rng.choice(n, size=min(WINDOW_TEST_N, n), replace=False))
    per_window: dict[str, list[dict]] = {w: [] for w in WINDOWS}

    for i in idx:
        tokens = sample_table.column("tokens")[int(i)].as_py()[:64]
        kp = int(k_primary[int(i)])
        for name, (ts, te) in WINDOWS.items():
            ig = _infgram_span(
                neox_tok,
                llama_tok,
                tokens,
                ts,
                te,
                cache_dir,
                use_continuation_eos_trim=(name == "continuation"),
            )
            per_window[name].append(
                {
                    "sample_row": int(i),
                    "k_primary": kp,
                    "k_infgram": ig["k_infgram"],
                    "skipped": bool(ig.get("infgram_skipped")),
                    "query_len_tokens": ig["query_len_tokens"],
                }
            )
        if (len(per_window["prompt"]) % 25) == 0:
            print(
                f"window test {len(per_window['prompt'])}/{len(idx)}",
                flush=True,
            )

    summary_windows = {}
    for name, rows in per_window.items():
        usable = [r for r in rows if not r["skipped"]]
        if not usable:
            summary_windows[name] = {
                "n_usable": 0,
                "violation_rate": float("nan"),
                "spearman": float("nan"),
            }
            continue
        kp = np.array([r["k_primary"] for r in usable], dtype=np.float64)
        ki = np.array([r["k_infgram"] for r in usable], dtype=np.float64)
        viol = float(np.mean(ki < kp))
        spear = float(stats.spearmanr(kp, ki).statistic) if len(usable) >= 10 else float("nan")
        summary_windows[name] = {
            "n_usable": len(usable),
            "n_skipped": len(rows) - len(usable),
            "violation_rate": viol,
            "spearman": spear,
            "mean_k_infgram": float(np.mean(ki)),
            "median_k_infgram": float(np.median(ki)),
        }
        print(
            f"  {name}: viol={viol:.4f} spearman={spear:.4f} usable={len(usable)}",
            flush=True,
        )

    # Best = lowest violation; tie-break highest Spearman.
    def key(item: tuple[str, dict]) -> tuple:
        w = item[1]
        return (w["violation_rate"], -w["spearman"])

    best = min(summary_windows.items(), key=key)[0]
    return {
        "n": len(idx),
        "seed": SEED,
        "windows": summary_windows,
        "best_window": best,
        "rows_path": "window_test_rows.jsonl",
        "_rows": per_window,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-window-test",
        action="store_true",
        help="Reuse results/P1.3c/window_test.json if present",
    )
    args = parser.parse_args()

    started_at = datetime.now(UTC)
    set_seed(SEED)
    rng = np.random.default_rng(SEED)

    sample_path = data_dir() / "P1.1" / "sample.parquet"
    counts_b_path = data_dir() / "P1.3b" / "counts.jsonl"
    if not sample_path.exists() or not counts_b_path.exists():
        raise FileNotFoundError("Need P1.1 sample and P1.3b counts.jsonl")

    out_dir = results_dir() / TASK_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = precomputed_dataset_meta()

    sample = pq.read_table(sample_path)
    n = sample.num_rows
    pile_idx = np.array(sample.column("index").to_pylist(), dtype=np.int64)
    print("Loading k_primary (deduped)...")
    k_primary = load_precomputed_counts("deduped")[pile_idx]

    neox_tok = AutoTokenizer.from_pretrained(NEOX_NAME, revision=NEOX_REVISION)
    llama_tok = AutoTokenizer.from_pretrained(LLAMA_NAME, use_fast=True)
    cache_dir = data_dir() / "P1.3b" / "infini_gram_ids_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    window_path = out_dir / "window_test.json"
    if args.skip_window_test and window_path.exists():
        window_test = json.loads(window_path.read_text())
        print("Reusing window_test.json; best=", window_test.get("best_window"))
    else:
        print(f"Running window test on {WINDOW_TEST_N} sequences...")
        window_test = run_window_test(
            sample, k_primary, neox_tok, llama_tok, cache_dir, rng
        )
        rows = window_test.pop("_rows")
        rows_path = data_dir() / "P1.3c" / "window_test_rows.jsonl"
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        with rows_path.open("w") as f:
            for name, lst in rows.items():
                for r in lst:
                    f.write(json.dumps({"window": name, **r}) + "\n")
        window_path.write_text(json.dumps(window_test, indent=2) + "\n")

    best = window_test["best_window"]
    if best not in {"continuation", "full64"}:
        decision = (
            f"STOP: empirical best window is `{best}`, not continuation/full64. "
            "Not re-stratifying; primary definition unchanged pending human review."
        )
        (out_dir / "notes.md").write_text(
            _notes_md(
                meta=meta,
                window_test=window_test,
                decision=decision,
                epoch_notes=_epoch_notes(),
            )
        )
        write_manifest(
            task_id=TASK_ID,
            config={"task_id": TASK_ID, "stage": "window_test_stop", "seed": SEED},
            seed=SEED,
            out_dir=out_dir,
            outputs=[
                str(window_path.relative_to(REPO_ROOT)),
                str((out_dir / "notes.md").relative_to(REPO_ROOT)),
            ],
            started_at=started_at,
            ended_at=datetime.now(UTC),
        )
        print(decision)
        sys.exit(2)

    decision = (
        f"KEEP k_primary as primary (source: continuation S[32:64]; "
        f"empirical best={best})."
    )
    print(decision)

    # Rebuild counts from P1.3b without k_full_pile_precomputed.
    print("Writing P1.3c counts (drop mis-joined duped column)...")
    rows_out: list[dict] = []
    with counts_b_path.open() as f:
        for line in f:
            crow = json.loads(line)
            kp = int(crow["k_primary"])
            assert kp == int(k_primary[int(crow["sample_row"])])
            row = {
                "sample_row": crow["sample_row"],
                "pile_index": crow["pile_index"],
                "k_primary": kp,
                "exposure_lower_bound": kp,
                "stratum": stratum_for_k(kp),
                "k_infgram": crow.get("k_infgram"),
                "query_len_tokens": crow.get("query_len_tokens"),
                "crosses_doc_boundary": crow.get("crosses_doc_boundary"),
                "too_short": crow.get("too_short"),
                "trivial": crow.get("trivial"),
                "n_distinct_tokens": crow.get("n_distinct_tokens"),
                "whitespace_token_frac": crow.get("whitespace_token_frac"),
                "method": crow.get("method", "token_aligned_query_ids"),
                "epoch_note": (
                    "seen_epochs_of_self not assigned; Index packing across the "
                    "~1.5th epoch is unverified (pythia#144)"
                ),
            }
            # Prefer recomputing trivial if missing
            if row["trivial"] is None:
                tokens = sample.column("tokens")[int(row["sample_row"])].as_py()
                row.update(is_trivial_sequence(tokens, neox_tok))
            rows_out.append(row)

    assert len(rows_out) == n
    full_path = data_dir() / "P1.3c" / "counts.jsonl"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with full_path.open("w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")

    non_trivial = [r for r in rows_out if not r["trivial"]]
    summary = summarize_counts([r["k_primary"] for r in non_trivial])
    trivial_by_stratum = {"unmatched": 0, "low": 0, "mid": 0, "high": 0}
    for r in rows_out:
        if r["trivial"]:
            trivial_by_stratum[stratum_for_k(r["k_primary"])] += 1
    summary.update(
        {
            "k_primary_source": meta["deduped"],
            "k_primary_window": "continuation S[32:64] (NeoX 32-gram exact)",
            "k_full_pile_precomputed": "REMOVED — cannot join duped Index onto deduped",
            "infgram_index": PRIMARY_INDEX,
            "n_total": n,
            "n_trivial": sum(1 for r in rows_out if r["trivial"]),
            "trivial_by_stratum": trivial_by_stratum,
            "n_crosses_doc_boundary": sum(
                1 for r in rows_out if r.get("crosses_doc_boundary")
            ),
            "n_unmatched_nontrivial": sum(
                1 for r in non_trivial if r["stratum"] == "unmatched"
            ),
            "window_test": {
                k: window_test[k]
                for k in ("n", "best_window", "windows")
            },
            "decision": decision,
            "epoch": {
                "deduped_approx_epochs": 1.5,
                "seen_epochs_assigned": False,
                "exposure_lower_bound": "k_primary",
                "note": "see notes.md Step 4",
            },
        }
    )
    summary_path = out_dir / "counts_summary.json"
    # Drop huge hist from being the only payload — keep it but OK
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = out_dir / "examples_200.csv"
    with csv_path.open("w", newline="") as f:
        fields = [
            "sample_row",
            "pile_index",
            "k_primary",
            "exposure_lower_bound",
            "k_infgram",
            "stratum",
            "trivial",
            "crosses_doc_boundary",
            "query_len_tokens",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows_out[:200]:
            w.writerow({k: row[k] for k in fields})

    (out_dir / "notes.md").write_text(
        _notes_md(
            meta=meta,
            window_test=window_test,
            decision=decision,
            epoch_notes=_epoch_notes(),
        )
    )

    ended_at = datetime.now(UTC)
    write_manifest(
        task_id=TASK_ID,
        config={
            "task_id": TASK_ID,
            "seed": SEED,
            "k_primary_dataset": DEDUPED_COUNTS_DS,
            "k_primary_window": "continuation_S[32:64]",
            "infgram_index": PRIMARY_INDEX,
            "dropped": ["k_full_pile_precomputed"],
        },
        seed=SEED,
        out_dir=out_dir,
        outputs=[
            str(summary_path.relative_to(REPO_ROOT)),
            str(csv_path.relative_to(REPO_ROOT)),
            str((out_dir / "notes.md").relative_to(REPO_ROOT)),
            str(window_path.relative_to(REPO_ROOT)),
        ],
        started_at=started_at,
        ended_at=ended_at,
    )
    print(json.dumps(summary["strata"], indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
