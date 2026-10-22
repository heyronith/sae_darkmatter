#!/usr/bin/env python3
"""P1.3b — corrected duplication counts (precomputed primary + token-aligned infini-gram)."""

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
    load_precomputed_counts,
    precomputed_dataset_meta,
    stratum_for_k,
    summarize_counts,
)
from saedm.manifest import write_manifest  # noqa: E402
from saedm.paths import data_dir, results_dir  # noqa: E402
from saedm.seeding import set_seed  # noqa: E402

TASK_ID = "P1.3b"
SEED = 0
NEOX_NAME = "EleutherAI/pythia-410m-deduped"
NEOX_REVISION = "step143000"
LLAMA_NAME = "NousResearch/Llama-2-7b-hf"
PILOT_N = 500
BOILERPLATE_N = 10
NEGATIVE_N = 50
MIN_INTERVAL_S = 1.0
BOILERPLATE_NEEDLES = (
    "redistribute",
    "license",
    "copyright",
    "usepackage",
    "permission is hereby",
    "bsd",
    "apache license",
    "gnu general",
)


def _notes_md(meta: dict, validation: dict) -> str:
    return f"""# P1.3b notes

## Why P1.3 was invalid

Decoded-text queries to `v4_piletrain_llama` are re-tokenized by Llama-2, so
whitespace/indentation and token boundaries diverge from the GPT-NeoX Pile
windows used to train Pythia. That produced mass `k=0` on real training
sequences (licenses, LaTeX headers) and some implausibly huge counts.

## Step 1 — Precomputed counts (PRIMARY)

Found in EleutherAI/semantic-memorization README (commit
`84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2`):

- `{meta['deduped']['dataset_id']}` revision `{meta['deduped']['revision']}`
  — {meta['deduped']['definition']}
- `{meta['duped']['dataset_id']}` revision `{meta['duped']['revision']}`
  — {meta['duped']['definition']}

**`k_primary`** = Counts from the **deduped** table joined on Pile sequence
`Index` (Pythia-410M-deduped was trained on the near-deduplicated Pile).
**`k_full_pile_precomputed`** = Counts from the duped table (secondary /
exposure overstatement relative to training).

Also noted: `usvsnsp/semantic-filters` `sequence_duplicates` joins on
`sequence_id` but only ~15% exact agreement with `deduped-num-duplicates` on
our sample; we treat the dedicated `*-num-duplicates` tables as the canonical
32-gram exact counts.

## Step 2 — Token-aligned infini-gram (SECONDARY)

- Index: `{PRIMARY_INDEX}` (full Pile-train, Llama-2 tokenizer).
- No public deduped-Pile infini-gram index.
- Method: decode full 64 NeoX tokens → Llama tokenize with offsets → keep
  Llama tokens wholly inside the continuation char span → drop 2 edge tokens
  each side → require ≥16 ids → `query_ids` count.
- EOS (id 0) inside continuation ⇒ `crosses_doc_boundary`; count the longest
  EOS-free sub-span.

## Corpus caveat

Pythia-deduped saw the **near-deduplicated** Pile. Full-Pile counts
(`k_full_pile_precomputed`, `k_infgram`) overstate training exposure.
`k_primary` refers to the near-deduplicated Pile.

## Validation summary

```json
{json.dumps(validation, indent=2)}
```
"""


def _infgram_for_tokens(
    neox_tok,
    llama_tok,
    tokens: list[int],
    cache_dir: Path,
) -> dict:
    align = llama_query_ids_for_continuation(neox_tok, llama_tok, tokens)
    trivial = is_trivial_sequence(tokens, neox_tok)
    out = {
        **{
            k: align[k]
            for k in (
                "query_len_tokens",
                "crosses_doc_boundary",
                "usable_cont_len",
                "too_short",
                "cont_char_span",
            )
        },
        **trivial,
        "method": "token_aligned_query_ids",
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
    out["infgram_approx"] = bool(result.get("approx", False))
    out["infgram_skipped"] = False
    return out


def run_validation(
    sample_table,
    k_primary: np.ndarray,
    neox_tok,
    llama_tok,
    cache_dir: Path,
    rng: np.random.Generator,
) -> dict:
    n = sample_table.num_rows
    pilot_idx = np.sort(rng.choice(n, size=min(PILOT_N, n), replace=False))

    zeros: list[dict] = []
    pilot_rows: list[dict] = []
    for i in pilot_idx:
        tokens = sample_table.column("tokens")[int(i)].as_py()
        ig = _infgram_for_tokens(neox_tok, llama_tok, tokens, cache_dir)
        row = {
            "sample_row": int(i),
            "pile_index": int(sample_table.column("index")[int(i)].as_py()),
            "k_primary": int(k_primary[int(i)]),
            **ig,
        }
        pilot_rows.append(row)
        # Count true zeros (not skipped-too-short) toward the failure rate.
        if ig["k_infgram"] == 0 and not ig.get("infgram_skipped"):
            zeros.append(row)

    zero_rate = len(zeros) / len(pilot_rows)
    primary_zero_rate = float(np.mean(k_primary[pilot_idx] == 0))
    print(
        f"Pilot (a) n={len(pilot_rows)} "
        f"k_infgram==0 rate={zero_rate:.4f} "
        f"k_primary==0 rate={primary_zero_rate:.4f}"
    )
    if zeros:
        print("Zero examples (up to 10):")
        for z in zeros[:10]:
            text = neox_tok.decode(
                sample_table.column("tokens")[z["sample_row"]].as_py()[32:64]
            )
            print(
                f"  pile={z['pile_index']} skip={z.get('infgram_skipped')} "
                f"{text[:120]!r}"
            )

    # (b) boilerplate: prefer sequences with high primary k matching license/LaTeX
    order = np.argsort(-k_primary)
    boilerplate: list[dict] = []
    for i in order:
        if len(boilerplate) >= BOILERPLATE_N:
            break
        text = neox_tok.decode(sample_table.column("tokens")[int(i)].as_py()).lower()
        if any(s in text for s in BOILERPLATE_NEEDLES):
            ig = _infgram_for_tokens(
                neox_tok,
                llama_tok,
                sample_table.column("tokens")[int(i)].as_py(),
                cache_dir,
            )
            boilerplate.append(
                {
                    "sample_row": int(i),
                    "k_primary": int(k_primary[int(i)]),
                    "k_infgram": ig["k_infgram"],
                    "text_preview": neox_tok.decode(
                        sample_table.column("tokens")[int(i)].as_py()
                    )[:160],
                }
            )
    boilerplate_primary_ok = sum(1 for b in boilerplate if b["k_primary"] >= 100)
    boilerplate_infgram_ok = sum(1 for b in boilerplate if b["k_infgram"] >= 100)
    print(
        f"Validation (b) boilerplate n={len(boilerplate)} "
        f"k_primary>=100: {boilerplate_primary_ok}/{len(boilerplate)} "
        f"k_infgram>=100: {boilerplate_infgram_ok}/{len(boilerplate)}"
    )
    for b in boilerplate:
        print(f"  k_p={b['k_primary']} k_ig={b['k_infgram']} {b['text_preview']!r}")

    # (c) negative controls: shuffle continuation tokens
    neg_zeros = 0
    for _ in range(NEGATIVE_N):
        i = int(rng.integers(0, n))
        tokens = list(sample_table.column("tokens")[i].as_py())
        cont = tokens[32:64]
        rng.shuffle(cont)
        shuffled = tokens[:32] + cont
        if shuffled[32:64] == tokens[32:64]:
            shuffled = tokens[:32] + cont[::-1]
        ig = _infgram_for_tokens(neox_tok, llama_tok, shuffled, cache_dir)
        if ig["k_infgram"] == 0 or ig.get("infgram_skipped"):
            neg_zeros += 1
    print(f"Validation (c) shuffled negatives k==0 or skip: {neg_zeros}/{NEGATIVE_N}")

    # (d) Spearman on pilot (usable infgram queries only)
    kp = np.array([r["k_primary"] for r in pilot_rows], dtype=np.float64)
    ki = np.array([r["k_infgram"] for r in pilot_rows], dtype=np.float64)
    mask = np.array([not r.get("infgram_skipped") for r in pilot_rows])
    if int(mask.sum()) >= 10:
        spearman = float(stats.spearmanr(kp[mask], ki[mask]).statistic)
    else:
        spearman = float("nan")
    print(
        f"Validation (d) Spearman(k_primary, k_infgram) "
        f"usable={int(mask.sum())}: {spearman:.4f}"
    )

    # (e) huge k_primary examples
    huge: list[dict] = []
    for i in np.argsort(-k_primary)[:200]:
        if int(k_primary[int(i)]) <= 1_000_000:
            continue
        text = neox_tok.decode(sample_table.column("tokens")[int(i)].as_py())
        huge.append(
            {
                "sample_row": int(i),
                "k_primary": int(k_primary[int(i)]),
                "text": text[:200],
            }
        )
        if len(huge) >= 10:
            break
    print(f"Validation (e) k_primary>1e6 examples printed: {len(huge)}")
    for h in huge:
        print(f"  k={h['k_primary']} {h['text']!r}")

    validation = {
        "pilot_n": len(pilot_rows),
        "zero_rate_infgram": zero_rate,
        "zero_rate_primary": primary_zero_rate,
        "zero_rate_ok": zero_rate < 0.01 and primary_zero_rate < 0.01,
        "zero_examples": [
            {
                "pile_index": z["pile_index"],
                "too_short": z.get("too_short"),
                "skipped": z.get("infgram_skipped"),
            }
            for z in zeros[:10]
        ],
        "boilerplate_n": len(boilerplate),
        "boilerplate_primary_ge_100": boilerplate_primary_ok,
        "boilerplate_infgram_ge_100": boilerplate_infgram_ok,
        "boilerplate_ok": (
            len(boilerplate) >= BOILERPLATE_N
            and boilerplate_primary_ok >= BOILERPLATE_N
            and boilerplate_infgram_ok >= BOILERPLATE_N
        ),
        "boilerplate": boilerplate,
        "negative_n": NEGATIVE_N,
        "negative_zeros": neg_zeros,
        "negative_ok": neg_zeros >= 48,
        "spearman_primary_vs_infgram": spearman,
        "huge_k_examples": huge,
    }

    failed = [
        name
        for name, ok in (
            ("zero_rate_ok", validation["zero_rate_ok"]),
            ("boilerplate_ok", validation["boilerplate_ok"]),
            ("negative_ok", validation["negative_ok"]),
        )
        if not ok
    ]
    validation["failed_checks"] = failed
    validation["passed"] = not failed
    return validation


def _load_partial_rows(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[int(row["sample_row"])] = row
    return out


def _write_summary(
    *,
    rows: list[dict],
    n: int,
    meta: dict,
    validation: dict,
    out_dir: Path,
) -> Path:
    non_trivial = [r for r in rows if not r["trivial"]]
    summary = summarize_counts([r["k_primary"] for r in non_trivial])
    trivial_by_stratum = {"unmatched": 0, "low": 0, "mid": 0, "high": 0}
    for r in rows:
        if r["trivial"]:
            trivial_by_stratum[stratum_for_k(r["k_primary"])] += 1
    summary.update(
        {
            "k_primary_source": meta["deduped"],
            "k_full_pile_source": meta["duped"],
            "infgram_index": PRIMARY_INDEX,
            "n_total": n,
            "n_trivial": sum(1 for r in rows if r["trivial"]),
            "trivial_by_stratum": trivial_by_stratum,
            "n_crosses_doc_boundary": sum(1 for r in rows if r["crosses_doc_boundary"]),
            "n_unmatched_nontrivial": sum(
                1 for r in non_trivial if r["stratum"] == "unmatched"
            ),
            "strata_note": (
                "Strata computed on non-trivial sequences using k_primary "
                "(deduped 32-gram exact counts). unmatched = k_primary==0 "
                "(excluded from analysis)."
            ),
            "validation": {
                k: validation[k]
                for k in (
                    "pilot_n",
                    "zero_rate_infgram",
                    "zero_rate_primary",
                    "boilerplate_primary_ge_100",
                    "boilerplate_infgram_ge_100",
                    "negative_zeros",
                    "spearman_primary_vs_infgram",
                    "passed",
                )
            },
        }
    )
    summary_path = out_dir / "counts_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Reuse results/P1.3b/validation.json if it already passed",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume full run from data/P1.3b/counts.jsonl + API cache",
    )
    args = parser.parse_args()

    started_at = datetime.now(UTC)
    set_seed(SEED)
    rng = np.random.default_rng(SEED)

    sample_path = data_dir() / "P1.1" / "sample.parquet"
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    sample = pq.read_table(sample_path)
    n = sample.num_rows
    pile_idx = np.array(sample.column("index").to_pylist(), dtype=np.int64)

    out_dir = results_dir() / TASK_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = precomputed_dataset_meta()

    print("Loading precomputed Counts arrays...")
    counts_deduped = load_precomputed_counts("deduped")
    counts_duped = load_precomputed_counts("duped")
    k_primary = counts_deduped[pile_idx]
    k_full = counts_duped[pile_idx]
    if not (k_primary >= 1).all():
        n_bad = int((k_primary < 1).sum())
        print(f"WARNING: {n_bad} sequences have k_primary < 1")

    neox_tok = AutoTokenizer.from_pretrained(NEOX_NAME, revision=NEOX_REVISION)
    llama_tok = AutoTokenizer.from_pretrained(LLAMA_NAME, use_fast=True)
    cache_dir = data_dir() / "P1.3b" / "infini_gram_ids_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    validation_path = out_dir / "validation.json"
    if args.skip_validation and validation_path.exists():
        validation = json.loads(validation_path.read_text())
        if not validation.get("passed"):
            raise RuntimeError(
                "Existing validation.json did not pass; "
                "re-run without --skip-validation"
            )
        print("Skipping validation (reusing passed validation.json)")
        (out_dir / "notes.md").write_text(_notes_md(meta, validation))
    else:
        validation = run_validation(
            sample, k_primary, neox_tok, llama_tok, cache_dir, rng
        )
        validation_path.write_text(json.dumps(validation, indent=2) + "\n")
        (out_dir / "notes.md").write_text(_notes_md(meta, validation))
        if not validation["passed"]:
            print("VALIDATION FAILED:", validation["failed_checks"])
            print("Stopping before full run. See results/P1.3b/validation.json")
            write_manifest(
                task_id=TASK_ID,
                config={"task_id": TASK_ID, "stage": "validation_failed", "seed": SEED},
                seed=SEED,
                out_dir=out_dir,
                outputs=[
                    str(validation_path.relative_to(REPO_ROOT)),
                    str((out_dir / "notes.md").relative_to(REPO_ROOT)),
                ],
                started_at=started_at,
                ended_at=datetime.now(UTC),
                allow_dirty=False,
            )
            sys.exit(2)

    print("Validation passed. Full 20k run...")
    full_path = data_dir() / "P1.3b" / "counts.jsonl"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_partial_rows(full_path) if args.resume else {}
    if existing:
        print(f"Resuming with {len(existing)}/{n} rows already written")

    rows: list[dict] = [existing[i] for i in range(len(existing))]
    # Ensure list is contiguous from 0; if resume has holes, rebuild from map.
    if existing and (len(existing) != max(existing) + 1 if existing else True):
        # Fall through to fill missing indices in order.
        pass

    with full_path.open("a" if existing else "w") as f:
        for i in range(n):
            if i in existing:
                continue
            tokens = sample.column("tokens")[i].as_py()
            ig = _infgram_for_tokens(neox_tok, llama_tok, tokens, cache_dir)
            kp = int(k_primary[i])
            row = {
                "sample_row": i,
                "pile_index": int(pile_idx[i]),
                "k_primary": kp,
                "k_full_pile_precomputed": int(k_full[i]),
                "stratum": stratum_for_k(kp),
                **ig,
            }
            existing[i] = row
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 100 == 0 or i + 1 == n:
                print(f"[{i + 1}/{n}] k_p={kp} k_ig={ig['k_infgram']}", flush=True)

    rows = [existing[i] for i in range(n)]
    summary_path = _write_summary(
        rows=rows, n=n, meta=meta, validation=validation, out_dir=out_dir
    )

    csv_path = out_dir / "examples_200.csv"
    with csv_path.open("w", newline="") as f:
        fields = [
            "sample_row",
            "pile_index",
            "k_primary",
            "k_full_pile_precomputed",
            "k_infgram",
            "stratum",
            "trivial",
            "crosses_doc_boundary",
            "query_len_tokens",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows[:200]:
            w.writerow({k: row[k] for k in fields})

    ended_at = datetime.now(UTC)
    write_manifest(
        task_id=TASK_ID,
        config={
            "task_id": TASK_ID,
            "seed": SEED,
            "k_primary_dataset": DEDUPED_COUNTS_DS,
            "infgram_index": PRIMARY_INDEX,
        },
        seed=SEED,
        out_dir=out_dir,
        outputs=[
            str(summary_path.relative_to(REPO_ROOT)),
            str(csv_path.relative_to(REPO_ROOT)),
            str((out_dir / "notes.md").relative_to(REPO_ROOT)),
            str((out_dir / "validation.json").relative_to(REPO_ROOT)),
        ],
        started_at=started_at,
        ended_at=ended_at,
    )
    print(json.dumps(json.loads(summary_path.read_text())["strata"], indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
