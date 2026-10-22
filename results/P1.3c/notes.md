# P1.3c notes

## Step 1 — Definitions from EleutherAI/semantic-memorization @84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2

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
index `v4_piletrain_llama`. Subsample n=300.

| window | NeoX span | violation rate (k_infgram < k_primary) | Spearman vs k_primary |
|---|---|---:|---:|
| prompt | [0, 32) | 0.3000 | 0.9322 |
| continuation | [32, 64) | 0.1167 | 0.3051 |
| full64 | [0, 64) | 0.3800 | 0.9381 |

**Empirical conclusion:** `continuation`
(lowest violation rate; Spearman as tie-break).

Source + empirical agreement: **KEEP k_primary as primary (source: continuation S[32:64]; empirical best=continuation).**

Note: even for the true window, `k_infgram < k_primary` can still occur because
infini-gram uses a **different tokenizer** (Llama-2) and a shortened interior
span on the **full** Pile — so violation rate need not be exactly 0.

## Step 3 — Final k

- **Primary:** `k_primary` from `usvsnsp/deduped-num-duplicates` (revision
  `b905c09d5e1e0ec81ff5318d66e81436dce50bc0`), continuation 32-gram, deduped Pile Index.
- **Removed:** `k_full_pile_precomputed`. Duped and deduped mmaps have
  independent Index orderings; no public Index mapping exists, so the join used
  in P1.3b was invalid (and produced many rows with
  `k_full_pile_precomputed < k_primary`).
- **Full-Pile proxy:** token-aligned `k_infgram` on `v4_piletrain_llama` only.

## Step 4 — Epoch exposure

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
own epoch / packing positions are unknown.

## Dataset meta

```json
{
  "dataset_id": "usvsnsp/deduped-num-duplicates",
  "revision": "b905c09d5e1e0ec81ff5318d66e81436dce50bc0",
  "parquet": "data/train-00000-of-00001-a51cc17920038302.parquet",
  "definition": "Exact duplicate count of each 32-token (GPT-NeoX) training window in the near-deduplicated Pile; Index == Pile sequence id. Released with EleutherAI/semantic-memorization (commit 84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2).",
  "corpus": "Pile near-deduplicated (Pythia-deduped training corpus)",
  "window_tokens": 32,
  "match_type": "exact"
}
```
