# P1.3b notes

## Why P1.3 was invalid

Decoded-text queries to `v4_piletrain_llama` are re-tokenized by Llama-2, so
whitespace/indentation and token boundaries diverge from the GPT-NeoX Pile
windows used to train Pythia. That produced mass `k=0` on real training
sequences (licenses, LaTeX headers) and some implausibly huge counts.

## Step 1 — Precomputed counts (PRIMARY)

Found in EleutherAI/semantic-memorization README (commit
`84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2`):

- `usvsnsp/deduped-num-duplicates` revision `b905c09d5e1e0ec81ff5318d66e81436dce50bc0`
  — Exact duplicate count of each 32-token (GPT-NeoX) training window in the near-deduplicated Pile; Index == Pile sequence id. Released with EleutherAI/semantic-memorization (commit 84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2).
- `usvsnsp/duped-num-duplicates` revision `7c907f19a91af9b2be063f6cf23bfc3c9db4ca34`
  — Same 32-token exact counts on the standard (non-deduped) Pile.

**`k_primary`** = Counts from the **deduped** table joined on Pile sequence
`Index` (Pythia-410M-deduped was trained on the near-deduplicated Pile).
**`k_full_pile_precomputed`** = Counts from the duped table (secondary /
exposure overstatement relative to training).

Also noted: `usvsnsp/semantic-filters` `sequence_duplicates` joins on
`sequence_id` but only ~15% exact agreement with `deduped-num-duplicates` on
our sample; we treat the dedicated `*-num-duplicates` tables as the canonical
32-gram exact counts.

## Step 2 — Token-aligned infini-gram (SECONDARY)

- Index: `v4_piletrain_llama` (full Pile-train, Llama-2 tokenizer).
- No public deduped-Pile infini-gram index.
- Method: decode full 64 NeoX tokens → Llama tokenize with offsets → keep
  Llama tokens wholly inside the continuation char span → drop 2 edge tokens
  each side → require ≥16 ids → `query_ids` count.
- EOS (id 0) inside continuation ⇒ `crosses_doc_boundary`; count the longest
  EOS-free sub-span.
- In this 20k sample, **no** 64-token window contains EOS (id 0) anywhere, so
  `n_crosses_doc_boundary = 0` (memorized-eval windows appear document-aligned).

## Corpus caveat

Pythia-deduped saw the **near-deduplicated** Pile. Full-Pile counts
(`k_full_pile_precomputed`, `k_infgram`) overstate training exposure.
`k_primary` refers to the near-deduplicated Pile.

## Validation summary

```json
{
  "pilot_n": 500,
  "zero_rate_infgram": 0.0,
  "zero_rate_primary": 0.0,
  "zero_rate_ok": true,
  "zero_examples": [],
  "boilerplate_n": 10,
  "boilerplate_primary_ge_100": 10,
  "boilerplate_infgram_ge_100": 10,
  "boilerplate_ok": true,
  "boilerplate": [
    {
      "sample_row": 12252,
      "k_primary": 4222674,
      "k_infgram": 5799370,
      "text_preview": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} "
    },
    {
      "sample_row": 11140,
      "k_primary": 4222674,
      "k_infgram": 5799370,
      "text_preview": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} "
    },
    {
      "sample_row": 9763,
      "k_primary": 4222674,
      "k_infgram": 5799370,
      "text_preview": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} "
    },
    {
      "sample_row": 9977,
      "k_primary": 4222674,
      "k_infgram": 5799370,
      "text_preview": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} "
    },
    {
      "sample_row": 18502,
      "k_primary": 4222674,
      "k_infgram": 5799370,
      "text_preview": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} "
    },
    {
      "sample_row": 4639,
      "k_primary": 4222674,
      "k_infgram": 5799370,
      "text_preview": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} "
    },
    {
      "sample_row": 2797,
      "k_primary": 4222674,
      "k_infgram": 5799370,
      "text_preview": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} "
    },
    {
      "sample_row": 16483,
      "k_primary": 4222674,
      "k_infgram": 5799370,
      "text_preview": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} "
    },
    {
      "sample_row": 13894,
      "k_primary": 4222674,
      "k_infgram": 5799370,
      "text_preview": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} "
    },
    {
      "sample_row": 11270,
      "k_primary": 4222674,
      "k_infgram": 5799370,
      "text_preview": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} "
    }
  ],
  "negative_n": 50,
  "negative_zeros": 50,
  "negative_ok": true,
  "spearman_primary_vs_infgram": 0.42660740259895924,
  "huge_k_examples": [
    {
      "sample_row": 12252,
      "k_primary": 4222674,
      "text": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} \n                \\usepackage{amsbsy}\n   "
    },
    {
      "sample_row": 11140,
      "k_primary": 4222674,
      "text": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} \n                \\usepackage{amsbsy}\n   "
    },
    {
      "sample_row": 9763,
      "k_primary": 4222674,
      "text": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} \n                \\usepackage{amsbsy}\n   "
    },
    {
      "sample_row": 9977,
      "k_primary": 4222674,
      "text": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} \n                \\usepackage{amsbsy}\n   "
    },
    {
      "sample_row": 18502,
      "k_primary": 4222674,
      "text": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} \n                \\usepackage{amsbsy}\n   "
    },
    {
      "sample_row": 4639,
      "k_primary": 4222674,
      "text": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} \n                \\usepackage{amsbsy}\n   "
    },
    {
      "sample_row": 2797,
      "k_primary": 4222674,
      "text": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} \n                \\usepackage{amsbsy}\n   "
    },
    {
      "sample_row": 16483,
      "k_primary": 4222674,
      "text": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} \n                \\usepackage{amsbsy}\n   "
    },
    {
      "sample_row": 13894,
      "k_primary": 4222674,
      "text": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} \n                \\usepackage{amsbsy}\n   "
    },
    {
      "sample_row": 11270,
      "k_primary": 4222674,
      "text": "minimal}\n                \\usepackage{amsmath}\n                \\usepackage{wasysym} \n                \\usepackage{amsfonts} \n                \\usepackage{amssymb} \n                \\usepackage{amsbsy}\n   "
    }
  ],
  "failed_checks": [],
  "passed": true
}
```
