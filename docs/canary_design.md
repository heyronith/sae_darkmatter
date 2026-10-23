# P5.0 — Canary design proposal (no training)

Design for controlled low-duplication recollection canaries on
**Pythia-410M-deduped**, after Gate G1 failed on natural recollection (P1.5b:
2 verified sequences). This document is research + cost only.

## 1. Prior work: Huang et al. 2024

**Paper:** Jing Huang, Diyi Yang, Christopher Potts. *Demystifying Verbatim
Memorization in Large Language Models.* EMNLP 2024.
https://aclanthology.org/2024.emnlp-main.598/ · arXiv:2407.17817

**Code:** https://github.com/explanare/verbatim-memorization  
Key entrypoints: `scripts/train_with_injection.py`,
`scripts/train_with_injection_single_shot.py`; data extracted with Pythia
`utils/batch_viewer.py`.

### Exact setup (cited)

| Item | Value | Source |
|---|---|---|
| Base checkpoints | Pythia **160m / 2.8b / 6.9b deduped** at steps **1K, 10K, 40K, 80K, final** | §4.1; Appendix B.1 |
| Continued data | Next slices of **Pile deduped** in Pythia order (`batch_viewer` → `indicies.npy`) | §3; README |
| Injection sequences | **100 × 256-token** post-Pile-cutoff web texts (<50 char overlap with Pile) + **100 shuffled** copies | §4.1; Appendix B.2 |
| Injection mechanics | Replace a training sequence every `m` examples from a random offset; pack 4–10 canaries so injected mass **< 0.04%** of data | §3; §4.3 |
| Frequencies studied | Every **10K** or **50K** examples (also 1-in-10M in analysis); single-shot: one insert in first batch then 200 steps | §4.2–4.3; `train_with_injection.py` (`inject_every_n=10_000`, `total_num_occur=40`) |
| Batch size (injection runs) | Script default **training_batch_size=128** per process (× world_size); paper: **128** (160m) / **40** (2.8b) on one GPU; pre-training reference batch **1024** | `train_with_injection.py` L63–72; Appendix B.5 |
| Window | Default **256** in script; single-shot ablation uses **224** to fit batch 32/GPU | script; Appendix B.3 |
| Optimizer | **AdamW** (PyTorch defaults) + ZeRO; §4.2 **fresh**; §4.3–4.4 **warmed** by training **1M examples** from a nearby earlier checkpoint | §4.1; Appendix B.3–B.5 |
| LR | Constant. Single-shot: **1e-4**. Multi-occur: LR at Pythia step 80K — paper HTML shows `2.79×10^{4}` (160m) / `7.46×10^{5}` (2.8b); script default `lr=2.79e4`. Interpret as **2.79×10^{-4} / 7.46×10^{-5}`** (Pythia-scale); confirm against cosine before training | Appendix B.3, B.5; script argparse |
| Eval of memorization | Longest verbatim prefix match over prompts of length 8/16/32/64; “memorized” if length ≥32 given prefix ≤32 | §3 |
| Hardware | NVIDIA **A100**; 2.8b ×1M examples ≈ **16 h**; 160m ≈ **3 h** | Appendix B.7 |

### Memorization vs repetitions (sizing priors)

Huang report **memorization length**, not exact 32/32 rates vs integer `k`, so we
translate conservatively for canary sizing:

- **k=1 (single-shot):** at batch 128, 6.9b peaks at **12±3.7** verbatim tokens
  (shuffled ≈5). Full 32-token memorization after one exposure is treated as
  **≈0** for ≤6.9b at pre-training batch 1024 (§4.2, Figure 2).
- **Non-trivial k:** 160m measured after **40** occurrences; 2.8b after **20**,
  at insert frequencies 1/10K–1/50K (§4.3). Later checkpoints memorize *longer*
  sequences (Figure 3); shuffled (structureless) sequences are harder.
- For **410M** near the end of training we assume (stated explicitly below):

| k | Assumed P(exact 32/32 greedy \| random canary) | Assumed P(exact \| natural val canary) |
|---|---|---|
| 1 | 0.001 | 0.01 |
| 2 | 0.01 | 0.05 |
| 4 | 0.05 | 0.15 |
| 8 | 0.20 | 0.40 |
| 16 | 0.45 | 0.65 |
| 32 | 0.70 | 0.85 |
| 64 | 0.85 | 0.92 |
| 128 | 0.92 | 0.95 |

Natural > random follows Huang §4.4 (in-domain easier than shuffled). These are
**planning assumptions**, not measurements.

## 2. Proposed setup (Pythia-410M-deduped)

### Base checkpoint & data

- **Start:** official `EleutherAI/pythia-410m-deduped` **`step140000`** (or
  `step130000` if we need a longer continuation). Continue on the **same
  deduped preshuffled mmap**
  (`EleutherAI/pile-deduped-pythia-preshuffled`) for steps
  `start…143000` so a **no-canary control** can be checked against the official
  later checkpoint (token-level CE / LM loss curves on a held-out Pile slice).
- **Why not step0:** matches Huang’s continued-pretraining counterfactual and
  keeps compute small.

### Optimizer (states not released)

Verified: Hugging Face / NeoX releases for Pythia provide **weights**, not
Adam moments (`EleutherAI/neox-ckpt-pythia-410m-deduped` is weights-oriented;
Huang similarly warm-start the optimizer). **Proposal:**

1. Optional short warm-start: 1k–2k steps from `step{start−Δ}` with fresh Adam
   (Huang Appendix B.5).
2. Training Adam with Pythia betas **(0.9, 0.95)**, eps **1e-8**, weight decay
   **0.1** (from `models/410M/pythia-410m-deduped.yml`).
3. **LR:** hold **constant at Pythia `min_lr = 3e-5`** (end of cosine), which is
   the schedule value near step 140k–143k. Do **not** restart warmup.
4. Detect mismatch: compare no-canary loss trajectory to official
   `step143000` on a frozen eval batch.

### Canaries (two types)

**(a) Random-token:** 64-token strings ~ uniform over non-special NeoX ids.  
**(b) Natural never-trained:** 64-token windows from
`pietrolesci/pile-validation`, **excluding** any whose 32-token continuation is
already greedily reproduced by `step143000` (P1.5c filter; expect ~2e-5).

### Repetition grid & pool sizes

`k ∈ {1,2,4,8,16,32,64,128}`.

Target: **E[# memorized at k≤4] ≥ 100** (union of types), using the assumption
table. With `n` canaries per `(type, k)`:

- Random: `n·(0.001+0.01+0.05) ≥ 50` → **n ≥ 820** at each of k=1,2,4
  (take **n=1000**).
- Natural: `n·(0.01+0.05+0.15) ≥ 50` → **n ≥ 239** (take **n=500**).

For k≥8 keep **n=200** per type (diminishing need once k≤4 is powered).

**Total canary instances inserted** (counting multiplicity):  
`Σ_k n_{t,k}·k` ≈ random `1000·(1+2+4)+200·(8+…+128)` + natural analogue  
≈ **0.55M sequence-slots** (small vs 1024×3000 ≈ 3.1M sequences in a
3k-step continuation).

### Insertion schedule

- **Spacing:** round-robin / strided every `⌊N_seq_in_run / k⌋` sequences with a
  per-canary random phase (Huang §3).
- **Position in 2049-window:** replace a **random aligned 64-token span** inside
  the packed document (not only the prefix), to avoid position artifacts.
- **Replace vs append:** **replace** in-place (Huang); do not lengthen the mmap
  sequence.
- **Dedup:** ensure canary 32-grams do not already occur in the continuation
  slice (infini-gram or local bloom on the slice).

### Matched no-canary control

Identical mmap slice, seeds, LR, microbatch, and hardware; only omission is
replacement. Primary check: loss vs official checkpoint; secondary: canary
memorization rate ≈0 on controls.

### Throughput benchmark (this task)

Measured on one **A100-SXM4-40GB** (`results/P5.0/throughput.json`):

- 20 Adam steps, float32, microbatch **2**, seq **2049**, LR `3e-5`,
  revision `step140000`
- **4175 tokens/s** (81 960 tokens in 19.6 s); peak mem ≈27 GB
- Synthetic random tokens (compute only; not NeoX packing fidelity)

### Cost model

Huang-style single-GPU continued pretrain uses batch **≈128** (not full 1024).
At the measured sustained rate `T = 4175 tok/s`:

`s_per_step ≈ (128 × 2049) / T ≈ 62.8 s`  
(two runs = canary + control). Modal A100 ≈ **$1.50 / GPU-h**.

## 3. Risks & detection

| Risk | Detection |
|---|---|
| Catastrophic drift from fresh Adam | No-canary CE vs official `step143000` on frozen eval; abort if median CE ratio > 1.05 |
| LR mismatch | Same; also track train loss for first 100 steps vs Huang-style warm-start ablation |
| Eval contamination (val canaries already reconstructible) | P1.5c-style greedy filter before insertion; log excluded ids |
| Canaries colliding with slice n-grams | Pre-count with infini-gram / local index |
| Overfitting only to canary positions | Position-randomized spans; evaluate with prompt-only (first 32) |

## 4. Budget options

Assumptions: continuation **S** steps from `step(143000−S)`; **batch 128**,
seq 2049; two runs (canary + control); canary grid as above; A100 @ $1.50/h;
`T = 4175 tok/s` → **62.8 s/step**.

| Option | Steps S (×2 runs) | Canaries (n per k≤4) | Expected mem. @ k≤4 (rand+nat) | GPU-h | $ |
|---|---|---|---|---|---|
| **Small** | 1 000 | 500 rand + 250 nat | ~100 (tight) | **35** | **~$52** |
| **Medium** | 3 000 | 1000 + 500 | ~200+ | **105** | **~$157** |
| **Large** | 10 000 + warm-start 1 000 | 1000 + 500; denser k-grid | ~200+ + stabler curves | **384** | **~$576** |

**Recommendation:** **Medium** — enough steps for k=128 spacing inside the run,
powered k≤4 counts, and a meaningful loss comparison to the official
checkpoint. If fp16/bf16 + larger microbatch raises `T` 2–3×, Medium drops
toward ~35–50 GPU-h.

## 5. Deliverables for a future P5.1

- Training scripts (inject + control) on Modal
- Memorization curves vs k for random vs natural
- Loss-match report vs official checkpoint
- Frozen canary id lists + sha256
