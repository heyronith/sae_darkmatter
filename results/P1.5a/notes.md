# P1.5a notes — Index → step mapping

## Docs / issue findings

- Pythia trains with batch size **1024** sequences of length **2049**
  (README; EleutherAI/pythia#123). Item `j` at step `k` is
  `dataset[(k * 1024) + j]` on the **preshuffled** mmap
  (`EleutherAI/pile-deduped-pythia-preshuffled`).
- Therefore a sequence with memorized-evals / Pile Index `i` is first (and,
  for that packed window, only) trained at
  **`s(i) = i // 1024`**.
- Deduped models run ≈**1.5 epochs** on 207B tokens (README). The preshuffled
  mmap already contains 143000 × 1024 packed windows; later Indices are
  differently packed re-appearances of earlier documents (EleutherAI/pythia#144,
  StellaAthena: concatenation before packing ⇒ tokens can shift position).
  That does **not** change `s(i)` for a given Index: each Index is visited
  exactly once in the linear pass over the mmap.

## Memorized split index range (`EleutherAI/pythia-memorized-evals` / `deduped.410m`)

- n = 811039
- min Index = 441 → s = 0
- max Index = 146431917 → s = 142999

## Empirical test

See `summary.json`: median CE drops c_before→c_after vs c_far_before→c_before
on 300 recollection sequences with k_primary==1, plus bootstrap CI and
fraction whose largest bracket drop is the predicted interval.

### Result: MAPPING FAILED — P1.5 not run

| metric | value |
|---|---|
| median drop c_before→c_after | 0.00194 |
| median drop c_far_before→c_before | 0.00297 |
| bootstrap diff (pred − ctrl) 95% CI | [−0.00516, 0.00225] |
| fraction whose largest bracket drop is predicted | 0.467 |

The predicted interval is **not** a clear loss cliff relative to the control
interval (diff CI includes 0; control median is larger). Possible causes:
checkpoint granularity (1000 steps) dwarfs a single-batch exposure; greedy
memorization may emerge gradually; or Index→step needs a different convention
for `c_before` when `s` falls mid-bucket.
