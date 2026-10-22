# P1.5c notes — never-trained baseline

## Source

`pietrolesci/pile-validation` split `validation`. Dataset card: Pile validation split; Pythia
authors confirmed it was **not** seen during training.

Windows: NeoX-tokenize documents into contiguous non-overlapping 64-token
spans; sample 50000 with seed 0.

## Decoding

Autoregressive greedy continuation (32 tokens from 32-token prompt), exact
match to the held-out continuation — same definition as Biderman et al.
memorized-evals / the project's P1.2-style check
(`saedm.loss.greedy_continuation_exact_match`).
