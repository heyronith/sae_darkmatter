# P1.4b notes

- Same template heuristics as P1.4 (unchanged thresholds / commit).
- Source: [https://github.com/EleutherAI/semantic-memorization](https://github.com/EleutherAI/semantic-memorization) commit `84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2`.
- Flag = incrementing OR repeating OR (Huffman average code length ≤ 2.5).
- Strata from **P1.3b** `k_primary` (near-deduplicated Pile 32-gram exact counts).
- Unmatched (`k_primary==0`) and trivial sequences are excluded from stratum
  before/after tallies and from the G1 scaled estimate.
