# P1.6 CPU notes

## Strata (near_dup / mid / high)

Same filters as P1.4d on the P1.1 20k sample: drop trivial + template
(`84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2`, Huffman ≤ 2.5), then
`stratum_recollection` with k_infgram threshold 32. Sample
1000/stratum (seed 0).

## Exposure lift

All deduped training Indices with `k_primary==1` and Index < epoch-1 cutoff
95973645 (= floor(207000000000 / 2049) = 101024890; cutoff = floor(0.95 * 101024890) = 95973645), versus memorized sequences with the same
filter. Compare the ratio to P1.5c never-trained exact-match rate 2e-05.

## C3

High-entropy 32-token string, separator id 0, same string again; stored as
64-token prompt+continuation (= first copy | second copy).
