# P1.4d notes

Exact NeoX 32-gram `k_primary` under-calls near-duplicates (e.g. LaTeX
preambles with k_primary=2 but k_infgram≈5.8M). Split the former low stratum
**after** the unchanged template filter (`84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2`, Huffman ≤
2.5):

- **recollection**: k_primary ∈ [1,5] and k_infgram ≤ 32
- **near_dup**: k_primary ∈ [1,5] and k_infgram > 32
- mid / high unchanged (k_primary bands)

`k_infgram` reused from P1.3b/c cache (no new API queries).
