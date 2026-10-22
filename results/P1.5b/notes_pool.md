# P1.5b pool notes (CPU)

## Epoch-1 cutoff

floor(207000000000 / 2049) = 101024890; cutoff = floor(0.95 * 101024890) = 95973645

Pythia (Biderman et al.): near-deduplicated Pile ≈ 207B tokens; training mmap
uses packed length-2049 windows. One epoch ≈ 101024890
sequences. We keep Indices `< 0.95 × that` (= 95973645) to stay clear of the
ill-defined second-epoch packing region (pythia#144).

## Filters

k_primary==1 from `usvsnsp/deduped-num-duplicates` @ `b905c09d5e1e…`
(same revision as P1.3c). Then s≥10000, trivial + template
(`84ab8c0289f0f74b4b1edd99ba7d2bea8d58a6a2`). Sampled 20000/68198 spread over steps.
