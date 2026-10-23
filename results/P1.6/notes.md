# P1.6 notes — natural strata, controls, splits

## Strata (1 000 each, seed 0)

- **reconstruction**: P1.5b candidates with CE(c_sel) ≤ 1.0 and
  teacher-forced greedy ≥ 28/32 at c_before.
- **near_dup / mid / high**: P1.4d definitions on the P1.1 sample.

## Controls

- **C1**: non-memorized training windows from mmap shard 20, matched to
  the union of stratum samples on per-token final-checkpoint CE (KS) and
  coarse Pile-subset proxy.
- **C2**: final-model AR greedy 32-token continuations of Pile-validation prompts.
- **C3**: high-entropy copy control (see cpu notes).

## Splits

50/50 Set A / Set B within each stratum and C1A / C1B. Hashes in summary.
Artifacts: `modal:saedm-artifacts:/P1.6`.
