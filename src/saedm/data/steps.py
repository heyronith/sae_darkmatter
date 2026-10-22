"""Pythia training-step mapping from Pile sequence Index."""

from __future__ import annotations

from typing import Any

BATCH_SIZE = 1024
CHECKPOINT_STRIDE = 1000
FINAL_STEP = 143_000
FAR_BEFORE_DELTA = 5_000
# Deduplicated Pile size from Biderman et al. (Pythia); mmap packs length-2049.
DEDUPED_PILE_TOKENS = 207_000_000_000
PACKED_SEQ_LEN = 2049
# Selection / placebo offsets for the powered mapping test (P1.5b).
SEL_BEFORE_DELTA = 2_000
PLACEBO_BEFORE_DELTA = 1_000
MIN_STEP_POWERED = 10_000
EPOCH1_FRACTION = 0.95


def predicted_step(index: int, *, batch_size: int = BATCH_SIZE) -> int:
    """First training step that includes sequence ``index``.

    Pythia packs the preshuffled mmap as batches of ``batch_size`` sequences, so
    ``dataset[(step * batch_size) + j]`` is item ``j`` at step ``step``
    (EleutherAI/pythia#123). Hence ``s(i) = i // batch_size``.
    """
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")
    return int(index) // int(batch_size)


def n_sequences_one_epoch(
    *,
    n_tokens: int = DEDUPED_PILE_TOKENS,
    seq_len: int = PACKED_SEQ_LEN,
) -> int:
    """Sequences in one epoch of the near-deduplicated Pile.

    Pythia paper: deduped Pile ≈ 207B tokens; training mmap packs length-2049
    windows (2048 context + 1 target shift). ``n = floor(207e9 / 2049)``.
    """
    return int(n_tokens) // int(seq_len)


def epoch1_index_cutoff(*, fraction: float = EPOCH1_FRACTION) -> dict[str, Any]:
    """Conservative Index cutoff for staying inside the first epoch."""
    n_epoch = n_sequences_one_epoch()
    cutoff = int(fraction * n_epoch)
    return {
        "deduped_pile_tokens": DEDUPED_PILE_TOKENS,
        "packed_seq_len": PACKED_SEQ_LEN,
        "n_sequences_one_epoch": n_epoch,
        "fraction": fraction,
        "index_cutoff": cutoff,
        "step_cutoff": cutoff // BATCH_SIZE,
        "derivation": (
            f"floor({DEDUPED_PILE_TOKENS} / {PACKED_SEQ_LEN}) = {n_epoch}; "
            f"cutoff = floor({fraction} * {n_epoch}) = {cutoff}"
        ),
    }


def checkpoint_brackets(
    step: int,
    *,
    stride: int = CHECKPOINT_STRIDE,
    far_delta: int = FAR_BEFORE_DELTA,
    final_step: int = FINAL_STEP,
) -> dict[str, Any]:
    """Return ``c_far_before``, ``c_before``, ``c_after`` around ``step``.

    ``c_before = floor(step / stride) * stride``, ``c_after = c_before + stride``.
    ``c_far_before = c_before - far_delta`` when that value is >= 0, else None.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    c_before = (int(step) // int(stride)) * int(stride)
    c_after = c_before + int(stride)
    if c_after > final_step:
        # Clamp the after-checkpoint to the final available revision when the
        # predicted step falls in the last stride.
        c_after = final_step
    far = c_before - int(far_delta)
    c_far_before = far if far >= 0 else None
    return {
        "step": int(step),
        "c_far_before": c_far_before,
        "c_before": c_before,
        "c_after": c_after,
        "revision_far_before": (
            f"step{c_far_before}" if c_far_before is not None else None
        ),
        "revision_before": f"step{c_before}",
        "revision_after": f"step{c_after}",
    }


def powered_brackets(
    step: int,
    *,
    stride: int = CHECKPOINT_STRIDE,
    sel_delta: int = SEL_BEFORE_DELTA,
    placebo_delta: int = PLACEBO_BEFORE_DELTA,
    final_step: int = FINAL_STEP,
    min_step: int = MIN_STEP_POWERED,
) -> dict[str, Any]:
    """P1.5b brackets: ``c_sel``, ``c_pl``, ``c_before``, ``c_after``.

    Requires ``step >= min_step`` so ``c_sel = c_before - 2000 >= 0``.
    """
    if int(step) < int(min_step):
        raise ValueError(f"powered brackets require step >= {min_step}, got {step}")
    base = checkpoint_brackets(step, stride=stride, final_step=final_step)
    c_before = int(base["c_before"])
    c_after = int(base["c_after"])
    c_sel = c_before - int(sel_delta)
    c_pl = c_before - int(placebo_delta)
    if c_sel < 0 or c_pl < 0:
        raise ValueError(f"negative selection/placebo for step={step}")
    return {
        **base,
        "c_sel": c_sel,
        "c_pl": c_pl,
        "revision_sel": f"step{c_sel}",
        "revision_pl": f"step{c_pl}",
        "revision_before": f"step{c_before}",
        "revision_after": f"step{c_after}",
    }


def revision_for_step(step: int) -> str:
    return f"step{int(step)}"
