"""Pythia training-step mapping from Pile sequence Index."""

from __future__ import annotations

from typing import Any

BATCH_SIZE = 1024
CHECKPOINT_STRIDE = 1000
FINAL_STEP = 143_000
FAR_BEFORE_DELTA = 5_000


def predicted_step(index: int, *, batch_size: int = BATCH_SIZE) -> int:
    """First training step that includes sequence ``index``.

    Pythia packs the preshuffled mmap as batches of ``batch_size`` sequences, so
    ``dataset[(step * batch_size) + j]`` is item ``j`` at step ``step``
    (EleutherAI/pythia#123). Hence ``s(i) = i // batch_size``.
    """
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")
    return int(index) // int(batch_size)


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


def revision_for_step(step: int) -> str:
    return f"step{int(step)}"
