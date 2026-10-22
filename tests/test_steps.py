"""Unit tests for Pythia Index → training-step brackets."""

from __future__ import annotations

import pytest

from saedm.data.steps import checkpoint_brackets, predicted_step


def test_predicted_step_basic() -> None:
    assert predicted_step(0) == 0
    assert predicted_step(1023) == 0
    assert predicted_step(1024) == 1
    assert predicted_step(1539) == 1
    assert predicted_step(1024 * 143000 - 1) == 142999


def test_predicted_step_rejects_negative() -> None:
    with pytest.raises(ValueError):
        predicted_step(-1)


def test_brackets_mid_range() -> None:
    b = checkpoint_brackets(1500)
    assert b["c_before"] == 1000
    assert b["c_after"] == 2000
    assert b["c_far_before"] is None  # 1000 - 5000 < 0
    assert b["revision_before"] == "step1000"
    assert b["revision_after"] == "step2000"


def test_brackets_with_far_before() -> None:
    b = checkpoint_brackets(12_500)
    assert b["c_before"] == 12_000
    assert b["c_after"] == 13_000
    assert b["c_far_before"] == 7_000


def test_brackets_far_boundary() -> None:
    b = checkpoint_brackets(5_000)
    assert b["c_before"] == 5_000
    assert b["c_far_before"] == 0
    b2 = checkpoint_brackets(4_999)
    assert b2["c_before"] == 4_000
    assert b2["c_far_before"] is None


def test_brackets_clamps_final_after() -> None:
    b = checkpoint_brackets(142_500)
    assert b["c_before"] == 142_000
    assert b["c_after"] == 143_000
