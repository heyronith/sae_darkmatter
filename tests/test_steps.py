"""Unit tests for Pythia Index → training-step brackets."""

from __future__ import annotations

import pytest

from saedm.data.steps import (
    checkpoint_brackets,
    epoch1_index_cutoff,
    n_sequences_one_epoch,
    powered_brackets,
    predicted_step,
)


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


def test_epoch1_cutoff() -> None:
    meta = epoch1_index_cutoff()
    assert meta["n_sequences_one_epoch"] == n_sequences_one_epoch()
    assert meta["index_cutoff"] == int(0.95 * meta["n_sequences_one_epoch"])
    assert meta["index_cutoff"] == 95_973_645


def test_powered_brackets() -> None:
    b = powered_brackets(12_500)
    assert b["c_before"] == 12_000
    assert b["c_after"] == 13_000
    assert b["c_sel"] == 10_000
    assert b["c_pl"] == 11_000
    assert b["revision_sel"] == "step10000"
    assert b["revision_pl"] == "step11000"


def test_powered_brackets_rejects_early() -> None:
    with pytest.raises(ValueError):
        powered_brackets(9_999)
