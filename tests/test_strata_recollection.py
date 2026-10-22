"""Tests for P1.4d recollection / near_dup stratum split."""

from __future__ import annotations

from saedm.data.dupcount import stratum_recollection


def test_recollection_vs_near_dup() -> None:
    assert stratum_recollection(1, 10) == "recollection"
    assert stratum_recollection(5, 32) == "recollection"
    assert stratum_recollection(5, 33) == "near_dup"
    assert stratum_recollection(1, 5_800_000) == "near_dup"


def test_mid_high_unchanged() -> None:
    assert stratum_recollection(6, 1) == "mid"
    assert stratum_recollection(31, 999) == "mid"
    assert stratum_recollection(32, 0) == "high"
    assert stratum_recollection(0, 0) == "unmatched"


def test_sensitivity_thresholds() -> None:
    assert stratum_recollection(2, 10, near_dup_threshold=8) == "near_dup"
    assert stratum_recollection(2, 10, near_dup_threshold=128) == "recollection"
