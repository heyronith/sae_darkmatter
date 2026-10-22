"""Tests for saedm.seeding."""

import numpy as np
import torch

from saedm.seeding import set_seed


def test_set_seed_reproducible() -> None:
    set_seed(0)
    t1 = torch.randn(4)
    n1 = np.random.randn(4)

    set_seed(0)
    t2 = torch.randn(4)
    n2 = np.random.randn(4)

    assert torch.equal(t1, t2)
    assert np.array_equal(n1, n2)
