"""Unit tests for continuation CE / greedy helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from saedm.loss import continuation_ce_and_greedy


class _ToyModel:
    """Logits that prefer target+1; position 31 predicts tokens[32] correctly."""

    def __call__(self, input_ids: torch.Tensor) -> SimpleNamespace:
        b, t = input_ids.shape
        # vocab large enough for our toy ids
        v = int(input_ids.max().item()) + 5
        logits = torch.zeros(b, t, v)
        for i in range(t - 1):
            tgt = int(input_ids[0, i + 1].item())
            logits[0, i, tgt] = 10.0  # strongly prefers the true next token
        return SimpleNamespace(logits=logits)


def test_continuation_all_pre_predicted() -> None:
    tokens = list(range(64))
    out = continuation_ce_and_greedy(_ToyModel(), tokens, device="cpu")
    assert out["n_pre_predicted"] == 32
    assert out["n_novel"] == 0
    assert out["mean_ce"] < 0.1
    assert all(out["pre_predicted"])
