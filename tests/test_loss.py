"""Unit tests for continuation CE / greedy helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from saedm.loss import continuation_ce_and_greedy


class _ToyModel:
    """Logits that prefer next = last+1 (matches ``list(range(64))`` windows)."""

    def __call__(self, input_ids: torch.Tensor) -> SimpleNamespace:
        b, t = input_ids.shape
        v = max(int(input_ids.max().item()) + 5, 70)
        logits = torch.zeros(b, t, v)
        for bi in range(b):
            for i in range(t):
                nxt = int(input_ids[bi, i].item()) + 1
                if nxt < v:
                    logits[bi, i, nxt] = 10.0
        return SimpleNamespace(logits=logits)


def test_continuation_all_pre_predicted() -> None:
    tokens = list(range(64))
    out = continuation_ce_and_greedy(_ToyModel(), tokens, device="cpu")
    assert out["n_pre_predicted"] == 32
    assert out["n_novel"] == 0
    assert out["mean_ce"] < 0.1
    assert all(out["pre_predicted"])


def test_greedy_exact_match_on_toy() -> None:
    from saedm.loss import greedy_continuation_exact_match

    tokens = list(range(64))
    # Toy model always emits the true next token under teacher forcing of the
    # growing prefix when the prefix is the true sequence — AR from prompt
    # also recovers the continuation because each next id = prompt_len+k.
    out = greedy_continuation_exact_match(_ToyModel(), tokens, device="cpu")
    assert out["exact_match"] is True
    assert out["n_match"] == 32


def test_greedy_batch_matches_single() -> None:
    from saedm.loss import (
        greedy_continuation_exact_match,
        greedy_continuation_exact_match_batch,
    )

    batch = [list(range(64)), list(range(10, 74)), list(range(5, 69))]
    model = _ToyModel()
    batched = greedy_continuation_exact_match_batch(model, batch, device="cpu")
    for tokens, got in zip(batch, batched, strict=True):
        single = greedy_continuation_exact_match(model, tokens, device="cpu")
        assert got == single
