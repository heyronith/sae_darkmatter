"""Tests for residual-stream hooks (tiny random GPT-NeoX, CPU, no network)."""

from __future__ import annotations

import torch
from transformers import GPTNeoXForCausalLM

from saedm.hooks import capture, splice


def _hooks_empty(model: GPTNeoXForCausalLM) -> bool:
    for layer in model.gpt_neox.layers:
        if layer._forward_hooks or layer._forward_pre_hooks:
            return False
    return True


def test_resid_post_equals_next_resid_pre(
    tiny_gpt_neox: GPTNeoXForCausalLM,
    tiny_input_ids: torch.Tensor,
) -> None:
    model = tiny_gpt_neox
    acts = capture(
        model,
        tiny_input_ids,
        ["resid_post.0", "resid_pre.1"],
    )
    assert torch.equal(acts["resid_post.0"], acts["resid_pre.1"])


def test_splice_identity_matches_forward(
    tiny_gpt_neox: GPTNeoXForCausalLM,
    tiny_input_ids: torch.Tensor,
) -> None:
    model = tiny_gpt_neox
    with torch.no_grad():
        baseline = model(input_ids=tiny_input_ids).logits
        spliced = splice(model, tiny_input_ids, "resid_post.0", lambda act: act)
    assert torch.equal(baseline, spliced)
    assert _hooks_empty(model)


def test_splice_zeros_changes_logits(
    tiny_gpt_neox: GPTNeoXForCausalLM,
    tiny_input_ids: torch.Tensor,
) -> None:
    model = tiny_gpt_neox
    with torch.no_grad():
        baseline = model(input_ids=tiny_input_ids).logits
        spliced = splice(
            model,
            tiny_input_ids,
            "resid_post.0",
            lambda act: torch.zeros_like(act),
        )
    assert not torch.equal(baseline, spliced)
    assert _hooks_empty(model)


def test_hooks_removed_when_fn_raises(
    tiny_gpt_neox: GPTNeoXForCausalLM,
    tiny_input_ids: torch.Tensor,
) -> None:
    model = tiny_gpt_neox

    def boom(_act: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("intentional splice failure")

    try:
        splice(model, tiny_input_ids, "resid_pre.0", boom)
    except RuntimeError as exc:
        assert "intentional splice failure" in str(exc)
    else:
        raise AssertionError("expected RuntimeError from splice fn")

    assert _hooks_empty(model)
    for layer in model.gpt_neox.layers:
        assert len(layer._forward_hooks) == 0
        assert len(layer._forward_pre_hooks) == 0
