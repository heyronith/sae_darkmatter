"""Tests for residual-stream hooks (tiny random GPT-NeoX, CPU, no network)."""

from __future__ import annotations

import torch
from transformers import GPTNeoXForCausalLM
from transformers.masking_utils import create_causal_mask

from saedm.hooks import _read_pre_hidden, capture, splice


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


def test_capture_hidden_states_kwarg(tiny_gpt_neox: GPTNeoXForCausalLM) -> None:
    """Pre-hooks must read hidden state from kwargs when not passed positionally."""
    model = tiny_gpt_neox
    layer = model.gpt_neox.layers[0]
    d_model = model.config.hidden_size
    batch, seq = 1, 4
    hidden_states = torch.randn(batch, seq, d_model)
    position_ids = torch.arange(seq).unsqueeze(0)
    position_embeddings = model.gpt_neox.rotary_emb(hidden_states, position_ids=position_ids)
    causal_mask = create_causal_mask(
        config=model.config,
        inputs_embeds=hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )

    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module, args, kwargs):
        captured["hs"] = _read_pre_hidden(args, kwargs).detach()

    handle = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
    finally:
        handle.remove()

    assert "hs" in captured
    assert torch.equal(captured["hs"], hidden_states)


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
