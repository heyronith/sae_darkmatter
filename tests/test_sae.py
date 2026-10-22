"""Tests for the unified SAE interface (CPU, no network)."""

from __future__ import annotations

import torch
from transformers import GPTNeoXForCausalLM

from saedm.hooks import splice
from saedm.sae import SAE


def test_topk_at_most_k_nonzeros_and_nonnegative() -> None:
    sae = SAE(d_model=16, d_sae=32, activation="topk", k=3)
    x = torch.randn(4, 8, 16)
    z = sae.encode(x)
    nnz = (z != 0).sum(dim=-1)
    assert torch.all(nnz <= 3)
    assert torch.all(z >= 0)


def test_topk_all_negative_preacts_are_zero() -> None:
    sae = SAE(d_model=4, d_sae=8, activation="topk", k=3)
    with torch.no_grad():
        sae.W_enc.zero_()
        sae.b_enc.fill_(-1.0)
        sae.b_dec.zero_()
    x = torch.zeros(2, 3, 4)
    z = sae.encode(x)
    assert torch.equal(z, torch.zeros_like(z))


def test_jumprelu_threshold() -> None:
    threshold = torch.tensor([0.5, 1.0, 2.0])
    sae = SAE(d_model=3, d_sae=3, activation="jumprelu", threshold=threshold)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(3))
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    # pre = x (identity encode without bias)
    x = torch.tensor([[0.5, 1.5, 3.0]])  # at, above, above threshold
    z = sae.encode(x)
    # 0.5 is at threshold -> zero; 1.5 > 1.0 -> 1.5; 3.0 > 2.0 -> 3.0
    assert torch.equal(z, torch.tensor([[0.0, 1.5, 3.0]]))


def test_apply_b_dec_to_input_false_changes_encoding() -> None:
    sae_on = SAE(d_model=8, d_sae=16, activation="relu", apply_b_dec_to_input=True)
    sae_off = SAE(d_model=8, d_sae=16, activation="relu", apply_b_dec_to_input=False)
    with torch.no_grad():
        sae_off.W_enc.copy_(sae_on.W_enc)
        sae_off.b_enc.copy_(sae_on.b_enc)
        sae_off.W_dec.copy_(sae_on.W_dec)
        sae_off.b_dec.copy_(sae_on.b_dec)
        sae_on.b_dec.fill_(0.5)
        sae_off.b_dec.fill_(0.5)
    x = torch.randn(2, 4, 8)
    z_on = sae_on.encode(x)
    z_off = sae_off.encode(x)
    assert not torch.allclose(z_on, z_off)


def test_relu_never_negative() -> None:
    sae = SAE(d_model=16, d_sae=32, activation="relu")
    x = torch.randn(4, 8, 16)
    z = sae.encode(x)
    assert torch.all(z >= 0)


def test_reconstruct_recovers_x() -> None:
    sae = SAE(d_model=16, d_sae=32, activation="relu")
    x = torch.randn(4, 8, 16)
    x_hat, err = sae.reconstruct(x)
    assert torch.allclose(x_hat + err, x.to(dtype=torch.float32), atol=1e-6)


def test_from_state_dict_roundtrip() -> None:
    sae = SAE(d_model=16, d_sae=24, activation="relu", apply_b_dec_to_input=False)
    sd = {
        "W_enc": sae.W_enc.detach().clone(),
        "b_enc": sae.b_enc.detach().clone(),
        "W_dec": sae.W_dec.detach().clone(),
        "b_dec": sae.b_dec.detach().clone(),
    }
    loaded = SAE.from_state_dict(sd, activation="relu", apply_b_dec_to_input=False)
    assert loaded.apply_b_dec_to_input is False
    x = torch.randn(2, 5, 16)
    z1 = sae.encode(x)
    z2 = loaded.encode(x)
    assert torch.allclose(z1, z2, atol=1e-6)

    # Alternate Linear-style keys (transposed weights).
    alt = {
        "encoder.weight": sae.W_enc.T.contiguous(),
        "encoder.bias": sae.b_enc.detach().clone(),
        "decoder.weight": sae.W_dec.T.contiguous(),
        "decoder.bias": sae.b_dec.detach().clone(),
    }
    loaded_alt = SAE.from_state_dict(alt, activation="relu", apply_b_dec_to_input=False)
    assert torch.allclose(sae.encode(x), loaded_alt.encode(x), atol=1e-6)


def test_splice_reconstruction_plus_error_reproduces_logits(
    tiny_gpt_neox: GPTNeoXForCausalLM,
    tiny_input_ids: torch.Tensor,
) -> None:
    model = tiny_gpt_neox
    d_model = model.config.hidden_size
    sae = SAE(d_model=d_model, d_sae=64, activation="relu")

    with torch.no_grad():
        baseline = model(input_ids=tiny_input_ids).logits

        def restore(act: torch.Tensor) -> torch.Tensor:
            x_hat, err = sae.reconstruct(act)
            return x_hat + err

        spliced = splice(model, tiny_input_ids, "resid_post.0", restore)

    assert torch.allclose(baseline, spliced, atol=1e-5)
