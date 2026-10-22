"""Tests for the unified SAE interface (CPU, no network)."""

from __future__ import annotations

import torch
from transformers import GPTNeoXForCausalLM

from saedm.hooks import splice
from saedm.sae import SAE


def test_topk_exactly_k_nonzeros() -> None:
    sae = SAE(d_model=16, d_sae=32, activation="topk", k=3)
    x = torch.randn(4, 8, 16)
    z = sae.encode(x)
    nnz = (z != 0).sum(dim=-1)
    assert torch.equal(nnz, torch.full_like(nnz, 3))


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
    sae = SAE(d_model=16, d_sae=24, activation="relu")
    sd = {
        "W_enc": sae.W_enc.detach().clone(),
        "b_enc": sae.b_enc.detach().clone(),
        "W_dec": sae.W_dec.detach().clone(),
        "b_dec": sae.b_dec.detach().clone(),
    }
    loaded = SAE.from_state_dict(sd, activation="relu")
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
    loaded_alt = SAE.from_state_dict(alt, activation="relu")
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
