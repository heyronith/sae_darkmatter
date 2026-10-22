"""Slow tests that download real models."""

from __future__ import annotations

import pytest
import torch

from saedm.models import load_model


@pytest.mark.slow
def test_pythia70m_float32_params() -> None:
    model, _tokenizer, info = load_model(
        "EleutherAI/pythia-70m-deduped",
        revision="step143000",
        dtype="float32",
        device="cpu",
    )
    assert info["dtype"] == "float32"
    assert info["revision"] == "step143000"
    assert info["resolved_commit_sha"] != "unknown"
    for name, param in model.named_parameters():
        assert param.dtype == torch.float32, f"{name} has dtype {param.dtype}"
