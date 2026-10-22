"""Shared fixtures for CPU-only unit tests (no network)."""

from __future__ import annotations

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM


@pytest.fixture
def tiny_gpt_neox() -> GPTNeoXForCausalLM:
    """2-layer random GPT-NeoX with d_model=32, vocab=100."""
    config = GPTNeoXConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        rotary_pct=0.25,
        use_parallel_residual=True,
    )
    model = GPTNeoXForCausalLM(config)
    model.eval()
    return model


@pytest.fixture
def tiny_input_ids() -> torch.Tensor:
    return torch.randint(0, 100, (2, 8), dtype=torch.long)
