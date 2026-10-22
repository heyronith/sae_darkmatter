"""Shared model / tokenizer loading with explicit dtype and revision."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _resolve_dtype(dtype: str | torch.dtype) -> tuple[str, torch.dtype]:
    aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    if isinstance(dtype, torch.dtype):
        mapping = {
            torch.float32: "float32",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported torch dtype: {dtype}")
        return mapping[dtype], dtype
    key = aliases.get(dtype.lower(), dtype.lower())
    if key not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"Unknown dtype {dtype!r}; expected float32/float16/bfloat16")
    return key, _DTYPE_MAP[key]


def load_model(
    name: str,
    revision: str,
    dtype: str = "float32",
    device: str = "cpu",
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a causal LM and tokenizer with an explicit dtype and revision.

    revision is required (no default). After load, every parameter must match
    the requested dtype or an AssertionError is raised.
    """
    dtype_name, torch_dtype = _resolve_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, dtype=dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        dtype=torch_dtype,
    )
    model.to(device=device)
    model.eval()

    bad = [
        (n, tuple(p.shape), str(p.dtype))
        for n, p in model.named_parameters()
        if p.dtype != torch_dtype
    ]
    if bad:
        raise AssertionError(
            f"Requested dtype {dtype_name} ({torch_dtype}), but found mismatched params: "
            f"{bad[:5]}{'...' if len(bad) > 5 else ''}"
        )

    config = model.config
    commit = getattr(config, "_commit_hash", None)
    if not commit:
        try:
            from huggingface_hub import repo_info as hf_repo_info
            from huggingface_hub.errors import HfHubHTTPError

            commit = hf_repo_info(name, revision=revision).sha
        except (OSError, ValueError, HfHubHTTPError):
            commit = "unknown"
    info = {
        "name": name,
        "revision": revision,
        "resolved_commit_sha": commit or "unknown",
        "dtype": dtype_name,
        "device": device,
        "n_layers": int(config.num_hidden_layers),
        "d_model": int(config.hidden_size),
    }
    return model, tokenizer, info
