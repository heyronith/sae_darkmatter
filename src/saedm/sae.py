"""Unified SAE interface (float32 encode / decode / reconstruct)."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor, nn

ActivationName = Literal["relu", "topk", "jumprelu"]

_CANONICAL_KEYS = ("W_enc", "b_enc", "W_dec", "b_dec")

# Maps our parameter name -> (alias keys, transpose_if_linear_weight)
_ALIASES: dict[str, list[tuple[str, bool]]] = {
    "W_enc": [
        ("W_enc", False),
        ("w_enc", False),
        ("encoder.weight", True),
        ("encoder.W", False),
    ],
    "b_enc": [
        ("b_enc", False),
        ("encoder.bias", False),
        ("encoder.b", False),
    ],
    "W_dec": [
        ("W_dec", False),
        ("w_dec", False),
        ("decoder.weight", True),
        ("decoder.W", False),
    ],
    "b_dec": [
        ("b_dec", False),
        ("decoder.bias", False),
        ("decoder.b", False),
        ("bias_dec", False),
    ],
}


class SAE(nn.Module):
    """Sparse autoencoder over residual-stream activations."""

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        activation: ActivationName = "relu",
        *,
        k: int | None = None,
        threshold: Tensor | None = None,
    ) -> None:
        super().__init__()
        if activation not in {"relu", "topk", "jumprelu"}:
            raise ValueError(f"Unknown activation {activation!r}")
        if activation == "topk" and (k is None or k <= 0):
            raise ValueError("topk activation requires positive k")
        if activation == "jumprelu" and threshold is None:
            raise ValueError("jumprelu activation requires a threshold vector")

        self.d_model = d_model
        self.d_sae = d_sae
        self.activation = activation
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae, dtype=torch.float32))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=torch.float32))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model, dtype=torch.float32))
        self.b_dec = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

        if threshold is not None:
            if tuple(threshold.shape) != (d_sae,):
                raise ValueError(
                    f"threshold must have shape ({d_sae},), got {tuple(threshold.shape)}"
                )
            self.register_buffer("threshold", threshold.detach().to(dtype=torch.float32))
        else:
            self.register_buffer("threshold", torch.empty(0, dtype=torch.float32))

    def _apply_activation(self, pre: Tensor) -> Tensor:
        if self.activation == "relu":
            return torch.relu(pre)
        if self.activation == "topk":
            assert self.k is not None
            values, indices = torch.topk(pre, self.k, dim=-1)
            out = torch.zeros_like(pre)
            out.scatter_(-1, indices, values)
            return out
        # jumprelu
        return pre * (pre > self.threshold).to(dtype=pre.dtype)

    def encode(self, x: Tensor) -> Tensor:
        x32 = x.to(dtype=torch.float32)
        pre = (x32 - self.b_dec) @ self.W_enc + self.b_enc
        return self._apply_activation(pre)

    def decode(self, z: Tensor) -> Tensor:
        z32 = z.to(dtype=torch.float32)
        return z32 @ self.W_dec + self.b_dec

    def reconstruct(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x32 = x.to(dtype=torch.float32)
        x_hat = self.decode(self.encode(x32))
        err = x32 - x_hat
        return x_hat, err

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.reconstruct(x)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        activation: ActivationName,
        **kwargs: Any,
    ) -> SAE:
        """Build an SAE from a state dict, accepting common alternate key names."""
        flat = {k: v for k, v in state_dict.items()}
        resolved: dict[str, Tensor] = {}
        missing: list[str] = []

        for canon in _CANONICAL_KEYS:
            found = False
            for alias, transpose in _ALIASES[canon]:
                if alias in flat:
                    tensor = flat[alias]
                    if not isinstance(tensor, Tensor):
                        tensor = torch.as_tensor(tensor)
                    if transpose:
                        tensor = tensor.T.contiguous()
                    resolved[canon] = tensor.to(dtype=torch.float32)
                    found = True
                    break
            if not found:
                missing.append(canon)

        if missing:
            available = sorted(flat.keys())
            raise KeyError(
                f"Could not map SAE parameters {missing}. "
                f"Available keys: {available}. "
                f"Accepted aliases: "
                + "; ".join(
                    f"{k} <- {[a for a, _ in _ALIASES[k]]}" for k in missing
                )
            )

        d_model, d_sae = resolved["W_enc"].shape
        if resolved["W_dec"].shape != (d_sae, d_model):
            raise ValueError(
                f"W_dec shape {tuple(resolved['W_dec'].shape)} incompatible with "
                f"W_enc shape {tuple(resolved['W_enc'].shape)}"
            )
        if resolved["b_enc"].shape != (d_sae,):
            raise ValueError(f"b_enc shape {tuple(resolved['b_enc'].shape)} != ({d_sae},)")
        if resolved["b_dec"].shape != (d_model,):
            raise ValueError(f"b_dec shape {tuple(resolved['b_dec'].shape)} != ({d_model},)")

        sae = cls(d_model=d_model, d_sae=d_sae, activation=activation, **kwargs)
        with torch.no_grad():
            sae.W_enc.copy_(resolved["W_enc"])
            sae.b_enc.copy_(resolved["b_enc"])
            sae.W_dec.copy_(resolved["W_dec"])
            sae.b_dec.copy_(resolved["b_dec"])
        return sae
