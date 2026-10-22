"""Residual-stream capture and splice hooks for GPT-NeoX (Pythia)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn


def _parse_hookpoint(hookpoint: str) -> tuple[str, int]:
    kind, sep, rest = hookpoint.partition(".")
    if sep != "." or kind not in {"resid_pre", "resid_post"} or not rest.isdigit():
        raise ValueError(
            f"Invalid hookpoint {hookpoint!r}; expected 'resid_pre.L' or 'resid_post.L'"
        )
    return kind, int(rest)


def _layers(model: nn.Module) -> nn.ModuleList:
    if not hasattr(model, "gpt_neox") or not hasattr(model.gpt_neox, "layers"):
        raise TypeError("Expected a GPT-NeoX / Pythia model with model.gpt_neox.layers")
    return model.gpt_neox.layers


def _hidden_from_output(output: Any) -> Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _replace_output(output: Any, new_hidden: Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden


def _read_pre_hidden(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Tensor:
    if args:
        return args[0]
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    raise KeyError("hidden_states not found in pre-hook args or kwargs")


def _write_pre_hidden(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    new_hidden: Tensor,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if args:
        return (new_hidden,) + args[1:], kwargs
    new_kwargs = dict(kwargs)
    new_kwargs["hidden_states"] = new_hidden
    return args, new_kwargs


@contextmanager
def _registered_hooks(handles: list[Any]) -> Iterator[None]:
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()


def capture(
    model: nn.Module,
    input_ids: Tensor,
    hookpoints: list[str],
) -> dict[str, Tensor]:
    """Run a forward pass and return detached residual activations.

    Hookpoints: ``resid_pre.L`` (block input) and ``resid_post.L`` (block output).
    Does not use ``output_hidden_states`` (final LN would pollute the last entry).
    """
    layers = _layers(model)
    captured: dict[str, Tensor] = {}
    handles: list[Any] = []

    for hp in hookpoints:
        kind, layer_idx = _parse_hookpoint(hp)
        if layer_idx < 0 or layer_idx >= len(layers):
            raise IndexError(f"Layer {layer_idx} out of range for {len(layers)} layers")
        layer = layers[layer_idx]

        if kind == "resid_pre":

            def make_pre(name: str = hp) -> Callable:
                def pre_hook(
                    _module: nn.Module,
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                ) -> None:
                    captured[name] = _read_pre_hidden(args, kwargs).detach()

                return pre_hook

            handles.append(layer.register_forward_pre_hook(make_pre(), with_kwargs=True))
        else:

            def make_post(name: str = hp) -> Callable:
                def fwd_hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
                    captured[name] = _hidden_from_output(output).detach()

                return fwd_hook

            handles.append(layer.register_forward_hook(make_post()))

    with _registered_hooks(handles):
        with torch.no_grad():
            model(input_ids=input_ids)

    missing = [hp for hp in hookpoints if hp not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture hookpoints: {missing}")
    return {hp: captured[hp] for hp in hookpoints}


def splice(
    model: nn.Module,
    input_ids: Tensor,
    hookpoint: str,
    fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """Forward pass with ``fn`` replacing the activation at ``hookpoint``.

    Returns logits. Hooks are always removed, including if ``fn`` raises.
    """
    layers = _layers(model)
    kind, layer_idx = _parse_hookpoint(hookpoint)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"Layer {layer_idx} out of range for {len(layers)} layers")
    layer = layers[layer_idx]
    handles: list[Any] = []

    if kind == "resid_pre":

        def pre_hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            new_act = fn(_read_pre_hidden(args, kwargs))
            return _write_pre_hidden(args, kwargs, new_act)

        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
    else:

        def fwd_hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
            hidden = _hidden_from_output(output)
            return _replace_output(output, fn(hidden))

        handles.append(layer.register_forward_hook(fwd_hook))

    with _registered_hooks(handles):
        outputs = model(input_ids=input_ids)
    return outputs.logits
