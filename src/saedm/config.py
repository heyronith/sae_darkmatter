"""YAML config loading with CLI-style dotted overrides."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class FrozenDict(Mapping[str, Any]):
    """Immutable nested mapping."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = {k: _freeze(v) for k, v in data.items()}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def to_dict(self) -> dict[str, Any]:
        return {k: _unfreeze(v) for k, v in self._data.items()}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _unfreeze(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_unfreeze(v) for v in value]
    return value


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _apply_override(cfg: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be key.sub=value, got: {override!r}")
    key_path, raw_value = override.split("=", 1)
    keys = key_path.split(".")
    if not keys or any(not k for k in keys):
        raise ValueError(f"Invalid override key path: {key_path!r}")

    cursor: dict[str, Any] = cfg
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = _parse_override_value(raw_value)


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> FrozenDict:
    """Load a YAML file into a frozen config, applying dotted CLI overrides."""
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")

    for override in overrides or []:
        _apply_override(data, override)

    return FrozenDict(data)


def save_config(config: Mapping[str, Any] | FrozenDict, path: str | Path) -> None:
    """Write a config mapping to YAML."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict() if isinstance(config, FrozenDict) else dict(config)
    with out.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
