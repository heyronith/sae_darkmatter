"""Tests for saedm.config."""

from pathlib import Path

import pytest
import yaml

from saedm.config import FrozenDict, load_config, save_config


def test_yaml_load(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("a:\n  b: 1\nc: hello\n")
    cfg = load_config(path)
    assert cfg["a"]["b"] == 1
    assert cfg["c"] == "hello"


def test_override_dotted(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("a:\n  b: 1\n")
    cfg = load_config(path, overrides=["a.b=3"])
    assert cfg["a"]["b"] == 3


def test_config_immutable(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("a:\n  b: 1\n")
    cfg = load_config(path)
    with pytest.raises(TypeError):
        cfg["a"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        cfg["a"]["b"] = 9  # type: ignore[index]
    assert isinstance(cfg, FrozenDict)
    assert isinstance(cfg["a"], FrozenDict)


def test_save_config(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("x: 1\n")
    cfg = load_config(path)
    out = tmp_path / "out.yaml"
    save_config(cfg, out)
    loaded = yaml.safe_load(out.read_text())
    assert loaded == {"x": 1}
