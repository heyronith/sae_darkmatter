"""Tests for saedm.manifest."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from saedm.config import FrozenDict
from saedm.manifest import write_manifest

REQUIRED_FIELDS = {
    "task_id",
    "git_commit",
    "git_dirty",
    "timestamp_utc",
    "seed",
    "config",
    "versions",
    "hostname",
    "device",
    "precision",
    "outputs",
}


def test_manifest_required_fields(tmp_path: Path) -> None:
    cfg = FrozenDict({"foo": "bar"})
    path = write_manifest(
        task_id="P0.1",
        config=cfg,
        seed=0,
        out_dir=tmp_path,
        outputs=["results/P0.1/smoke.json"],
        git_commit="a" * 40,
        git_dirty=False,
    )
    data = json.loads(path.read_text())
    assert REQUIRED_FIELDS <= set(data.keys())
    assert set(data["timestamp_utc"].keys()) >= {"start", "end"}
    assert "python" in data["versions"]
    assert "torch" in data["versions"]
    assert "transformers" in data["versions"]
    assert data["device"]["type"] in {"cpu", "cuda"}
    assert isinstance(data["git_dirty"], bool)
    assert isinstance(data["outputs"], list)
    assert "cuda_matmul_allow_tf32" in data["precision"]
    assert "cudnn_allow_tf32" in data["precision"]
    assert "float32_matmul_precision" in data["precision"]


def test_git_commit_hash_in_repo(tmp_path: Path) -> None:
    path = write_manifest(
        task_id="P0.1",
        config={"x": 1},
        seed=0,
        out_dir=tmp_path,
        allow_dirty=True,
    )
    data = json.loads(path.read_text())
    commit = data["git_commit"]
    if commit == "unknown":
        pytest.skip("not inside a git repository with commits")
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)


def test_refuses_dirty_tree(tmp_path: Path) -> None:
    with mock.patch("saedm.manifest._git_dirty", return_value=True):
        with mock.patch("saedm.manifest._git_commit", return_value="b" * 40):
            with pytest.raises(RuntimeError, match="uncommitted code"):
                write_manifest(
                    task_id="P0.1",
                    config={"x": 1},
                    seed=0,
                    out_dir=tmp_path,
                    allow_dirty=False,
                )


def test_explicit_git_overrides(tmp_path: Path) -> None:
    path = write_manifest(
        task_id="P0.1",
        config={"x": 1},
        seed=0,
        out_dir=tmp_path,
        git_commit="c" * 40,
        git_dirty=True,
        allow_dirty=True,
    )
    data = json.loads(path.read_text())
    assert data["git_commit"] == "c" * 40
    assert data["git_dirty"] is True
