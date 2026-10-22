"""Tests for saedm.manifest."""

from pathlib import Path

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
    )
    import json

    data = json.loads(path.read_text())
    assert REQUIRED_FIELDS <= set(data.keys())
    assert set(data["timestamp_utc"].keys()) >= {"start", "end"}
    assert "python" in data["versions"]
    assert "torch" in data["versions"]
    assert "transformers" in data["versions"]
    assert data["device"]["type"] in {"cpu", "cuda"}
    assert isinstance(data["git_dirty"], bool)
    assert isinstance(data["outputs"], list)


def test_git_commit_hash_in_repo(tmp_path: Path) -> None:
    path = write_manifest(
        task_id="P0.1",
        config={"x": 1},
        seed=0,
        out_dir=tmp_path,
    )
    import json

    data = json.loads(path.read_text())
    commit = data["git_commit"]
    if commit == "unknown":
        pytest.skip("not inside a git repository with commits")
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)
