"""Experiment provenance manifests."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from saedm.config import FrozenDict
from saedm.paths import results_dir


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=results_dir().parent,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=results_dir().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _device_info() -> dict[str, Any]:
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        return {
            "type": "cuda",
            "name": torch.cuda.get_device_name(idx),
            "index": idx,
        }
    return {"type": "cpu", "name": None}


def write_manifest(
    task_id: str,
    config: Mapping[str, Any] | FrozenDict,
    seed: int,
    out_dir: str | Path | None = None,
    *,
    outputs: list[str] | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
) -> Path:
    """Write results/<task_id>/manifest.json and return its path."""
    task_dir = Path(out_dir) if out_dir is not None else results_dir() / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(config, FrozenDict):
        config_payload = config.to_dict()
    else:
        config_payload = dict(config)

    start = started_at or datetime.now(UTC)
    end = ended_at or datetime.now(UTC)

    manifest = {
        "task_id": task_id,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "timestamp_utc": {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
        },
        "seed": seed,
        "config": config_payload,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "hostname": platform.node(),
        "device": _device_info(),
        "outputs": list(outputs or []),
    }

    path = task_dir / "manifest.json"
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return path
