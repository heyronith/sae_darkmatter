"""Shared Modal app, image, volumes, and local git helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

hf_cache_vol = modal.Volume.from_name("saedm-hf-cache", create_if_missing=True)
artifacts_vol = modal.Volume.from_name("saedm-artifacts", create_if_missing=True)

VOLUMES = {
    "/cache": hf_cache_vol,
    "/artifacts": artifacts_vol,
}

# Optional HF token (Pythia is public). Present in this workspace as "hf-token".
SECRETS = [modal.Secret.from_name("hf-token")]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REPO_ROOT / "modal_jobs" / "requirements.lock"))
    .env({"HF_HOME": "/cache/hf"})
    # Mount package sources onto PYTHONPATH (/root).
    .add_local_dir(str(REPO_ROOT / "src" / "saedm"), remote_path="/root/saedm")
    .add_local_dir(str(REPO_ROOT / "modal_jobs"), remote_path="/root/modal_jobs")
)

app = modal.App("saedm", image=image)


def local_git_info() -> tuple[str, bool]:
    """Return (commit, dirty), ignoring results/. Refuse launches if dirty."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", ".", ":(exclude)results"],
        cwd=REPO_ROOT,
        text=True,
    )
    dirty = bool(status.strip())
    if dirty:
        raise RuntimeError("uncommitted code; commit before launching Modal jobs")
    return commit, dirty
