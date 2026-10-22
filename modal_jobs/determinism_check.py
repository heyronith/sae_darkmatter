"""Modal GPU determinism check for Pythia-70m (eager / same-seed / SDPA)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modal_jobs.common import REPO_ROOT, SECRETS, VOLUMES, app, local_git_info  # noqa: E402

DEFAULT_GPU = "A100-80GB"
MODEL = "EleutherAI/pythia-70m-deduped"
REVISION = "step143000"
PROMPTS = [
    "The capital of France is",
    "In the beginning,",
    "def fibonacci(n):",
    "Once upon a time",
    "The quick brown fox",
    "import numpy as np",
    "To be or not to be",
    "Machine learning models",
]


@app.function(
    gpu=DEFAULT_GPU,
    timeout=900,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def determinism_remote(seed: int = 0) -> dict:
    """Within-process and cross-call-ready logits for 8 fixed prompts."""
    import torch

    from saedm.models import load_model
    from saedm.seeding import set_seed

    def _run(attn: str) -> tuple[list[torch.Tensor], dict]:
        set_seed(seed)
        model, tokenizer, info = load_model(
            MODEL,
            revision=REVISION,
            dtype="float32",
            device="cuda",
            attn_implementation=attn,
        )
        outs: list[torch.Tensor] = []
        with torch.no_grad():
            for prompt in PROMPTS:
                ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
                # Last-position logits only (enough for equality / max-abs checks).
                outs.append(model(input_ids=ids).logits[0, -1].detach().float().cpu())
        return outs, info

    eager_a, info = _run("eager")
    eager_b, _ = _run("eager")
    sdpa, _ = _run("sdpa")

    within_equal = all(torch.equal(a, b) for a, b in zip(eager_a, eager_b, strict=True))
    within_max_abs = max(
        float((a - b).abs().max()) for a, b in zip(eager_a, eager_b, strict=True)
    )
    eager_vs_sdpa_max_abs = max(
        float((a - b).abs().max()) for a, b in zip(eager_a, sdpa, strict=True)
    )

    return {
        "within_process_bitwise_identical": within_equal,
        "within_process_max_abs_diff": within_max_abs,
        "eager_vs_sdpa_max_abs_diff": eager_vs_sdpa_max_abs,
        # Contiguous float32 lists for cross-call max-abs on the local entrypoint.
        "eager_last_logits": [t.contiguous().tolist() for t in eager_a],
        "prompts": PROMPTS,
        "model_info": info,
        "device": {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "index": 0,
        },
        "precision": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, seed: int = 0) -> None:
    started_at = datetime.now(UTC)
    git_commit, git_dirty = local_git_info()

    print(f"Determinism call 1 on {gpu} (commit={git_commit[:12]})")
    r1 = determinism_remote.with_options(gpu=gpu).remote(seed=seed)
    print(f"Determinism call 2 on {gpu}")
    r2 = determinism_remote.with_options(gpu=gpu).remote(seed=seed)

    import torch

    cross_diffs = []
    for a, b in zip(r1["eager_last_logits"], r2["eager_last_logits"], strict=True):
        ta = torch.tensor(a, dtype=torch.float32)
        tb = torch.tensor(b, dtype=torch.float32)
        cross_diffs.append(float((ta - tb).abs().max()))
    cross_call_max_abs_diff = max(cross_diffs) if cross_diffs else float("nan")
    cross_call_bitwise_identical = all(d == 0.0 for d in cross_diffs)

    payload = {
        "within_process_bitwise_identical": r1["within_process_bitwise_identical"],
        "within_process_max_abs_diff": r1["within_process_max_abs_diff"],
        "cross_call_bitwise_identical": cross_call_bitwise_identical,
        "cross_call_max_abs_diff": cross_call_max_abs_diff,
        "eager_vs_sdpa_max_abs_diff": r1["eager_vs_sdpa_max_abs_diff"],
        "prompts": PROMPTS,
        "call1": {
            k: r1[k]
            for k in ("device", "precision", "model_info", "cuda_version", "torch_version")
        },
        "call2": {
            k: r2[k]
            for k in ("device", "precision", "model_info", "cuda_version", "torch_version")
        },
        # Do not persist full vocab logits (large); diffs above are the artifact.
    }

    out_dir = REPO_ROOT / "results" / "P0.4b"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "determinism.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    from saedm.manifest import write_manifest

    ended_at = datetime.now(UTC)
    write_manifest(
        task_id="P0.4b",
        config={
            "model": MODEL,
            "revision": REVISION,
            "dtype": "float32",
            "attn_implementation": "eager",
            "seed": seed,
            "gpu": gpu,
            "task_id": "P0.4b",
            "n_prompts": len(PROMPTS),
        },
        seed=seed,
        out_dir=out_dir,
        outputs=[str(out_path.relative_to(REPO_ROOT))],
        started_at=started_at,
        ended_at=ended_at,
        model_info=r1["model_info"],
        git_commit=git_commit,
        git_dirty=git_dirty,
        device=r1["device"],
        precision=r1["precision"],
    )

    print(f"within_process_bitwise_identical={r1['within_process_bitwise_identical']}")
    print(f"within_process_max_abs_diff={r1['within_process_max_abs_diff']}")
    print(f"cross_call_bitwise_identical={cross_call_bitwise_identical}")
    print(f"cross_call_max_abs_diff={cross_call_max_abs_diff}")
    print(f"eager_vs_sdpa_max_abs_diff={r1['eager_vs_sdpa_max_abs_diff']}")
    print(f"Wrote {out_path}")
