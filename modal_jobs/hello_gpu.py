"""Modal GPU smoke: Pythia-70m fp32 on CUDA vs local CPU fp32 baseline."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modal_jobs.common import REPO_ROOT, SECRETS, VOLUMES, app, local_git_info  # noqa: E402

DEFAULT_GPU = "A100-80GB"
PROMPT = "The capital of France is"
MODEL = "EleutherAI/pythia-70m-deduped"
REVISION = "step143000"


@app.function(
    gpu=DEFAULT_GPU,
    timeout=600,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def gpu_smoke(seed: int = 0) -> dict:
    """Run nvidia-smi + one Pythia forward pass on GPU; return top-5 and meta."""
    import torch

    from saedm.models import load_model
    from saedm.seeding import set_seed

    smi = subprocess.run(
        ["nvidia-smi"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(smi.stdout)

    t0 = time.perf_counter()
    set_seed(seed)
    model, tokenizer, info = load_model(
        MODEL,
        revision=REVISION,
        dtype="float32",
        device="cuda",
    )
    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    next_logits = logits[0, -1].float()
    probs = torch.softmax(next_logits, dim=-1)
    topk = torch.topk(probs, k=5)
    top5 = []
    for token_id, prob in zip(topk.indices.tolist(), topk.values.tolist(), strict=True):
        top5.append(
            {
                "token": tokenizer.decode([token_id]),
                "token_id": token_id,
                "probability": float(prob),
            }
        )

    wall_time_s = time.perf_counter() - t0
    return {
        "top5_next_tokens": top5,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "wall_time_s": wall_time_s,
        "precision": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "model_info": info,
        "nvidia_smi": smi.stdout,
        "input_ids": input_ids[0].tolist(),
        "logits_shape": list(logits.shape),
    }


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, seed: int = 0) -> None:
    started_at = datetime.now(UTC)
    git_commit, git_dirty = local_git_info()

    print(f"Launching gpu_smoke on {gpu} (commit={git_commit[:12]}, dirty={git_dirty})")
    result = gpu_smoke.with_options(gpu=gpu).remote(seed=seed)

    cpu_path = REPO_ROOT / "results" / "P0.2" / "smoke.json"
    cpu = json.loads(cpu_path.read_text())
    cpu_top5 = cpu["top5_next_tokens"]
    gpu_top5 = result["top5_next_tokens"]

    same_order = [g["token_id"] for g in gpu_top5] == [c["token_id"] for c in cpu_top5]
    abs_diffs = [
        abs(g["probability"] - c["probability"]) for g, c in zip(gpu_top5, cpu_top5, strict=True)
    ]
    max_abs_diff = max(abs_diffs) if abs_diffs else float("nan")

    comparison = {
        "cpu_path": str(cpu_path.relative_to(REPO_ROOT)),
        "same_top5_token_order": same_order,
        "max_abs_probability_diff": max_abs_diff,
        "per_rank_abs_diff": abs_diffs,
        "cpu_top5": cpu_top5,
        "gpu_top5": gpu_top5,
    }

    out_dir = REPO_ROOT / "results" / "P0.4"
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = out_dir / "gpu_smoke.json"
    payload = {
        **result,
        "comparison": comparison,
        "prompt": PROMPT,
        "seed": seed,
        "gpu_requested": gpu,
    }
    # Drop bulky nvidia-smi from the printed summary but keep in file.
    with smoke_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    from saedm.manifest import write_manifest

    ended_at = datetime.now(UTC)
    write_manifest(
        task_id="P0.4",
        config={
            "model": MODEL,
            "revision": REVISION,
            "dtype": "float32",
            "device": "cuda",
            "prompt": PROMPT,
            "seed": seed,
            "gpu": gpu,
            "task_id": "P0.4",
        },
        seed=seed,
        out_dir=out_dir,
        outputs=[
            str(smoke_path.relative_to(REPO_ROOT)),
        ],
        started_at=started_at,
        ended_at=ended_at,
        model_info=result["model_info"],
        git_commit=git_commit,
        git_dirty=git_dirty,
        device={
            "type": "cuda",
            "name": result["gpu_name"],
            "index": 0,
        },
        precision=result["precision"],
    )

    print("GPU top-5:")
    for t in gpu_top5:
        print(f"  {t['token']!r}: {t['probability']:.6f}")
    print("CPU top-5:")
    for t in cpu_top5:
        print(f"  {t['token']!r}: {t['probability']:.6f}")
    print(f"same_order={same_order} max_abs_diff={max_abs_diff:.6g}")
    print(
        f"gpu={result['gpu_name']} cuda={result['cuda_version']} "
        f"wall={result['wall_time_s']:.3f}s"
    )
    print(f"Wrote {smoke_path}")
    print(f"Wrote {out_dir / 'manifest.json'}")
