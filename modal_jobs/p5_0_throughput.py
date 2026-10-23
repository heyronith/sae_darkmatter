"""P5.0 — 20-step continued-pretraining throughput benchmark (A100)."""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modal_jobs.common import SECRETS, VOLUMES, app, local_git_info  # noqa: E402

DEFAULT_GPU = "A100"
MODEL = "EleutherAI/pythia-410m-deduped"
REVISION = "step140000"  # late checkpoint; leave headroom before 143000
N_STEPS = 20
SEQ_LEN = 2049
MICRO_BATCH = 2  # fits float32 410M + Adam on 40GB with long context
LR = 3.0e-5  # Pythia min_lr / end-of-training LR


@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
    memory=65536,
)
def bench_train(payload: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    t_load0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=payload["revision"],
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.to("cuda")
    model.train()
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(payload["lr"]),
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    load_s = time.perf_counter() - t_load0

    torch.manual_seed(int(payload.get("seed", 0)))
    vocab = int(model.config.vocab_size)
    micro = int(payload["micro_batch"])
    seq_len = int(payload["seq_len"])
    n_steps = int(payload["n_steps"])

    # Warmup step (excluded from timing).
    batch = torch.randint(0, vocab, (micro, seq_len), device="cuda")
    out = model(input_ids=batch, labels=batch)
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        batch = torch.randint(0, vocab, (micro, seq_len), device="cuda")
        out = model(input_ids=batch, labels=batch)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tokens = n_steps * micro * seq_len
    return {
        "elapsed_s": elapsed,
        "load_s": load_s,
        "n_steps": n_steps,
        "micro_batch": micro,
        "seq_len": seq_len,
        "tokens": tokens,
        "tokens_per_s": tokens / elapsed,
        "steps_per_s": n_steps / elapsed,
        "device_name": torch.cuda.get_device_name(0),
        "peak_mem_bytes": int(torch.cuda.max_memory_allocated()),
        "lr": float(payload["lr"]),
        "revision": payload["revision"],
        "dtype": "float32",
        "optimizer": "Adam(betas=(0.9,0.95))",
        "note": (
            "Synthetic random tokens; measures compute throughput only. "
            "Not a fidelity match to NeoX megatron packing."
        ),
    }


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU) -> None:
    started_at = datetime.now(UTC)
    git_commit, git_dirty = local_git_info()

    from saedm.manifest import write_manifest
    from saedm.paths import results_dir

    out_dir = results_dir() / "P5.0"
    out_dir.mkdir(parents=True, exist_ok=True)

    res = bench_train.with_options(gpu=gpu).remote(
        {
            "revision": REVISION,
            "n_steps": N_STEPS,
            "micro_batch": MICRO_BATCH,
            "seq_len": SEQ_LEN,
            "lr": LR,
            "seed": 0,
        }
    )
    summary = {
        "benchmark": res,
        "gpu_requested": gpu,
        "extrapolations": {
            "assumption": (
                f"Scale tokens/s linearly with micro_batch up to effective "
                f"global batch 1024×2049 (Pythia); this bench used micro={MICRO_BATCH}."
            ),
            "tokens_per_s_measured": res["tokens_per_s"],
            "pythia_tokens_per_step": 1024 * SEQ_LEN,
            "estimated_s_per_pythia_step_at_measured_micro": (
                (1024 * SEQ_LEN) / res["tokens_per_s"]
            ),
            "note": (
                "Real NeoX uses 32×A100 with microbatch 32/GPU (global 1024). "
                "Single-GPU float32 Adam here is a lower bound on tokens/s/GPU "
                "for costing; see docs/canary_design.md for budget table."
            ),
        },
        "gpu_hours": res["elapsed_s"] / 3600.0,
    }
    (out_dir / "throughput.json").write_text(json.dumps(summary, indent=2) + "\n")

    ended_at = datetime.now(UTC)
    write_manifest(
        task_id="P5.0",
        config={
            "task_id": "P5.0",
            "n_steps": N_STEPS,
            "micro_batch": MICRO_BATCH,
            "seq_len": SEQ_LEN,
            "revision": REVISION,
        },
        seed=0,
        out_dir=out_dir,
        outputs=["results/P5.0/throughput.json"],
        started_at=started_at,
        ended_at=ended_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
        device={"type": "cuda", "name": res["device_name"], "index": 0},
        model_info={"name": MODEL, "revision": REVISION},
    )
    print(json.dumps(summary["benchmark"], indent=2))
