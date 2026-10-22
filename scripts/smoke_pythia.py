#!/usr/bin/env python3
"""CPU smoke test: load Pythia-70m and run one forward pass."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running without install when developing.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saedm.config import load_config  # noqa: E402
from saedm.manifest import write_manifest  # noqa: E402
from saedm.paths import results_dir  # noqa: E402
from saedm.seeding import set_seed  # noqa: E402


def main() -> None:
    started_at = datetime.now(UTC)
    config_path = REPO_ROOT / "configs" / "smoke.yaml"
    cfg = load_config(config_path)

    seed = int(cfg["seed"])
    task_id = str(cfg["task_id"])
    model_name = str(cfg["model"])
    prompt = str(cfg["prompt"])

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    next_logits = logits[0, -1]
    probs = torch.softmax(next_logits, dim=-1)
    topk = torch.topk(probs, k=5)

    top5 = []
    for token_id, prob in zip(topk.indices.tolist(), topk.values.tolist(), strict=True):
        token_str = tokenizer.decode([token_id])
        top5.append({"token": token_str, "token_id": token_id, "probability": float(prob)})
        print(f"{token_str!r}: {prob:.6f}")

    out_dir = results_dir() / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = out_dir / "smoke.json"

    smoke_payload = {
        "input_ids": input_ids[0].tolist(),
        "logits_shape": list(logits.shape),
        "top5_next_tokens": top5,
    }
    with smoke_path.open("w") as f:
        json.dump(smoke_payload, f, indent=2)
        f.write("\n")

    ended_at = datetime.now(UTC)
    write_manifest(
        task_id=task_id,
        config=cfg,
        seed=seed,
        out_dir=out_dir,
        outputs=[str(smoke_path.relative_to(REPO_ROOT))],
        started_at=started_at,
        ended_at=ended_at,
    )
    print(f"Wrote {smoke_path}")
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
