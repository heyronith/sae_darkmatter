#!/usr/bin/env python3
"""CPU smoke test: load Pythia-70m and run one forward pass."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saedm.config import load_config  # noqa: E402
from saedm.manifest import write_manifest  # noqa: E402
from saedm.models import load_model  # noqa: E402
from saedm.paths import results_dir  # noqa: E402
from saedm.seeding import set_seed  # noqa: E402


def main() -> None:
    started_at = datetime.now(UTC)
    config_path = REPO_ROOT / "configs" / "smoke.yaml"
    overrides = sys.argv[1:]
    cfg = load_config(config_path, overrides=overrides or None)

    seed = int(cfg["seed"])
    task_id = str(cfg["task_id"])
    model_name = str(cfg["model"])
    revision = str(cfg["revision"])
    dtype = str(cfg["dtype"])
    device = str(cfg["device"]) if "device" in cfg else "cpu"
    prompt = str(cfg["prompt"])

    set_seed(seed)

    model, tokenizer, info = load_model(
        model_name,
        revision=revision,
        dtype=dtype,
        device=device,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    next_logits = logits[0, -1].float()
    probs = torch.softmax(next_logits, dim=-1)
    topk = torch.topk(probs, k=5)

    top5 = []
    for token_id, prob in zip(topk.indices.tolist(), topk.values.tolist(), strict=True):
        token_str = tokenizer.decode([token_id])
        top5.append({"token": token_str, "token_id": token_id, "probability": float(prob)})
        print(f"{token_str!r}: {prob:.6f}")

    out_dir = results_dir() / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = "smoke_fp16.json" if dtype in {"float16", "fp16"} else "smoke.json"
    smoke_path = out_dir / out_name

    smoke_payload = {
        "input_ids": input_ids[0].tolist(),
        "logits_shape": list(logits.shape),
        "dtype": dtype,
        "top5_next_tokens": top5,
        "model_info": info,
    }
    with smoke_path.open("w") as f:
        json.dump(smoke_payload, f, indent=2)
        f.write("\n")

    # Primary (float32) run owns the manifest; include both smoke files when present.
    if dtype in {"float32", "fp32"}:
        outputs: list[str] = []
        for name in ("smoke.json", "smoke_fp16.json"):
            path = out_dir / name
            if path.exists():
                outputs.append(str(path.relative_to(REPO_ROOT)))
        ended_at = datetime.now(UTC)
        write_manifest(
            task_id=task_id,
            config=cfg,
            seed=seed,
            out_dir=out_dir,
            outputs=outputs,
            started_at=started_at,
            ended_at=ended_at,
            model_info=info,
        )
        print(f"Wrote {out_dir / 'manifest.json'}")
    else:
        # Keep float32 manifest model_info; just ensure this file is listed if manifest exists.
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            rel = str(smoke_path.relative_to(REPO_ROOT))
            outs = list(data.get("outputs", []))
            if rel not in outs:
                outs.append(rel)
                data["outputs"] = outs
                with manifest_path.open("w") as f:
                    json.dump(data, f, indent=2)
                    f.write("\n")

    print(f"Wrote {smoke_path}")
    print(f"model_info: {info}")


if __name__ == "__main__":
    main()
