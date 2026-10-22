"""P1.5 — pre-exposure filter on the recollection stratum (Modal GPU)."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modal_jobs.common import (  # noqa: E402
    REPO_ROOT,
    SECRETS,
    VOLUMES,
    app,
    artifacts_vol,
    local_git_info,
)

DEFAULT_GPU = "A100"
MODEL = "EleutherAI/pythia-410m-deduped"
FINAL_REV = "step143000"
SEED = 0
MIN_NOVEL = 8
G1_MIN_KEPT = 200


@app.function(
    gpu=DEFAULT_GPU,
    timeout=3 * 60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
    memory=65536,
)
def eval_revision_preexposure(payload: dict) -> dict:
    """One checkpoint → per-sequence greedy/CE on the continuation."""
    import torch

    from saedm.loss import continuation_ce_and_greedy
    from saedm.models import load_model
    from saedm.seeding import set_seed

    t0 = time.perf_counter()
    set_seed(int(payload.get("seed", 0)))
    rev = payload["revision"]
    sequences = payload["sequences"]
    model, _tok, _info = load_model(
        MODEL,
        revision=rev,
        dtype="float32",
        device="cuda",
        attn_implementation="eager",
    )
    outs: dict[int, dict] = {}
    with torch.no_grad():
        for seq in sequences:
            out = continuation_ce_and_greedy(model, seq["tokens"], device="cuda")
            outs[int(seq["seq_i"])] = {
                "mean_ce": out["mean_ce"],
                "pre_predicted": out["pre_predicted"],
                "token_logprob": out["token_logprob"],
                "n_pre_predicted": out["n_pre_predicted"],
                "n_novel": out["n_novel"],
            }
    del model
    torch.cuda.empty_cache()
    from modal_jobs.common import hf_cache_vol

    hf_cache_vol.commit()
    return {
        "revision": rev,
        "outs": outs,
        "elapsed_s": time.perf_counter() - t0,
        "device_name": torch.cuda.get_device_name(0),
        "precision": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def _summarize(rows: list[dict], sample_size: int, total: int, min_novel: int) -> dict:
    from saedm.stats import wilson_interval

    n_novel_hist = {str(k): 0 for k in range(0, 33)}
    for r in rows:
        n_novel_hist[str(r["n_novel"])] += 1
    kept = [r for r in rows if r["kept"]]
    kept_k1 = [r for r in kept if r["k_primary"] == 1]
    kept_k25 = [r for r in kept if 2 <= r["k_primary"] <= 5]
    all_k1 = [r for r in rows if r["k_primary"] == 1]
    all_k25 = [r for r in rows if 2 <= r["k_primary"] <= 5]

    def scaled(n: int) -> dict:
        lo, hi = wilson_interval(n, sample_size)
        return {
            "sample_count": n,
            "point": n / sample_size * total,
            "ci95_low": lo * total,
            "ci95_high": hi * total,
        }

    return {
        "n_recollection": len(rows),
        "n_novel_histogram": n_novel_hist,
        "min_novel": min_novel,
        "kept_total": scaled(len(kept)),
        "kept_k_primary_1": {
            **scaled(len(kept_k1)),
            "pool": len(all_k1),
            "kept_fraction_of_pool": (len(kept_k1) / len(all_k1)) if all_k1 else None,
        },
        "kept_k_primary_2_to_5": {
            **scaled(len(kept_k25)),
            "pool": len(all_k25),
            "kept_fraction_of_pool": (len(kept_k25) / len(all_k25)) if all_k25 else None,
        },
        "g1_final": {
            "kept_in_sample": len(kept),
            "threshold": G1_MIN_KEPT,
            "verdict": (
                "continue_to_analysis"
                if len(kept) >= G1_MIN_KEPT
                else "fail_g1_below_200"
            ),
        },
    }


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, seed: int = SEED, min_novel: int = MIN_NOVEL) -> None:
    started_at = datetime.now(UTC)
    git_commit, git_dirty = local_git_info()
    wall0 = time.perf_counter()

    from saedm.data.steps import checkpoint_brackets, predicted_step
    from saedm.paths import data_dir, results_dir

    p15a = json.loads((results_dir() / "P1.5a" / "summary.json").read_text())
    if not p15a.get("passed"):
        raise RuntimeError("P1.5a mapping did not pass; refusing to run P1.5")

    summary_p11 = json.loads((results_dir() / "P1.1" / "summary.json").read_text())
    total_memorized = int(summary_p11["total_memorized_count"])
    sample_size = int(summary_p11["sample_size"])

    rec_path = data_dir() / "P1.4d" / "recollection.jsonl"
    expected_hash = (data_dir() / "P1.4d" / "recollection.sha256").read_text().strip()
    rows_in = [json.loads(line) for line in rec_path.read_text().splitlines() if line.strip()]
    canon = json.dumps(
        [
            {
                "sample_row": r["sample_row"],
                "pile_index": r["pile_index"],
                "k_primary": r["k_primary"],
                "k_infgram": r["k_infgram"],
                "tokens": r["tokens"],
            }
            for r in sorted(rows_in, key=lambda x: x["sample_row"])
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    got = hashlib.sha256(canon.encode()).hexdigest()
    if got != expected_hash:
        raise RuntimeError("recollection freeze hash mismatch")

    enriched = []
    for seq_i, seq in enumerate(rows_in):
        step = predicted_step(int(seq["pile_index"]))
        br = checkpoint_brackets(step)
        enriched.append({**seq, **br, "predicted_step": step, "seq_i": seq_i})

    by_rev: dict[str, list[dict]] = defaultdict(list)
    for seq in enriched:
        by_rev[seq["revision_before"]].append(
            {"seq_i": seq["seq_i"], "tokens": seq["tokens"]}
        )
        by_rev[FINAL_REV].append({"seq_i": seq["seq_i"], "tokens": seq["tokens"]})

    payloads = [
        {"revision": rev, "sequences": seqs, "seed": seed}
        for rev, seqs in sorted(by_rev.items())
    ]
    print(
        f"Launching {len(payloads)} parallel revision jobs on {gpu} "
        f"for {len(enriched)} recollection sequences"
    )
    results = list(
        eval_revision_preexposure.with_options(gpu=gpu).map(
            payloads, order_outputs=False
        )
    )

    before_out: list[dict | None] = [None] * len(enriched)
    final_out: list[dict | None] = [None] * len(enriched)
    gpu_seconds = 0.0
    device_name = None
    precision = None
    for res in results:
        gpu_seconds += float(res["elapsed_s"])
        device_name = res["device_name"]
        precision = res["precision"]
        rev = res["revision"]
        for seq_i, out in res["outs"].items():
            i = int(seq_i)
            if rev == FINAL_REV:
                final_out[i] = out
            if rev == enriched[i]["revision_before"]:
                before_out[i] = out

    rows = []
    masks = []
    pos_hits = np.zeros(32, dtype=np.float64)
    for i, seq in enumerate(enriched):
        b = before_out[i]
        f = final_out[i]
        assert b is not None and f is not None
        n_novel = int(b["n_novel"])
        kept = n_novel >= min_novel
        rows.append(
            {
                "sample_row": seq["sample_row"],
                "pile_index": seq["pile_index"],
                "k_primary": seq["k_primary"],
                "k_infgram": seq["k_infgram"],
                "predicted_step": seq["predicted_step"],
                "c_before": seq["c_before"],
                "n_pre_predicted": b["n_pre_predicted"],
                "n_novel": n_novel,
                "kept": kept,
                "mean_ce_before": b["mean_ce"],
                "mean_ce_final": f["mean_ce"],
            }
        )
        masks.append(
            {
                "sample_row": seq["sample_row"],
                "pile_index": seq["pile_index"],
                "pre_predicted": b["pre_predicted"],
                "logprob_before": b["token_logprob"],
                "logprob_final": f["token_logprob"],
            }
        )
        pos_hits += np.asarray(b["pre_predicted"], dtype=np.float64)

    pos_rate = (pos_hits / max(1, len(enriched))).tolist()
    mask_blob = json.dumps(masks, separators=(",", ":"))
    mask_hash = hashlib.sha256(mask_blob.encode()).hexdigest()

    mask_path = _write_masks_remote.remote(mask_blob, mask_hash)

    summary = _summarize(rows, sample_size, total_memorized, min_novel)
    summary.update(
        {
            "model": MODEL,
            "final_revision": FINAL_REV,
            "recollection_sha256": expected_hash,
            "mask_sha256": mask_hash,
            "mask_artifacts_path": mask_path,
            "pre_predicted_rate_by_position": pos_rate,
            "n_unique_revisions": len(by_rev),
            "gpu_hours": gpu_seconds / 3600.0,
            "wall_clock_s": time.perf_counter() - wall0,
            "gpu_requested": gpu,
        }
    )

    out_dir = results_dir() / "P1.5"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notes.md").write_text(
        f"""# P1.5 notes — pre-exposure filter

Recollection stratum only (from P1.4d). For each sequence, evaluate the last
checkpoint **before** predicted step `s(i)=i//1024` (`c_before`). Tokens whose
greedy prediction already matches the target at `c_before` are
`pre_predicted`. Keep sequences with `n_novel = 32 - n_pre_predicted >= {min_novel}`.

For k_primary 2–5 an earlier duplicate may already have been seen by
`c_before`, so the filter is conservative. Results reported separately for
k_primary==1 and 2–5.

Masks on `saedm-artifacts` at `{mask_path}` (sha256={mask_hash}).
"""
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    rows_path = out_dir / "sequence_results.jsonl"
    with rows_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    ended_at = datetime.now(UTC)
    from saedm.manifest import write_manifest

    write_manifest(
        task_id="P1.5",
        config={
            "task_id": "P1.5",
            "seed": seed,
            "min_novel": min_novel,
            "model": MODEL,
            "gpu": gpu,
            "recollection_sha256": expected_hash,
            "mask_sha256": mask_hash,
        },
        seed=seed,
        out_dir=out_dir,
        outputs=[
            str(summary_path.relative_to(REPO_ROOT)),
            str(rows_path.relative_to(REPO_ROOT)),
            str((out_dir / "notes.md").relative_to(REPO_ROOT)),
        ],
        started_at=started_at,
        ended_at=ended_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
        device={"type": "cuda", "name": device_name, "index": 0},
        precision=precision,
        model_info={"name": MODEL, "revision_final": FINAL_REV},
    )
    print(json.dumps(summary["g1_final"], indent=2))
    print("kept_total", summary["kept_total"])
    print(
        "k1",
        summary["kept_k_primary_1"]["sample_count"],
        "k2-5",
        summary["kept_k_primary_2_to_5"]["sample_count"],
    )
    print(f"gpu_hours={summary['gpu_hours']:.3f}")


@app.function(volumes=VOLUMES, timeout=600)
def _write_masks_remote(blob: str, h: str) -> str:
    path = Path("/artifacts") / "P1.5" / f"preexposure_masks_{h[:16]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(blob)
    artifacts_vol.commit()
    return str(path)
