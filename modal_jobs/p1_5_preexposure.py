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
    timeout=12 * 60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
    memory=65536,
)
def preexposure_remote(payload: dict) -> dict:
    """Run c_before + final CE/greedy for every recollection sequence."""
    import torch

    from saedm.data.steps import checkpoint_brackets, predicted_step
    from saedm.loss import continuation_ce_and_greedy
    from saedm.models import load_model
    from saedm.seeding import set_seed

    t0 = time.perf_counter()
    set_seed(int(payload.get("seed", 0)))
    sequences = payload["sequences"]
    final_rev = payload.get("final_revision", FINAL_REV)
    min_novel = int(payload.get("min_novel", MIN_NOVEL))

    enriched = []
    for seq in sequences:
        step = predicted_step(int(seq["pile_index"]))
        br = checkpoint_brackets(step)
        enriched.append({**seq, **br, "predicted_step": step})

    by_rev: dict[str, list[int]] = defaultdict(list)
    for i, seq in enumerate(enriched):
        by_rev[seq["revision_before"]].append(i)
        by_rev[final_rev].append(i)

    # Store per-seq outputs keyed by role.
    before_out: list[dict | None] = [None] * len(enriched)
    final_out: list[dict | None] = [None] * len(enriched)
    device = "cuda"

    for rev, indices in sorted(by_rev.items(), key=lambda x: x[0]):
        print(f"Loading {MODEL}@{rev} for {len(indices)} seqs", flush=True)
        model, _tok, _info = load_model(
            MODEL,
            revision=rev,
            dtype="float32",
            device=device,
            attn_implementation="eager",
        )
        with torch.no_grad():
            for i in indices:
                out = continuation_ce_and_greedy(
                    model, enriched[i]["tokens"], device=device
                )
                if rev == final_rev:
                    final_out[i] = out
                if rev == enriched[i]["revision_before"]:
                    before_out[i] = out
        del model
        torch.cuda.empty_cache()
        from modal_jobs.common import hf_cache_vol

        hf_cache_vol.commit()

    rows = []
    masks = []
    pos_hits = np.zeros(32, dtype=np.float64)
    for i, seq in enumerate(enriched):
        b = before_out[i]
        f = final_out[i]
        assert b is not None and f is not None
        n_novel = int(b["n_novel"])
        kept = n_novel >= min_novel
        row = {
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
        rows.append(row)
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

    # Persist masks on artifacts volume.
    mask_blob = json.dumps(masks, separators=(",", ":"))
    mask_hash = hashlib.sha256(mask_blob.encode()).hexdigest()
    mask_path = Path("/artifacts") / "P1.5" / f"preexposure_masks_{mask_hash[:16]}.json"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.write_text(mask_blob)
    artifacts_vol.commit()

    elapsed_s = time.perf_counter() - t0
    return {
        "rows": rows,
        "mask_hash": mask_hash,
        "mask_path": str(mask_path),
        "pre_predicted_rate_by_position": pos_rate,
        "elapsed_s": elapsed_s,
        "gpu_hours": elapsed_s / 3600.0,
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

    # Per-position pre_predicted rate from masks is computed locally after return;
    # here we only have n_pre_predicted aggregates. Position rates need masks —
    # recompute from rows is insufficient. We'll compute position rates in
    # local_entrypoint from the returned mask file... masks stay on volume.
    # Instead, have remote also return position rates.
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

    # Require P1.5a pass.
    from saedm.paths import data_dir, results_dir

    p15a = json.loads((results_dir() / "P1.5a" / "summary.json").read_text())
    if not p15a.get("passed"):
        raise RuntimeError("P1.5a mapping did not pass; refusing to run P1.5")

    summary_p11 = json.loads((results_dir() / "P1.1" / "summary.json").read_text())
    total_memorized = int(summary_p11["total_memorized_count"])
    sample_size = int(summary_p11["sample_size"])

    rec_path = data_dir() / "P1.4d" / "recollection.jsonl"
    expected_hash = (data_dir() / "P1.4d" / "recollection.sha256").read_text().strip()
    rows = [json.loads(line) for line in rec_path.read_text().splitlines() if line.strip()]
    canon = json.dumps(
        [
            {
                "sample_row": r["sample_row"],
                "pile_index": r["pile_index"],
                "k_primary": r["k_primary"],
                "k_infgram": r["k_infgram"],
                "tokens": r["tokens"],
            }
            for r in sorted(rows, key=lambda x: x["sample_row"])
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    got = hashlib.sha256(canon.encode()).hexdigest()
    if got != expected_hash:
        raise RuntimeError("recollection freeze hash mismatch")

    print(f"Launching preexposure on {gpu} for {len(rows)} recollection sequences")
    result = preexposure_remote.with_options(gpu=gpu).remote(
        {
            "sequences": rows,
            "seed": seed,
            "final_revision": FINAL_REV,
            "min_novel": min_novel,
            "gpu": gpu,
        }
    )

    # Position-wise pre_predicted rates: fetch masks from volume via a tiny
    # remote read is awkward; recompute from returned rows only has counts.
    # Ask remote to include position rates — patch by computing here if we
    # add them to the return. For now, recompute using a second lightweight
    # field. We'll add position rates in the remote return.
    summary = _summarize(result["rows"], sample_size, total_memorized, min_novel)
    summary.update(
        {
            "model": MODEL,
            "final_revision": FINAL_REV,
            "recollection_sha256": expected_hash,
            "mask_sha256": result["mask_hash"],
            "mask_artifacts_path": result["mask_path"],
            "gpu_hours": result["gpu_hours"],
            "elapsed_s": result["elapsed_s"],
            "gpu_requested": gpu,
            "p15a_commit_ref": p15a.get("passed"),
            "pre_predicted_rate_by_position": result.get(
                "pre_predicted_rate_by_position"
            ),
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

Masks (per-token pre_predicted + log-probs at c_before and {FINAL_REV}) are on
the `saedm-artifacts` volume at `{result['mask_path']}`
(sha256={result['mask_hash']}).
"""
    )
    summary_path = out_dir / "summary.json"

    # Write per-sequence results locally (without huge logprob vectors).
    rows_path = out_dir / "sequence_results.jsonl"
    with rows_path.open("w") as f:
        for row in result["rows"]:
            f.write(json.dumps(row) + "\n")

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

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
            "mask_sha256": result["mask_hash"],
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
        device=result["device"],
        precision=result["precision"],
        model_info={"name": MODEL, "revision_final": FINAL_REV},
    )
    print(json.dumps(summary["g1_final"], indent=2))
    print("kept_total", summary["kept_total"])
    print(f"gpu_hours={result['gpu_hours']:.3f}")
