"""P1.5a — verify Index → training-step mapping via checkpoint loss drops (Modal GPU)."""

from __future__ import annotations

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

from modal_jobs.common import REPO_ROOT, SECRETS, VOLUMES, app, local_git_info  # noqa: E402

DEFAULT_GPU = "A100"
MODEL = "EleutherAI/pythia-410m-deduped"
FINAL_REV = "step143000"
N_PROBE = 300
SEED = 0


def _select_probes(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Spread k_primary==1 recollection sequences across the Index range."""
    ones = [r for r in rows if int(r["k_primary"]) == 1]
    ones = sorted(ones, key=lambda r: int(r["pile_index"]))
    if len(ones) <= n:
        return ones
    # Evenly spaced by rank in sorted Index order.
    idxs = np.linspace(0, len(ones) - 1, num=n, dtype=int)
    # Deduplicate linspace collisions then fill.
    chosen = []
    seen = set()
    for i in idxs:
        if int(i) not in seen:
            chosen.append(ones[int(i)])
            seen.add(int(i))
    rng = np.random.default_rng(seed)
    while len(chosen) < n:
        j = int(rng.integers(0, len(ones)))
        if j not in seen:
            chosen.append(ones[j])
            seen.add(j)
    return chosen


@app.function(
    gpu=DEFAULT_GPU,
    timeout=6 * 60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
    memory=65536,
)
def step_mapping_remote(payload: dict) -> dict:
    """Evaluate continuation CE at bracketed checkpoints for probe sequences."""
    import torch

    from saedm.data.steps import checkpoint_brackets, predicted_step
    from saedm.loss import continuation_ce_and_greedy
    from saedm.models import load_model
    from saedm.seeding import set_seed

    t0 = time.perf_counter()
    set_seed(int(payload.get("seed", 0)))
    sequences = payload["sequences"]
    final_rev = payload.get("final_revision", FINAL_REV)

    # Attach brackets.
    enriched = []
    for seq in sequences:
        step = predicted_step(int(seq["pile_index"]))
        br = checkpoint_brackets(step)
        enriched.append({**seq, **br, "predicted_step": step})

    # Group by checkpoint revision to load each once.
    by_rev: dict[str, list[int]] = defaultdict(list)
    for i, seq in enumerate(enriched):
        revs = [seq["revision_before"], seq["revision_after"], final_rev]
        if seq["revision_far_before"] is not None:
            revs.append(seq["revision_far_before"])
        for rev in revs:
            by_rev[rev].append(i)

    # losses[i][rev] = mean_ce
    losses: list[dict[str, float]] = [{} for _ in enriched]
    model_infos: dict[str, dict] = {}
    device = "cuda"

    for rev, indices in sorted(by_rev.items(), key=lambda x: x[0]):
        print(f"Loading {MODEL}@{rev} for {len(indices)} sequences", flush=True)
        model, _tok, info = load_model(
            MODEL,
            revision=rev,
            dtype="float32",
            device=device,
            attn_implementation="eager",
        )
        model_infos[rev] = info
        with torch.no_grad():
            for i in indices:
                out = continuation_ce_and_greedy(
                    model, enriched[i]["tokens"], device=device
                )
                losses[i][rev] = float(out["mean_ce"])
        del model
        torch.cuda.empty_cache()
        # Persist HF cache.
        from modal_jobs.common import hf_cache_vol

        hf_cache_vol.commit()

    rows_out = []
    drops_pred = []
    drops_ctrl = []
    largest_is_predicted = 0
    n_with_far = 0

    for i, seq in enumerate(enriched):
        rev_b = seq["revision_before"]
        rev_a = seq["revision_after"]
        rev_f = seq["revision_far_before"]
        ce_b = losses[i][rev_b]
        ce_a = losses[i][rev_a]
        ce_final = losses[i][final_rev]
        drop_pred = ce_b - ce_a
        drops_pred.append(drop_pred)
        row = {
            "sample_row": seq["sample_row"],
            "pile_index": seq["pile_index"],
            "predicted_step": seq["predicted_step"],
            "c_before": seq["c_before"],
            "c_after": seq["c_after"],
            "c_far_before": seq["c_far_before"],
            "ce_before": ce_b,
            "ce_after": ce_a,
            "ce_final": ce_final,
            "drop_before_to_after": drop_pred,
        }
        if rev_f is not None and rev_f in losses[i]:
            ce_f = losses[i][rev_f]
            drop_ctrl = ce_f - ce_b
            drops_ctrl.append(drop_ctrl)
            row["ce_far_before"] = ce_f
            row["drop_far_to_before"] = drop_ctrl
            n_with_far += 1
            # Largest single-interval drop among available consecutive brackets.
            intervals = {
                "far_to_before": drop_ctrl,
                "before_to_after": drop_pred,
            }
            # Also compare before→final as a long interval? Spec says
            # "largest single-interval drop is the predicted interval"
            # → among far→before and before→after (the two comparable brackets).
            if max(intervals, key=intervals.get) == "before_to_after":
                largest_is_predicted += 1
        rows_out.append(row)

    from saedm.stats import bootstrap_median_diff_ci

    # Align control drops: only sequences with far_before.
    pred_for_ctrl = [
        r["drop_before_to_after"]
        for r in rows_out
        if "drop_far_to_before" in r
    ]
    ctrl = [r["drop_far_to_before"] for r in rows_out if "drop_far_to_before" in r]
    boot = bootstrap_median_diff_ci(
        np.asarray(pred_for_ctrl),
        np.asarray(ctrl),
        seed=int(payload.get("seed", 0)),
    )
    median_pred = float(np.median(drops_pred)) if drops_pred else float("nan")
    median_ctrl = float(np.median(ctrl)) if ctrl else float("nan")
    frac = (largest_is_predicted / n_with_far) if n_with_far else float("nan")

    # Pass: median pred drop clearly larger than control (diff CI lower > 0).
    passed = bool(boot["ci_low"] > 0 and median_pred > median_ctrl)

    elapsed_s = time.perf_counter() - t0
    return {
        "n_sequences": len(enriched),
        "n_with_far_before": n_with_far,
        "median_drop_before_to_after": median_pred,
        "median_drop_far_to_before": median_ctrl,
        "bootstrap_diff": boot,
        "fraction_largest_drop_is_predicted": frac,
        "passed": passed,
        "rows": rows_out,
        "model_infos": model_infos,
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
        "elapsed_s": elapsed_s,
        "gpu_hours": elapsed_s / 3600.0,
        "gpu_requested": payload.get("gpu", DEFAULT_GPU),
    }


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, n_probe: int = N_PROBE, seed: int = SEED) -> None:
    started_at = datetime.now(UTC)
    git_commit, git_dirty = local_git_info()

    from saedm.data.memorized import DATASET_ID, SPLIT, load_memorized_table
    from saedm.paths import data_dir, results_dir

    # Docs notes are written locally (also cover index range).
    out_dir = results_dir() / "P1.5a"
    out_dir.mkdir(parents=True, exist_ok=True)

    table, meta = load_memorized_table()
    all_idx = table.column("index").to_pylist()
    index_range = {"min": int(min(all_idx)), "max": int(max(all_idx)), "n": len(all_idx)}

    notes = f"""# P1.5a notes — Index → step mapping

## Docs / issue findings

- Pythia trains with batch size **1024** sequences of length **2049**
  (README; EleutherAI/pythia#123). Item `j` at step `k` is
  `dataset[(k * 1024) + j]` on the **preshuffled** mmap
  (`EleutherAI/pile-deduped-pythia-preshuffled`).
- Therefore a sequence with memorized-evals / Pile Index `i` is first (and,
  for that packed window, only) trained at
  **`s(i) = i // 1024`**.
- Deduped models run ≈**1.5 epochs** on 207B tokens (README). The preshuffled
  mmap already contains 143000 × 1024 packed windows; later Indices are
  differently packed re-appearances of earlier documents (EleutherAI/pythia#144,
  StellaAthena: concatenation before packing ⇒ tokens can shift position).
  That does **not** change `s(i)` for a given Index: each Index is visited
  exactly once in the linear pass over the mmap.

## Memorized split index range (`{DATASET_ID}` / `{SPLIT}`)

- n = {index_range['n']}
- min Index = {index_range['min']} → s = {index_range['min'] // 1024}
- max Index = {index_range['max']} → s = {index_range['max'] // 1024}

## Empirical test

See `summary.json`: median CE drops c_before→c_after vs c_far_before→c_before
on {n_probe} recollection sequences with k_primary==1, plus bootstrap CI and
fraction whose largest bracket drop is the predicted interval.
"""
    (out_dir / "notes.md").write_text(notes)

    rec_path = data_dir() / "P1.4d" / "recollection.jsonl"
    if not rec_path.exists():
        raise FileNotFoundError(rec_path)
    rows = [json.loads(line) for line in rec_path.read_text().splitlines() if line.strip()]
    expected_hash = (data_dir() / "P1.4d" / "recollection.sha256").read_text().strip()
    import hashlib

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
        raise RuntimeError(f"recollection freeze hash mismatch: {got} != {expected_hash}")

    probes = _select_probes(rows, n_probe, seed)
    print(
        f"Selected {len(probes)} k_primary==1 probes; "
        f"index range {probes[0]['pile_index']}–{probes[-1]['pile_index']}"
    )

    # Stash probes on artifacts volume via a tiny remote write for provenance.
    probe_blob = json.dumps(probes)
    print(f"Launching step_mapping_remote on {gpu} (commit={git_commit[:12]})")
    result = step_mapping_remote.with_options(gpu=gpu).remote(
        {
            "sequences": probes,
            "seed": seed,
            "final_revision": FINAL_REV,
            "gpu": gpu,
        }
    )

    summary = {
        "model": MODEL,
        "final_revision": FINAL_REV,
        "n_probe_requested": n_probe,
        "n_probe": result["n_sequences"],
        "index_range_memorized_split": index_range,
        "dataset_meta": {k: meta[k] for k in ("dataset_id", "split", "revision", "n_rows")},
        "recollection_sha256": expected_hash,
        "median_drop_before_to_after": result["median_drop_before_to_after"],
        "median_drop_far_to_before": result["median_drop_far_to_before"],
        "bootstrap_diff_pred_minus_ctrl": result["bootstrap_diff"],
        "fraction_largest_drop_is_predicted": result["fraction_largest_drop_is_predicted"],
        "n_with_far_before": result["n_with_far_before"],
        "passed": result["passed"],
        "gpu_hours": result["gpu_hours"],
        "elapsed_s": result["elapsed_s"],
        "gpu_requested": gpu,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    rows_path = out_dir / "probe_rows.jsonl"
    with rows_path.open("w") as f:
        for row in result["rows"]:
            f.write(json.dumps(row) + "\n")

    # Save probe list + full remote payload hash on artifacts via local copy note.
    artifacts_note = out_dir / "artifacts_pointer.txt"
    artifacts_note.write_text(
        f"probes_sha256={hashlib.sha256(probe_blob.encode()).hexdigest()}\n"
        f"recollection_sha256={expected_hash}\n"
    )

    ended_at = datetime.now(UTC)
    from saedm.manifest import write_manifest

    write_manifest(
        task_id="P1.5a",
        config={
            "task_id": "P1.5a",
            "seed": seed,
            "n_probe": n_probe,
            "model": MODEL,
            "gpu": gpu,
            "recollection_sha256": expected_hash,
            "mapping_passed": result["passed"],
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
        model_info={"name": MODEL, "revisions": sorted(result["model_infos"])},
    )

    print(json.dumps({k: summary[k] for k in (
        "passed",
        "median_drop_before_to_after",
        "median_drop_far_to_before",
        "bootstrap_diff_pred_minus_ctrl",
        "fraction_largest_drop_is_predicted",
        "gpu_hours",
    )}, indent=2))
    if not result["passed"]:
        print("MAPPING FAILED — do not run P1.5")
        raise SystemExit(2)
