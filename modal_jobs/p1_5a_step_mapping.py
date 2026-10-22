"""P1.5a — verify Index → training-step mapping via checkpoint loss drops (Modal GPU)."""

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
    idxs = np.linspace(0, len(ones) - 1, num=n, dtype=int)
    chosen = []
    seen: set[int] = set()
    for i in idxs:
        ii = int(i)
        if ii not in seen:
            chosen.append(ones[ii])
            seen.add(ii)
    rng = np.random.default_rng(seed)
    while len(chosen) < n:
        j = int(rng.integers(0, len(ones)))
        if j not in seen:
            chosen.append(ones[j])
            seen.add(j)
    return chosen


@app.function(
    gpu=DEFAULT_GPU,
    timeout=2 * 60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
    memory=65536,
)
def eval_revision(payload: dict) -> dict:
    """Load one checkpoint revision; return mean CE per sequence index."""
    import torch

    from saedm.loss import continuation_ce_and_greedy
    from saedm.models import load_model
    from saedm.seeding import set_seed

    t0 = time.perf_counter()
    set_seed(int(payload.get("seed", 0)))
    rev = payload["revision"]
    sequences = payload["sequences"]  # list of {probe_i, tokens}
    model, _tok, info = load_model(
        MODEL,
        revision=rev,
        dtype="float32",
        device="cuda",
        attn_implementation="eager",
    )
    losses: dict[int, float] = {}
    with torch.no_grad():
        for seq in sequences:
            out = continuation_ce_and_greedy(model, seq["tokens"], device="cuda")
            losses[int(seq["probe_i"])] = float(out["mean_ce"])
    del model
    torch.cuda.empty_cache()
    from modal_jobs.common import hf_cache_vol

    hf_cache_vol.commit()
    elapsed = time.perf_counter() - t0
    return {
        "revision": rev,
        "losses": losses,
        "elapsed_s": elapsed,
        "model_info": info,
        "device_name": torch.cuda.get_device_name(0),
        "precision": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, n_probe: int = N_PROBE, seed: int = SEED) -> None:
    started_at = datetime.now(UTC)
    git_commit, git_dirty = local_git_info()
    wall0 = time.perf_counter()

    from saedm.data.memorized import DATASET_ID, SPLIT, load_memorized_table
    from saedm.data.steps import checkpoint_brackets, predicted_step
    from saedm.paths import data_dir, results_dir
    from saedm.stats import bootstrap_median_diff_ci

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
    enriched = []
    for probe_i, seq in enumerate(probes):
        step = predicted_step(int(seq["pile_index"]))
        br = checkpoint_brackets(step)
        enriched.append({**seq, **br, "predicted_step": step, "probe_i": probe_i})

    by_rev: dict[str, list[dict]] = defaultdict(list)
    for seq in enriched:
        revs = [seq["revision_before"], seq["revision_after"], FINAL_REV]
        if seq["revision_far_before"] is not None:
            revs.append(seq["revision_far_before"])
        for rev in set(revs):
            by_rev[rev].append({"probe_i": seq["probe_i"], "tokens": seq["tokens"]})

    print(
        f"Selected {len(probes)} probes; {len(by_rev)} unique revisions; "
        f"index {probes[0]['pile_index']}–{probes[-1]['pile_index']}"
    )

    payloads = [
        {"revision": rev, "sequences": seqs, "seed": seed}
        for rev, seqs in sorted(by_rev.items())
    ]
    print(f"Launching {len(payloads)} parallel eval_revision calls on {gpu}")
    results = list(
        eval_revision.with_options(gpu=gpu).map(payloads, order_outputs=False)
    )

    losses: list[dict[str, float]] = [{} for _ in enriched]
    model_infos: dict[str, dict] = {}
    gpu_seconds = 0.0
    device_name = None
    precision = None
    for res in results:
        rev = res["revision"]
        model_infos[rev] = res["model_info"]
        gpu_seconds += float(res["elapsed_s"])
        device_name = res["device_name"]
        precision = res["precision"]
        for probe_i, ce in res["losses"].items():
            losses[int(probe_i)][rev] = float(ce)

    rows_out = []
    drops_pred = []
    largest_is_predicted = 0
    n_with_far = 0
    for i, seq in enumerate(enriched):
        rev_b = seq["revision_before"]
        rev_a = seq["revision_after"]
        rev_f = seq["revision_far_before"]
        ce_b = losses[i][rev_b]
        ce_a = losses[i][rev_a]
        ce_final = losses[i][FINAL_REV]
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
            row["ce_far_before"] = ce_f
            row["drop_far_to_before"] = drop_ctrl
            n_with_far += 1
            intervals = {"far_to_before": drop_ctrl, "before_to_after": drop_pred}
            if max(intervals, key=intervals.get) == "before_to_after":
                largest_is_predicted += 1
        rows_out.append(row)

    pred_for_ctrl = [
        r["drop_before_to_after"] for r in rows_out if "drop_far_to_before" in r
    ]
    ctrl = [r["drop_far_to_before"] for r in rows_out if "drop_far_to_before" in r]
    boot = bootstrap_median_diff_ci(
        np.asarray(pred_for_ctrl), np.asarray(ctrl), seed=seed
    )
    median_pred = float(np.median(drops_pred)) if drops_pred else float("nan")
    median_ctrl = float(np.median(ctrl)) if ctrl else float("nan")
    frac = (largest_is_predicted / n_with_far) if n_with_far else float("nan")
    passed = bool(boot["ci_low"] > 0 and median_pred > median_ctrl)

    wall_s = time.perf_counter() - wall0
    summary = {
        "model": MODEL,
        "final_revision": FINAL_REV,
        "n_probe_requested": n_probe,
        "n_probe": len(enriched),
        "n_unique_revisions": len(by_rev),
        "index_range_memorized_split": index_range,
        "dataset_meta": {
            k: meta[k] for k in ("dataset_id", "split", "revision", "n_rows")
        },
        "recollection_sha256": expected_hash,
        "median_drop_before_to_after": median_pred,
        "median_drop_far_to_before": median_ctrl,
        "bootstrap_diff_pred_minus_ctrl": boot,
        "fraction_largest_drop_is_predicted": frac,
        "n_with_far_before": n_with_far,
        "passed": passed,
        "gpu_hours": gpu_seconds / 3600.0,
        "wall_clock_s": wall_s,
        "gpu_requested": gpu,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    rows_path = out_dir / "probe_rows.jsonl"
    with rows_path.open("w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")

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
            "mapping_passed": passed,
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
        model_info={"name": MODEL, "revisions": sorted(model_infos)},
    )

    print(json.dumps({
        "passed": passed,
        "median_drop_before_to_after": median_pred,
        "median_drop_far_to_before": median_ctrl,
        "bootstrap_diff_pred_minus_ctrl": boot,
        "fraction_largest_drop_is_predicted": frac,
        "gpu_hours": summary["gpu_hours"],
        "n_unique_revisions": len(by_rev),
    }, indent=2))
    if not passed:
        print("MAPPING FAILED — do not run P1.5")
        raise SystemExit(2)
