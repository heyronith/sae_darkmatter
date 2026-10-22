"""P1.5b — powered Index→step mapping + verified recollection mining (Modal GPU)."""

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
SEED = 0
HARD_CE_THRESH = 1.0
DROP_PRED_MIN = 0.5
MIN_NOVEL = 8
INFGRAM_MAX = 32


@app.function(
    gpu=DEFAULT_GPU,
    timeout=3 * 60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
    memory=65536,
)
def eval_revision_ce(payload: dict) -> dict:
    """Load one revision; return mean CE (+ optional greedy flags) per seq_i."""
    import torch

    from saedm.loss import continuation_ce_and_greedy
    from saedm.models import load_model
    from saedm.seeding import set_seed

    t0 = time.perf_counter()
    set_seed(int(payload.get("seed", 0)))
    rev = payload["revision"]
    want_greedy = bool(payload.get("want_greedy", False))
    model, _tok, _info = load_model(
        MODEL,
        revision=rev,
        dtype="float32",
        device="cuda",
        attn_implementation="eager",
    )
    outs: dict[int, dict] = {}
    with torch.no_grad():
        for seq in payload["sequences"]:
            out = continuation_ce_and_greedy(model, seq["tokens"], device="cuda")
            entry = {"mean_ce": float(out["mean_ce"])}
            if want_greedy:
                entry["pre_predicted"] = out["pre_predicted"]
                entry["n_novel"] = int(out["n_novel"])
                entry["n_pre_predicted"] = int(out["n_pre_predicted"])
            outs[int(seq["seq_i"])] = entry
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


def _run_revisions(
    by_rev: dict[str, list[dict]],
    *,
    gpu: str,
    seed: int,
    want_greedy_revs: set[str] | None = None,
) -> tuple[dict[str, dict[int, dict]], float, str | None, dict | None]:
    want_greedy_revs = want_greedy_revs or set()
    payloads = [
        {
            "revision": rev,
            "sequences": seqs,
            "seed": seed,
            "want_greedy": rev in want_greedy_revs,
        }
        for rev, seqs in sorted(by_rev.items())
    ]
    print(f"Launching {len(payloads)} revision jobs on {gpu}", flush=True)
    results = list(
        eval_revision_ce.with_options(gpu=gpu).map(payloads, order_outputs=False)
    )
    losses: dict[str, dict[int, dict]] = {}
    gpu_s = 0.0
    device_name = None
    precision = None
    for res in results:
        losses[res["revision"]] = {
            int(k): v for k, v in res["outs"].items()
        }
        gpu_s += float(res["elapsed_s"])
        device_name = res["device_name"]
        precision = res["precision"]
    return losses, gpu_s, device_name, precision


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, seed: int = SEED) -> None:
    started_at = datetime.now(UTC)
    git_commit, git_dirty = local_git_info()
    wall0 = time.perf_counter()

    from transformers import AutoTokenizer

    from saedm.data.dupcount import (
        PRIMARY_INDEX,
        count_ngram_ids,
        llama_query_ids_for_continuation,
    )
    from saedm.paths import data_dir, results_dir
    from saedm.stats import bootstrap_median_diff_ci, wilson_interval

    out_dir = results_dir() / "P1.5b"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_summary = json.loads((out_dir / "pool_summary.json").read_text())
    cand_path = data_dir() / "P1.5b" / "candidates.jsonl"
    candidates = [
        json.loads(line) for line in cand_path.read_text().splitlines() if line.strip()
    ]
    n_cand = len(candidates)
    print(f"Loaded {n_cand} candidates")

    # --- Stage 1: CE at c_sel ---
    by_sel: dict[str, list[dict]] = defaultdict(list)
    for seq in candidates:
        by_sel[f"step{seq['c_sel']}"].append(
            {"seq_i": seq["sample_i"], "tokens": seq["tokens"]}
        )
    sel_losses, gpu_s1, device_name, precision = _run_revisions(
        by_sel, gpu=gpu, seed=seed
    )

    hard: list[dict] = []
    n_reconstruction = 0
    for seq in candidates:
        rev = f"step{seq['c_sel']}"
        ce = float(sel_losses[rev][seq["sample_i"]]["mean_ce"])
        seq = {**seq, "ce_sel": ce}
        if ce > HARD_CE_THRESH:
            hard.append(seq)
        else:
            n_reconstruction += 1
    print(
        f"Stage1: hard={len(hard)} reconstruction(CE_sel<={HARD_CE_THRESH})="
        f"{n_reconstruction}"
    )

    # --- Stage 2: hard only ---
    by_rev2: dict[str, list[dict]] = defaultdict(list)
    for seq in hard:
        for key in ("c_pl", "c_before", "c_after"):
            by_rev2[f"step{seq[key]}"].append(
                {"seq_i": seq["sample_i"], "tokens": seq["tokens"]}
            )
        by_rev2[FINAL_REV].append(
            {"seq_i": seq["sample_i"], "tokens": seq["tokens"]}
        )
    want_greedy = {f"step{seq['c_before']}" for seq in hard}
    stage2, gpu_s2, device_name, precision = _run_revisions(
        by_rev2, gpu=gpu, seed=seed, want_greedy_revs=want_greedy
    )
    gpu_hours = (gpu_s1 + gpu_s2) / 3600.0

    rows = []
    drops_pred = []
    drops_placebo = []
    n_pred_gt_placebo = 0
    for seq in hard:
        i = seq["sample_i"]
        ce_pl = float(stage2[f"step{seq['c_pl']}"][i]["mean_ce"])
        ce_b = float(stage2[f"step{seq['c_before']}"][i]["mean_ce"])
        ce_a = float(stage2[f"step{seq['c_after']}"][i]["mean_ce"])
        ce_f = float(stage2[FINAL_REV][i]["mean_ce"])
        g = stage2[f"step{seq['c_before']}"][i]
        drop_pred = ce_b - ce_a
        drop_placebo = ce_pl - ce_b
        drops_pred.append(drop_pred)
        drops_placebo.append(drop_placebo)
        if drop_pred > drop_placebo:
            n_pred_gt_placebo += 1
        rows.append(
            {
                **{k: seq[k] for k in (
                    "sample_i",
                    "memorized_row",
                    "pile_index",
                    "k_primary",
                    "predicted_step",
                    "c_sel",
                    "c_pl",
                    "c_before",
                    "c_after",
                    "ce_sel",
                )},
                "ce_pl": ce_pl,
                "ce_before": ce_b,
                "ce_after": ce_a,
                "ce_final": ce_f,
                "drop_pred": drop_pred,
                "drop_placebo": drop_placebo,
                "n_novel_before": int(g["n_novel"]),
                "n_pre_predicted_before": int(g["n_pre_predicted"]),
                "pre_predicted_before": g["pre_predicted"],
                "tokens": seq["tokens"],
            }
        )

    boot = bootstrap_median_diff_ci(
        np.asarray(drops_pred),
        np.asarray(drops_placebo),
        seed=seed,
        paired=True,
    )
    med_pred = float(np.median(drops_pred)) if drops_pred else float("nan")
    med_pl = float(np.median(drops_placebo)) if drops_placebo else float("nan")
    frac = (n_pred_gt_placebo / len(rows)) if rows else float("nan")
    ratio_ok = bool(med_pl > 0 and med_pred >= 3.0 * med_pl) or (
        med_pl <= 0 and med_pred > 0
    )
    passed = bool(boot["ci_low"] > 0 and ratio_ok and len(rows) > 0)
    print(
        f"Mapping: med_pred={med_pred:.4f} med_placebo={med_pl:.4f} "
        f"ci_low={boot['ci_low']:.4f} frac={frac:.3f} passed={passed}"
    )

    mapping = {
        "n_hard": len(rows),
        "median_drop_pred": med_pred,
        "median_drop_placebo": med_pl,
        "bootstrap_diff_pred_minus_placebo": boot,
        "fraction_drop_pred_gt_placebo": frac,
        "pass_rule": (
            "ci_low > 0 AND median_drop_pred >= 3 * median_drop_placebo "
            "(or placebo<=0 with pred>0)"
        ),
        "passed": passed,
    }

    verified: list[dict] = []
    n_infgram_excluded = 0
    if passed:
        neox = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-410m-deduped", revision="step143000"
        )
        llama = AutoTokenizer.from_pretrained(
            "NousResearch/Llama-2-7b-hf", use_fast=True
        )
        cache_dir = data_dir() / "P1.3b" / "infini_gram_ids_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            if row["drop_pred"] < DROP_PRED_MIN:
                continue
            if row["n_novel_before"] < MIN_NOVEL:
                continue
            align = llama_query_ids_for_continuation(neox, llama, row["tokens"])
            if align["too_short"] or not align["query_ids"]:
                k_ig = 0
            else:
                res = count_ngram_ids(
                    align["query_ids"],
                    index=PRIMARY_INDEX,
                    cache_dir=cache_dir,
                    min_interval_s=1.0,
                )
                k_ig = int(res.get("count", 0))
            if k_ig > INFGRAM_MAX:
                n_infgram_excluded += 1
                continue
            verified.append({**row, "k_infgram": k_ig})

    n_filtered = int(
        pool_summary["filter_counts"]["n_candidates_after_filters"]
    )
    n_verified = len(verified)
    # Scale verified rate in the GPU sample up to the filtered candidate pool.
    if n_cand > 0:
        lo, hi = wilson_interval(n_verified, n_cand)
        scaled = {
            "sample_count": n_verified,
            "sample_size": n_cand,
            "pool_after_cpu_filters": n_filtered,
            "point": (n_verified / n_cand) * n_filtered,
            "ci95_low": lo * n_filtered,
            "ci95_high": hi * n_filtered,
            "method": (
                "wilson(verified/n_gpu_sample) * n_candidates_after_cpu_filters"
            ),
        }
    else:
        scaled = {"sample_count": 0, "point": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}

    # Freeze verified list
    freeze = [
        {
            "pile_index": r["pile_index"],
            "memorized_row": r["memorized_row"],
            "k_primary": r["k_primary"],
            "k_infgram": r["k_infgram"],
            "predicted_step": r["predicted_step"],
            "drop_pred": r["drop_pred"],
            "n_novel_before": r["n_novel_before"],
            "tokens": r["tokens"],
        }
        for r in sorted(verified, key=lambda x: x["pile_index"])
    ]
    freeze_blob = json.dumps(freeze, sort_keys=True, separators=(",", ":"))
    freeze_hash = hashlib.sha256(freeze_blob.encode()).hexdigest()
    freeze_path = data_dir() / "P1.5b" / "verified_recollection.jsonl"
    with freeze_path.open("w") as f:
        for r in freeze:
            f.write(json.dumps(r) + "\n")
    (data_dir() / "P1.5b" / "verified_recollection.sha256").write_text(
        freeze_hash + "\n"
    )

    summary = {
        "pool": pool_summary,
        "stage1": {
            "n_candidates": n_cand,
            "hard_ce_threshold": HARD_CE_THRESH,
            "n_hard": len(hard),
            "n_reconstruction_ce_sel_le_1": n_reconstruction,
            "reconstruction_fraction": n_reconstruction / n_cand if n_cand else None,
        },
        "mapping": mapping,
        "verified": {
            "drop_pred_min": DROP_PRED_MIN,
            "min_novel": MIN_NOVEL,
            "infgram_max": INFGRAM_MAX,
            "n_before_infgram_filter": sum(
                1
                for r in rows
                if r["drop_pred"] >= DROP_PRED_MIN and r["n_novel_before"] >= MIN_NOVEL
            )
            if passed
            else 0,
            "n_infgram_excluded": n_infgram_excluded,
            "n_verified": n_verified,
            "scaled_to_filtered_pool": scaled,
            "sha256": freeze_hash,
        },
        "passed": passed,
        "gpu_hours": gpu_hours,
        "wall_clock_s": time.perf_counter() - wall0,
        "gpu_requested": gpu,
        "model": MODEL,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    rows_path = out_dir / "hard_rows.jsonl"
    with rows_path.open("w") as f:
        for r in rows:
            slim = {k: v for k, v in r.items() if k != "tokens"}
            f.write(json.dumps(slim) + "\n")

    # Examples
    notes = f"""# P1.5b notes

## Epoch-1 cutoff
{pool_summary['epoch1']['derivation']}

## Mapping test
passed={passed}
median drop_pred={med_pred:.4f}, median drop_placebo={med_pl:.4f}
bootstrap CI (paired) of median(pred)-median(placebo):
[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]
fraction drop_pred > drop_placebo: {frac:.3f}

## Verified recollection
n={n_verified} (infgram exclusions={n_infgram_excluded}); sha256={freeze_hash}
"""
    (out_dir / "notes.md").write_text(notes)

    if passed and verified:
        import random

        from transformers import AutoTokenizer as AT

        tok = AT.from_pretrained("EleutherAI/pythia-410m-deduped", revision="step143000")
        rng = random.Random(SEED)
        pick = verified if len(verified) <= 30 else rng.sample(verified, 30)
        lines = [
            "# P1.5b verified recollection examples\n",
            f"Showing {len(pick)} of {len(verified)}.\n",
        ]
        for j, r in enumerate(pick, 1):
            lines.append(
                f"## {j}. pile_index={r['pile_index']} "
                f"drop_pred={r['drop_pred']:.3f} k_infgram={r['k_infgram']} "
                f"n_novel={r['n_novel_before']}\n"
            )
            lines.append("```\n" + tok.decode(r["tokens"]) + "\n```\n")
        (out_dir / "examples.md").write_text("\n".join(lines))
    else:
        (out_dir / "examples.md").write_text(
            "# P1.5b examples\n\nMapping failed or no verified sequences; none.\n"
        )

    ended_at = datetime.now(UTC)
    from saedm.manifest import write_manifest

    write_manifest(
        task_id="P1.5b",
        config={
            "task_id": "P1.5b",
            "seed": seed,
            "gpu": gpu,
            "mapping_passed": passed,
            "verified_sha256": freeze_hash,
            "hard_ce_threshold": HARD_CE_THRESH,
        },
        seed=seed,
        out_dir=out_dir,
        outputs=[
            str(summary_path.relative_to(REPO_ROOT)),
            str(rows_path.relative_to(REPO_ROOT)),
            str((out_dir / "notes.md").relative_to(REPO_ROOT)),
            str((out_dir / "examples.md").relative_to(REPO_ROOT)),
            str((out_dir / "pool_summary.json").relative_to(REPO_ROOT)),
        ],
        started_at=started_at,
        ended_at=ended_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
        device={"type": "cuda", "name": device_name, "index": 0},
        precision=precision,
        model_info={"name": MODEL, "final_revision": FINAL_REV},
    )
    print(json.dumps({
        "passed": passed,
        "n_hard": len(hard),
        "n_verified": n_verified,
        "reconstruction_fraction": summary["stage1"]["reconstruction_fraction"],
        "gpu_hours": gpu_hours,
        "scaled": scaled,
    }, indent=2))
    if not passed:
        print("MAPPING FAILED — verified list not built beyond Stage 2")
        raise SystemExit(2)
