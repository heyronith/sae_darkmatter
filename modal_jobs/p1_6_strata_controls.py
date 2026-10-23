"""P1.6 GPU — reconstruction filter, C1/C2 controls, splits, artifact push."""

from __future__ import annotations

import json
import subprocess
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
N_PER = 1_000
HARD_CE = 1.0
MIN_PREDICTED = 28
C1_CANDIDATE_POOL = 5_000
C1_SHARD = 20  # small trailing shard (~78 MB)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=4 * 60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
    memory=65536,
)
def eval_ce_greedy(payload: dict) -> dict:
    """Teacher-forced CE (+ optional greedy counts) at one revision."""
    import torch

    from saedm.loss import continuation_ce_and_greedy_batch
    from saedm.models import load_model
    from saedm.seeding import set_seed

    t0 = time.perf_counter()
    set_seed(int(payload.get("seed", 0)))
    model, _tok, info = load_model(
        MODEL,
        revision=payload["revision"],
        dtype="float32",
        device="cuda",
        attn_implementation="eager",
    )
    sequences = payload["sequences"]
    micro = int(payload.get("micro_batch", 32))
    want_greedy = bool(payload.get("want_greedy", True))
    outs: dict[int, dict] = {}
    with torch.no_grad():
        for start in range(0, len(sequences), micro):
            chunk = sequences[start : start + micro]
            batch = continuation_ce_and_greedy_batch(
                model, [s["tokens"] for s in chunk], device="cuda"
            )
            for seq, out in zip(chunk, batch, strict=True):
                entry = {
                    "mean_ce": float(out["mean_ce"]),
                    "token_ce": out["token_ce"],
                }
                if want_greedy:
                    entry["n_pre_predicted"] = int(out["n_pre_predicted"])
                    entry["pre_predicted"] = out["pre_predicted"]
                outs[int(seq["seq_i"])] = entry
    del model
    torch.cuda.empty_cache()
    from modal_jobs.common import hf_cache_vol

    hf_cache_vol.commit()
    return {
        "revision": payload["revision"],
        "outs": outs,
        "elapsed_s": time.perf_counter() - t0,
        "device_name": torch.cuda.get_device_name(0),
        "precision": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "model_info": info,
    }


@app.function(
    gpu=DEFAULT_GPU,
    timeout=2 * 60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
    memory=65536,
)
def generate_c2(payload: dict) -> dict:
    """AR greedy continuations of validation prompts → C2 sequences."""
    import torch

    from saedm.loss import greedy_continuation_exact_match_batch
    from saedm.models import load_model
    from saedm.seeding import set_seed

    t0 = time.perf_counter()
    set_seed(int(payload.get("seed", 0)))
    model, _tok, info = load_model(
        MODEL,
        revision=FINAL_REV,
        dtype="float32",
        device="cuda",
        attn_implementation="eager",
    )
    prompts = payload["prompts"]  # each: {seq_i, prompt_tokens} length 32
    micro = int(payload.get("micro_batch", 32))
    rows = []
    with torch.no_grad():
        for start in range(0, len(prompts), micro):
            chunk = prompts[start : start + micro]
            # Dummy targets so the batch helper runs; we only need generated.
            fake = [p["prompt_tokens"] + [0] * 32 for p in chunk]
            outs = greedy_continuation_exact_match_batch(model, fake, device="cuda")
            for p, out in zip(chunk, outs, strict=True):
                gen = out["generated"]
                tokens = list(p["prompt_tokens"]) + list(gen)
                rows.append(
                    {
                        "seq_i": int(p["seq_i"]),
                        "doc_i": int(p.get("doc_i", -1)),
                        "tokens": tokens,
                        "control": "C2",
                        "pile_subset": p.get("pile_subset", "val"),
                    }
                )
    del model
    torch.cuda.empty_cache()
    from modal_jobs.common import hf_cache_vol

    hf_cache_vol.commit()
    return {
        "rows": rows,
        "elapsed_s": time.perf_counter() - t0,
        "device_name": torch.cuda.get_device_name(0),
        "precision": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "model_info": info,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    from saedm.data.controls import canonical_sha256

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    h = canonical_sha256(rows)
    path.with_suffix(path.suffix + ".sha256").write_text(h + "\n")
    return h


def _push_artifacts(local_dir: Path) -> str:
    remote = "/P1.6"
    cmd = [
        "modal",
        "volume",
        "put",
        "saedm-artifacts",
        str(local_dir),
        remote,
        "--force",
    ]
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        return f"modal:saedm-artifacts:{remote}"
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"modal_upload_failed: {exc}"


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, seed: int = SEED) -> None:
    started_at = datetime.now(UTC)
    git_commit, git_dirty = local_git_info()
    wall0 = time.perf_counter()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from saedm.data.controls import (
        coarse_pile_subset,
        freeze_split,
        match_by_loss_and_subset,
        sample_rows,
    )
    from saedm.data.mmap_pile import ShardWindowReader, load_index
    from saedm.manifest import write_manifest
    from saedm.paths import data_dir, results_dir

    out_dir = results_dir() / "P1.6"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = data_dir() / "P1.6"
    pool_dir.mkdir(parents=True, exist_ok=True)
    cpu = json.loads((out_dir / "cpu_summary.json").read_text())

    tok = AutoTokenizer.from_pretrained(MODEL, revision=FINAL_REV)
    gpu_s = 0.0
    device_name = None
    precision = None

    # ----- Reconstruction from P1.5b candidates -----
    cand_path = data_dir() / "P1.5b" / "candidates.jsonl"
    candidates = [
        json.loads(line) for line in cand_path.read_text().splitlines() if line.strip()
    ]
    print(f"Loaded {len(candidates)} P1.5b candidates")

    by_sel: dict[str, list[dict]] = defaultdict(list)
    for seq in candidates:
        by_sel[f"step{seq['c_sel']}"].append(
            {"seq_i": seq["sample_i"], "tokens": seq["tokens"]}
        )
    print(f"Stage1: {len(by_sel)} c_sel revisions")
    sel_results = list(
        eval_ce_greedy.with_options(gpu=gpu).map(
            [
                {
                    "revision": rev,
                    "sequences": seqs,
                    "seed": seed,
                    "want_greedy": False,
                    "micro_batch": 64,
                }
                for rev, seqs in sorted(by_sel.items())
            ],
            order_outputs=False,
        )
    )
    sel_losses: dict[str, dict[int, dict]] = {}
    for res in sel_results:
        sel_losses[res["revision"]] = {int(k): v for k, v in res["outs"].items()}
        gpu_s += float(res["elapsed_s"])
        device_name = res["device_name"]
        precision = res["precision"]

    recon_cand = []
    for seq in candidates:
        ce = float(sel_losses[f"step{seq['c_sel']}"][seq["sample_i"]]["mean_ce"])
        if ce <= HARD_CE:
            recon_cand.append({**seq, "ce_sel": ce})
    print(f"CE(c_sel)<={HARD_CE}: {len(recon_cand)}")

    by_before: dict[str, list[dict]] = defaultdict(list)
    for seq in recon_cand:
        by_before[f"step{seq['c_before']}"].append(
            {"seq_i": seq["sample_i"], "tokens": seq["tokens"]}
        )
    before_results = list(
        eval_ce_greedy.with_options(gpu=gpu).map(
            [
                {
                    "revision": rev,
                    "sequences": seqs,
                    "seed": seed,
                    "want_greedy": True,
                    "micro_batch": 64,
                }
                for rev, seqs in sorted(by_before.items())
            ],
            order_outputs=False,
        )
    )
    before_losses: dict[str, dict[int, dict]] = {}
    for res in before_results:
        before_losses[res["revision"]] = {int(k): v for k, v in res["outs"].items()}
        gpu_s += float(res["elapsed_s"])
        device_name = res["device_name"]
        precision = res["precision"]

    reconstruction_pool = []
    for seq in recon_cand:
        g = before_losses[f"step{seq['c_before']}"][seq["sample_i"]]
        n_pre = int(g["n_pre_predicted"])
        if n_pre >= MIN_PREDICTED:
            reconstruction_pool.append(
                {
                    "memorized_row": seq["memorized_row"],
                    "pile_index": seq["pile_index"],
                    "k_primary": 1,
                    "predicted_step": seq["predicted_step"],
                    "c_sel": seq["c_sel"],
                    "c_before": seq["c_before"],
                    "ce_sel": seq["ce_sel"],
                    "n_pre_predicted_before": n_pre,
                    "tokens": seq["tokens"],
                    "stratum": "reconstruction",
                }
            )
    print(f"reconstruction pool (n_pre>={MIN_PREDICTED}): {len(reconstruction_pool)}")
    recon_sample = sample_rows(reconstruction_pool, N_PER, seed=seed)
    recon_sample = sorted(recon_sample, key=lambda r: r["pile_index"])
    recon_hash = _write_jsonl(pool_dir / "reconstruction.jsonl", recon_sample)

    # ----- Load near_dup / mid / high samples; build combined memorized for C1 -----
    strata_samples = {"reconstruction": recon_sample}
    for name in ("near_dup", "mid", "high"):
        path = pool_dir / f"{name}.jsonl"
        strata_samples[name] = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]

    mem_for_c1 = []
    for name, rows in strata_samples.items():
        for r in rows:
            text = tok.decode(r["tokens"])
            mem_for_c1.append(
                {
                    **r,
                    "stratum": name,
                    "pile_subset": coarse_pile_subset(text),
                }
            )

    # Final-checkpoint token losses for memorized pool.
    mem_payloads = [
        {
            "revision": FINAL_REV,
            "sequences": [
                {"seq_i": i, "tokens": r["tokens"]} for i, r in enumerate(mem_for_c1)
            ],
            "seed": seed,
            "want_greedy": False,
            "micro_batch": 64,
        }
    ]
    mem_res = eval_ce_greedy.with_options(gpu=gpu).remote(mem_payloads[0])
    gpu_s += float(mem_res["elapsed_s"])
    device_name = mem_res["device_name"]
    precision = mem_res["precision"]
    for i, r in enumerate(mem_for_c1):
        r["token_losses"] = mem_res["outs"][i]["token_ce"]

    # ----- C1: non-memorized windows from trailing mmap shard -----
    print("Loading mmap index + shard for C1 candidates...")
    index = load_index()
    from saedm.data.memorized import load_memorized_table

    mem_table, _ = load_memorized_table()
    memorized_set = set(int(x) for x in mem_table.column("index").to_pylist())

    with ShardWindowReader(index, C1_SHARD) as reader:
        local = reader.local_indices
        rng = np.random.default_rng(seed)
        rng.shuffle(local)
        c1_cand_rows = []
        for idx in local:
            ii = int(idx)
            if ii in memorized_set:
                continue
            tokens = reader.read_prefix(ii, 64)
            text = tok.decode(tokens)
            c1_cand_rows.append(
                {
                    "pile_index": ii,
                    "tokens": tokens,
                    "pile_subset": coarse_pile_subset(text),
                }
            )
            if len(c1_cand_rows) >= C1_CANDIDATE_POOL:
                break
    print(f"C1 candidate windows: {len(c1_cand_rows)}")

    c1_loss_res = eval_ce_greedy.with_options(gpu=gpu).remote(
        {
            "revision": FINAL_REV,
            "sequences": [
                {"seq_i": i, "tokens": r["tokens"]} for i, r in enumerate(c1_cand_rows)
            ],
            "seed": seed,
            "want_greedy": False,
            "micro_batch": 64,
        }
    )
    gpu_s += float(c1_loss_res["elapsed_s"])
    for i, r in enumerate(c1_cand_rows):
        r["token_losses"] = c1_loss_res["outs"][i]["token_ce"]

    c1_matched, c1_quality = match_by_loss_and_subset(
        mem_for_c1, c1_cand_rows, n=N_PER, seed=seed
    )
    c1_matched = sorted(c1_matched, key=lambda r: r["pile_index"])
    c1_hash = _write_jsonl(pool_dir / "C1.jsonl", c1_matched)
    print(f"C1 matched={len(c1_matched)} KS={c1_quality['ks_token_loss']:.4f}")

    # ----- C2: greedy continuations of Pile-val prompts -----
    print("Sampling Pile-val prompts for C2...")
    ds = load_dataset(
        "pietrolesci/pile-validation",
        split="validation",
        verification_mode="no_checks",
    )
    rng = np.random.default_rng(seed + 1)
    prompts = []
    doc_order = rng.permutation(len(ds))
    for doc_i in doc_order:
        text = ds[int(doc_i)]["text"]
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < 32:
            continue
        # Random 32-token prompt window.
        if len(ids) == 32:
            start = 0
        else:
            start = int(rng.integers(0, len(ids) - 32))
        prompt_tokens = ids[start : start + 32]
        prompts.append(
            {
                "seq_i": len(prompts),
                "doc_i": int(doc_i),
                "prompt_tokens": prompt_tokens,
                "pile_subset": coarse_pile_subset(tok.decode(prompt_tokens)),
            }
        )
        if len(prompts) >= N_PER:
            break
    c2_res = generate_c2.with_options(gpu=gpu).remote(
        {"prompts": prompts, "seed": seed, "micro_batch": 32}
    )
    gpu_s += float(c2_res["elapsed_s"])
    c2_rows = sorted(c2_res["rows"], key=lambda r: r["seq_i"])
    c2_hash = _write_jsonl(pool_dir / "C2.jsonl", c2_rows)

    # ----- Splits A/B -----
    split_hashes = {}
    for name, rows in {
        **{k: strata_samples[k] for k in strata_samples},
        "C1": c1_matched,
    }.items():
        a, b, ha, hb = freeze_split(rows, seed=seed)
        _write_jsonl(pool_dir / f"{name}_A.jsonl", a)
        _write_jsonl(pool_dir / f"{name}_B.jsonl", b)
        split_hashes[name] = {
            "n_a": len(a),
            "n_b": len(b),
            "sha256_a": ha,
            "sha256_b": hb,
        }

    # C3 already written by CPU; record hash from file.
    c3_rows = [
        json.loads(line)
        for line in (pool_dir / "C3.jsonl").read_text().splitlines()
        if line.strip()
    ]
    c3_hash = (pool_dir / "C3.jsonl.sha256").read_text().strip()

    remote = _push_artifacts(pool_dir)

    summary = {
        "seed": seed,
        "n_per": N_PER,
        "strata": {
            "reconstruction": {
                "n_pool": len(reconstruction_pool),
                "n_sampled": len(recon_sample),
                "sha256": recon_hash,
                "filters": {
                    "ce_sel_max": HARD_CE,
                    "min_pre_predicted_at_c_before": MIN_PREDICTED,
                    "source": "P1.5b candidates",
                },
            },
            **{
                k: {
                    "n_pool": cpu["pools"][k]["n_pool"],
                    "n_sampled": cpu["pools"][k]["n_sampled"],
                    "sha256": cpu["pools"][k]["sha256"],
                }
                for k in ("near_dup", "mid", "high")
            },
        },
        "controls": {
            "C1": {
                "n": len(c1_matched),
                "sha256": c1_hash,
                "matching": c1_quality,
                "mmap_shard": C1_SHARD,
                "candidate_pool": len(c1_cand_rows),
            },
            "C2": {"n": len(c2_rows), "sha256": c2_hash},
            "C3": {"n": len(c3_rows), "sha256": c3_hash},
        },
        "splits": split_hashes,
        "exposure_lift": cpu["exposure_lift"],
        "artifacts_remote": remote,
        "gpu_hours": gpu_s / 3600.0,
        "wall_clock_s": time.perf_counter() - wall0,
        "gpu_requested": gpu,
        "model": MODEL,
        "revision_final": FINAL_REV,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "notes.md").write_text(
        f"""# P1.6 notes — natural strata, controls, splits

## Strata (1 000 each, seed {seed})

- **reconstruction**: P1.5b candidates with CE(c_sel) ≤ {HARD_CE} and
  teacher-forced greedy ≥ {MIN_PREDICTED}/32 at c_before.
- **near_dup / mid / high**: P1.4d definitions on the P1.1 sample.

## Controls

- **C1**: non-memorized training windows from mmap shard {C1_SHARD}, matched to
  the union of stratum samples on per-token final-checkpoint CE (KS) and
  coarse Pile-subset proxy.
- **C2**: final-model AR greedy 32-token continuations of Pile-validation prompts.
- **C3**: high-entropy copy control (see cpu notes).

## Splits

50/50 Set A / Set B within each stratum and C1A / C1B. Hashes in summary.
Artifacts: `{remote}`.
"""
    )

    ended_at = datetime.now(UTC)
    write_manifest(
        task_id="P1.6",
        config={
            "task_id": "P1.6",
            "seed": seed,
            "n_per": N_PER,
            "hashes": {
                "reconstruction": recon_hash,
                "near_dup": cpu["pools"]["near_dup"]["sha256"],
                "mid": cpu["pools"]["mid"]["sha256"],
                "high": cpu["pools"]["high"]["sha256"],
                "C1": c1_hash,
                "C2": c2_hash,
                "C3": c3_hash,
                "splits": split_hashes,
            },
        },
        seed=seed,
        out_dir=out_dir,
        outputs=[
            "results/P1.6/summary.json",
            "results/P1.6/notes.md",
            "results/P1.6/cpu_summary.json",
        ],
        started_at=started_at,
        ended_at=ended_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
        device={"type": "cuda", "name": device_name, "index": 0},
        precision=precision,
        model_info={"name": MODEL, "revision": FINAL_REV},
    )
    # Touch volume commit path for any in-container writes (none here).
    artifacts_vol.commit()
    print(json.dumps({
        "strata_n": {k: v["n_sampled"] if "n_sampled" in v else v.get("n")
                     for k, v in summary["strata"].items()},
        "reconstruction_pool": len(reconstruction_pool),
        "C1_ks": c1_quality["ks_token_loss"],
        "exposure": cpu["exposure_lift"]["memorized_k1_over_all_k1"],
        "gpu_hours": summary["gpu_hours"],
    }, indent=2))
