"""P1.5c — never-trained reconstruction baseline on Pile validation (Modal GPU)."""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modal_jobs.common import REPO_ROOT, SECRETS, VOLUMES, app, local_git_info  # noqa: E402

DEFAULT_GPU = "A100"
MODEL = "EleutherAI/pythia-410m-deduped"
REVISION = "step143000"
SEED = 0
N_WINDOWS = 50_000
VAL_DATASET = "pietrolesci/pile-validation"
# Authors confirmed this is the Pile validation split never seen in Pythia training.
VAL_NOTE = (
    "pietrolesci/pile-validation — Pile validation split; Pythia authors confirmed "
    "it was not seen during training (dataset card)."
)


def _sample_windows_local(n: int, seed: int) -> tuple[list[dict], dict]:
    """Tokenize Pile val docs with NeoX; draw ``n`` random 64-token windows."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from saedm.data.memorized import SEQ_LEN

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    # Hub card lists a mismatched split name; force validation without checks.
    ds = load_dataset(
        VAL_DATASET,
        split="validation",
        verification_mode="no_checks",
    )
    # Column name: try common fields.
    col = "text" if "text" in ds.column_names else ds.column_names[0]
    rng = np.random.default_rng(seed)

    # Build a pool of windows by scanning documents.
    windows: list[list[int]] = []
    doc_ids: list[int] = []
    for doc_i, row in enumerate(ds):
        text = row[col]
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < SEQ_LEN:
            continue
        # Non-overlapping windows inside the document.
        for start in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN):
            windows.append(ids[start : start + SEQ_LEN])
            doc_ids.append(doc_i)
        if len(windows) >= n * 3 and doc_i > 1000:
            # Enough pool; stop early for speed.
            break
    if len(windows) < n:
        # Continue through remaining docs if needed.
        start_doc = (max(doc_ids) + 1) if doc_ids else 0
        for di in range(start_doc, len(ds)):
            text = ds[di][col]
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) < SEQ_LEN:
                continue
            for start in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN):
                windows.append(ids[start : start + SEQ_LEN])
                doc_ids.append(di)
            if len(windows) >= n * 2:
                break

    if len(windows) < n:
        raise RuntimeError(f"Only collected {len(windows)} windows; need {n}")

    pick = rng.choice(len(windows), size=n, replace=False)
    out = [
        {
            "window_i": int(j),
            "doc_i": int(doc_ids[int(i)]),
            "tokens": list(windows[int(i)]),
        }
        for j, i in enumerate(pick)
    ]
    meta = {
        "dataset_id": VAL_DATASET,
        "split": "validation",
        "column": col,
        "n_docs_scanned": int(max(doc_ids)) + 1 if doc_ids else 0,
        "n_windows_pool": len(windows),
        "n_sampled": n,
        "note": VAL_NOTE,
    }
    return out, meta


@app.function(
    gpu=DEFAULT_GPU,
    timeout=6 * 60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
    memory=65536,
)
def greedy_match_batch(payload: dict) -> dict:
    """Autoregressive greedy exact-match on a batch of 64-token windows."""
    import torch

    from saedm.loss import greedy_continuation_exact_match_batch
    from saedm.models import load_model
    from saedm.seeding import set_seed

    t0 = time.perf_counter()
    set_seed(int(payload.get("seed", 0)))
    model, _tok, info = load_model(
        MODEL,
        revision=REVISION,
        dtype="float32",
        device="cuda",
        attn_implementation="eager",
    )
    sequences = payload["sequences"]
    micro = int(payload.get("micro_batch", 64))
    results = []
    with torch.no_grad():
        for start in range(0, len(sequences), micro):
            chunk = sequences[start : start + micro]
            outs = greedy_continuation_exact_match_batch(
                model, [s["tokens"] for s in chunk], device="cuda"
            )
            for seq, out in zip(chunk, outs, strict=True):
                results.append(
                    {
                        "window_i": seq["window_i"],
                        "exact_match": bool(out["exact_match"]),
                        "n_match": int(out["n_match"]),
                    }
                )
    del model
    torch.cuda.empty_cache()
    from modal_jobs.common import hf_cache_vol

    hf_cache_vol.commit()
    return {
        "results": results,
        "elapsed_s": time.perf_counter() - t0,
        "model_info": info,
        "device_name": torch.cuda.get_device_name(0),
        "precision": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU, n_windows: int = N_WINDOWS, seed: int = SEED) -> None:
    started_at = datetime.now(UTC)
    git_commit, git_dirty = local_git_info()
    wall0 = time.perf_counter()

    from transformers import AutoTokenizer

    from saedm.data.dupcount import is_trivial_sequence
    from saedm.data.templates import (
        HUFFMAN_TEMPLATE_THRESHOLD,
        SEMANTIC_MEM_COMMIT,
        classify_template,
    )
    from saedm.paths import data_dir, results_dir
    from saedm.stats import wilson_interval

    out_dir = results_dir() / "P1.5c"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir().joinpath("P1.5c").mkdir(parents=True, exist_ok=True)

    print(f"Sampling {n_windows} windows from {VAL_DATASET}...")
    windows, val_meta = _sample_windows_local(n_windows, seed)
    win_path = data_dir() / "P1.5c" / "windows.jsonl"
    with win_path.open("w") as f:
        for w in windows:
            f.write(json.dumps(w) + "\n")

    # Chunk for Modal; each container micro-batches AR greedy on GPU.
    chunk = 2000
    payloads = []
    for start in range(0, len(windows), chunk):
        payloads.append(
            {
                "sequences": windows[start : start + chunk],
                "seed": seed,
                "micro_batch": 64,
            }
        )
    print(f"Launching {len(payloads)} greedy batches on {gpu}")
    batch_results = list(
        greedy_match_batch.with_options(gpu=gpu).map(payloads, order_outputs=True)
    )
    gpu_s = sum(float(r["elapsed_s"]) for r in batch_results)
    device_name = batch_results[0]["device_name"]
    precision = batch_results[0]["precision"]

    match_by_i = {}
    for br in batch_results:
        for row in br["results"]:
            match_by_i[int(row["window_i"])] = row

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    n_exact = 0
    n_pass_filters = 0
    n_exact_and_filters = 0
    exact_rows = []
    for w in windows:
        m = match_by_i[w["window_i"]]
        tr = is_trivial_sequence(w["tokens"], tok)
        text = tok.decode(w["tokens"])
        flags = classify_template(text)
        passes = (not tr["trivial"]) and (not flags["is_template"])
        if passes:
            n_pass_filters += 1
        if m["exact_match"]:
            n_exact += 1
            exact_rows.append({**w, **m, "full_text": text, **flags, **tr})
            if passes:
                n_exact_and_filters += 1

    def rate(k: int, n: int) -> dict:
        lo, hi = wilson_interval(k, n)
        return {
            "count": k,
            "n": n,
            "rate": k / n if n else 0.0,
            "ci95_low": lo,
            "ci95_high": hi,
        }

    summary = {
        "val_source": val_meta,
        "model": MODEL,
        "revision": REVISION,
        "n_windows": n_windows,
        "exact_match": rate(n_exact, n_windows),
        "nontrivial_nontemplate": rate(n_pass_filters, n_windows),
        "exact_match_after_filters": rate(n_exact_and_filters, n_windows),
        "exact_match_rate_within_filtered": rate(
            n_exact_and_filters, n_pass_filters
        ),
        "template_commit": SEMANTIC_MEM_COMMIT,
        "huffman_threshold": HUFFMAN_TEMPLATE_THRESHOLD,
        "gpu_hours": gpu_s / 3600.0,
        "wall_clock_s": time.perf_counter() - wall0,
        "gpu_requested": gpu,
        "decoding": "autoregressive greedy (Biderman memorized-evals / P1.2-style)",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    import random

    rng = random.Random(seed)
    pick = exact_rows if len(exact_rows) <= 30 else rng.sample(exact_rows, 30)
    lines = [
        "# P1.5c exact greedy matches on Pile validation\n",
        f"Showing {len(pick)} of {len(exact_rows)} exact matches.\n",
        f"Source: `{VAL_DATASET}` validation (never trained).\n",
    ]
    for j, r in enumerate(pick, 1):
        lines.append(
            f"## {j}. window_i={r['window_i']} doc_i={r['doc_i']} "
            f"trivial={r['trivial']} template={r['is_template']}\n"
        )
        lines.append("```\n" + r["full_text"] + "\n```\n")
    (out_dir / "examples.md").write_text("\n".join(lines))
    (out_dir / "notes.md").write_text(
        f"""# P1.5c notes — never-trained baseline

## Source

`{VAL_DATASET}` split `validation`. Dataset card: Pile validation split; Pythia
authors confirmed it was **not** seen during training.

Windows: NeoX-tokenize documents into contiguous non-overlapping 64-token
spans; sample {n_windows} with seed {seed}.

## Decoding

Autoregressive greedy continuation (32 tokens from 32-token prompt), exact
match to the held-out continuation — same definition as Biderman et al.
memorized-evals / the project's P1.2-style check
(`saedm.loss.greedy_continuation_exact_match`).
"""
    )

    ended_at = datetime.now(UTC)
    from saedm.manifest import write_manifest

    write_manifest(
        task_id="P1.5c",
        config={
            "task_id": "P1.5c",
            "seed": seed,
            "n_windows": n_windows,
            "val_dataset": VAL_DATASET,
            "model": MODEL,
            "gpu": gpu,
        },
        seed=seed,
        out_dir=out_dir,
        outputs=[
            str(summary_path.relative_to(REPO_ROOT)),
            str((out_dir / "examples.md").relative_to(REPO_ROOT)),
            str((out_dir / "notes.md").relative_to(REPO_ROOT)),
        ],
        started_at=started_at,
        ended_at=ended_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
        device={"type": "cuda", "name": device_name, "index": 0},
        precision=precision,
        model_info={"name": MODEL, "revision": REVISION},
    )
    print(json.dumps({
        "exact_match": summary["exact_match"],
        "exact_match_after_filters": summary["exact_match_after_filters"],
        "gpu_hours": summary["gpu_hours"],
    }, indent=2))
