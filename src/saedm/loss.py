"""Teacher-forced continuation losses / autoregressive greedy matches."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from saedm.data.memorized import CONT_LEN, PROMPT_LEN, SEQ_LEN


def continuation_ce_and_greedy(
    model: Any,
    token_ids: list[int],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Mean CE and per-token teacher-forced greedy match on the continuation.

    Uses teacher forcing on the full 64-token window. Logits at positions
    ``PROMPT_LEN-1 … SEQ_LEN-2`` predict targets ``PROMPT_LEN … SEQ_LEN-1``.
    """
    if len(token_ids) < SEQ_LEN:
        raise ValueError(f"need >= {SEQ_LEN} tokens, got {len(token_ids)}")
    ids = torch.tensor([token_ids[:SEQ_LEN]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits  # [1, 64, V]
    # Predict continuation tokens.
    cont_logits = logits[0, PROMPT_LEN - 1 : SEQ_LEN - 1]  # [32, V]
    targets = ids[0, PROMPT_LEN:SEQ_LEN]  # [32]
    log_probs = F.log_softmax(cont_logits.float(), dim=-1)
    token_logprob = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    token_ce = (-token_logprob).detach().cpu().tolist()
    ce = float((-token_logprob).mean().item())
    greedy = cont_logits.argmax(dim=-1)
    pre_predicted = (greedy == targets).tolist()
    return {
        "mean_ce": ce,
        "token_ce": token_ce,
        "token_logprob": token_logprob.detach().cpu().tolist(),
        "pre_predicted": [bool(x) for x in pre_predicted],
        "n_pre_predicted": int(sum(pre_predicted)),
        "n_novel": int(CONT_LEN - sum(pre_predicted)),
        "greedy_ids": greedy.detach().cpu().tolist(),
        "target_ids": targets.detach().cpu().tolist(),
    }


def continuation_ce_and_greedy_batch(
    model: Any,
    token_ids_batch: list[list[int]],
    *,
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    """Batched teacher-forced continuation CE / greedy (same as single-row)."""
    if not token_ids_batch:
        return []
    for token_ids in token_ids_batch:
        if len(token_ids) < SEQ_LEN:
            raise ValueError(f"need >= {SEQ_LEN} tokens, got {len(token_ids)}")
    ids = torch.tensor(
        [t[:SEQ_LEN] for t in token_ids_batch], dtype=torch.long, device=device
    )
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    cont_logits = logits[:, PROMPT_LEN - 1 : SEQ_LEN - 1, :]
    targets = ids[:, PROMPT_LEN:SEQ_LEN]
    log_probs = F.log_softmax(cont_logits.float(), dim=-1)
    token_logprob = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    greedy = cont_logits.argmax(dim=-1)
    out: list[dict[str, Any]] = []
    for i in range(ids.shape[0]):
        tl = token_logprob[i]
        pred = (greedy[i] == targets[i]).tolist()
        out.append(
            {
                "mean_ce": float((-tl).mean().item()),
                "token_ce": (-tl).detach().cpu().tolist(),
                "token_logprob": tl.detach().cpu().tolist(),
                "pre_predicted": [bool(x) for x in pred],
                "n_pre_predicted": int(sum(pred)),
                "n_novel": int(CONT_LEN - sum(pred)),
                "greedy_ids": greedy[i].detach().cpu().tolist(),
                "target_ids": targets[i].detach().cpu().tolist(),
            }
        )
    return out


def greedy_continuation_exact_match(
    model: Any,
    token_ids: list[int],
    *,
    device: str | torch.device = "cpu",
    prompt_len: int = PROMPT_LEN,
    cont_len: int = CONT_LEN,
) -> dict[str, Any]:
    """Autoregressive greedy decode (Biderman et al. memorized-evals).

    Prompt = ``token_ids[:prompt_len]``; generate ``cont_len`` tokens with
    argmax at each step; exact match iff generated == ``token_ids[prompt_len:]``.
    """
    return greedy_continuation_exact_match_batch(
        model,
        [token_ids],
        device=device,
        prompt_len=prompt_len,
        cont_len=cont_len,
    )[0]


def greedy_continuation_exact_match_batch(
    model: Any,
    token_ids_batch: list[list[int]],
    *,
    device: str | torch.device = "cpu",
    prompt_len: int = PROMPT_LEN,
    cont_len: int = CONT_LEN,
) -> list[dict[str, Any]]:
    """Batched AR greedy; per-row identical to ``greedy_continuation_exact_match``."""
    if not token_ids_batch:
        return []
    need = prompt_len + cont_len
    for token_ids in token_ids_batch:
        if len(token_ids) < need:
            raise ValueError(f"need >= {need} tokens, got {len(token_ids)}")
    prompts = [list(t[:prompt_len]) for t in token_ids_batch]
    targets = [list(t[prompt_len:need]) for t in token_ids_batch]
    ids = torch.tensor(prompts, dtype=torch.long, device=device)
    # generated_steps[step][batch]
    generated_steps: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(cont_len):
            logits = model(input_ids=ids).logits[:, -1, :]
            next_ids = logits.argmax(dim=-1)
            generated_steps.append(next_ids)
            ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
    gen_mat = torch.stack(generated_steps, dim=1)  # [B, cont_len]
    out: list[dict[str, Any]] = []
    for i, target in enumerate(targets):
        generated = [int(x) for x in gen_mat[i].tolist()]
        n_match = sum(int(a == b) for a, b in zip(generated, target, strict=True))
        out.append(
            {
                "exact_match": generated == target,
                "n_match": n_match,
                "generated": generated,
                "target": target,
            }
        )
    return out
