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
    ce = float((-token_logprob).mean().item())
    greedy = cont_logits.argmax(dim=-1)
    pre_predicted = (greedy == targets).tolist()
    return {
        "mean_ce": ce,
        "token_logprob": token_logprob.detach().cpu().tolist(),
        "pre_predicted": [bool(x) for x in pre_predicted],
        "n_pre_predicted": int(sum(pre_predicted)),
        "n_novel": int(CONT_LEN - sum(pre_predicted)),
        "greedy_ids": greedy.detach().cpu().tolist(),
        "target_ids": targets.detach().cpu().tolist(),
    }


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
    if len(token_ids) < prompt_len + cont_len:
        raise ValueError(
            f"need >= {prompt_len + cont_len} tokens, got {len(token_ids)}"
        )
    prompt = list(token_ids[:prompt_len])
    target = list(token_ids[prompt_len : prompt_len + cont_len])
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(cont_len):
            logits = model(input_ids=ids).logits[0, -1]
            next_id = int(logits.argmax(dim=-1).item())
            generated.append(next_id)
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )
    n_match = sum(int(a == b) for a, b in zip(generated, target, strict=True))
    return {
        "exact_match": generated == target,
        "n_match": n_match,
        "generated": generated,
        "target": target,
    }
