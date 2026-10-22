"""Teacher-forced continuation losses / greedy matches."""

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
    """Mean CE and per-token greedy match on the 32-token continuation.

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
