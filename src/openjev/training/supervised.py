"""Full-distribution cross entropy reduced by decision, never by option."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def decision_cross_entropy(log_probs: Tensor, targets: Tensor, option_mask: Tensor,
                           sample_weight: Tensor | None = None, *, reduction: str = "mean") -> Tensor:
    """Compute FP32 decision CE from already normalized K-way log probabilities.

    ``mean`` divides by number of decisions, or by the sum of supplied weights.
    ``sum`` is the numerator needed for exact distributed/microbatch reduction;
    the integration must divide once by the global decision/weight denominator.
    ``none`` returns weighted per-decision losses. Padded ``-inf`` log entries
    are replaced before multiplication, avoiding 0 * -inf NaNs.
    """
    import torch

    if log_probs.ndim != 2 or targets.shape != log_probs.shape or option_mask.shape != log_probs.shape:
        raise ValueError("log_probs, targets and option_mask must have matching [B,K] shapes")
    if log_probs.shape[0] == 0 or option_mask.dtype != torch.bool:
        raise ValueError("a nonempty batch and boolean option_mask are required")
    if not bool(option_mask.any(dim=-1).all()):
        raise ValueError("each decision requires at least one valid option")
    target = targets.float()
    if not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
        raise ValueError("targets must be finite and nonnegative")
    if bool((target.masked_select(~option_mask) != 0).any()):
        raise ValueError("padded options cannot have target mass")
    if not torch.allclose(target.sum(-1), torch.ones_like(target.sum(-1)), atol=1e-6, rtol=0):
        raise ValueError("each decision target must sum to 1")
    logs = log_probs.float()
    valid_logs = logs.masked_select(option_mask)
    if bool(torch.isnan(valid_logs).any()) or bool(torch.isposinf(valid_logs).any()):
        raise ValueError("valid log probabilities cannot contain NaN or +inf")
    normalizer = torch.logsumexp(logs.masked_fill(~option_mask, -torch.inf), dim=-1)
    if not torch.allclose(normalizer, torch.zeros_like(normalizer), atol=2e-3, rtol=0):
        raise ValueError("valid log_probs must be normalized over the full option set")
    safe_logs = torch.where(option_mask & (target > 0), logs, torch.zeros_like(logs))
    losses = -(target * safe_logs).sum(-1)
    if sample_weight is None:
        denominator = float(log_probs.shape[0])
    else:
        if sample_weight.shape != (log_probs.shape[0],):
            raise ValueError("sample_weight must have shape [B]")
        weights = sample_weight.float()
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
            raise ValueError("sample weights must be finite, nonnegative and have positive sum")
        # Zero-weight impossible events contribute zero rather than 0 * inf.
        losses = torch.where(weights > 0, losses, torch.zeros_like(losses)) * weights
        denominator = weights.sum()
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.sum() / denominator
    raise ValueError("reduction must be none, sum or mean")


supervised_loss = decision_cross_entropy
