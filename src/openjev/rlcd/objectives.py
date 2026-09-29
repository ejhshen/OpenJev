"""Population objectives used to specify and audit the RLCD estimators.

Full targets belong to this reference/evaluation interface, never to a bandit
actor batch. Losses in ``reinforce`` and ``ppo`` only consume selected feedback.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    return value


def validate_distribution(probabilities: Tensor, option_mask: Tensor | None = None) -> Tensor:
    """Check the [decision, option] contract without repairing probabilities."""
    if probabilities.ndim != 2 or not probabilities.is_floating_point():
        raise ValueError("probabilities must be a floating [B, K] tensor")
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("an empty decision or option batch is not supported")
    mask = torch.ones_like(probabilities, dtype=torch.bool) if option_mask is None else option_mask
    if mask.shape != probabilities.shape or mask.dtype != torch.bool or mask.device != probabilities.device:
        raise ValueError("option_mask must be bool [B, K] on the policy device")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every decision needs at least one valid option")
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
        raise ValueError("probabilities must be finite and nonnegative")
    if bool((probabilities.masked_select(~mask) != 0).any()):
        raise ValueError("masked options must have probability zero")
    if not torch.allclose(probabilities.sum(dim=1), torch.ones_like(probabilities[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("each decision distribution must sum to one")
    return mask


def validate_policy(probabilities: Tensor, log_probabilities: Tensor, option_mask: Tensor | None = None) -> Tensor:
    mask = validate_distribution(probabilities, option_mask)
    if log_probabilities.shape != probabilities.shape or not log_probabilities.is_floating_point():
        raise ValueError("log_probabilities must be a floating [B, K] tensor")
    if log_probabilities.device != probabilities.device:
        raise ValueError("probabilities and log_probabilities must share a device")
    selected_logs = log_probabilities.masked_select(mask)
    if not bool(torch.isfinite(selected_logs).all()):
        raise ValueError("valid options need finite log probabilities")
    if not torch.allclose(selected_logs.exp(), probabilities.masked_select(mask), atol=1e-5, rtol=1e-5):
        raise ValueError("probabilities and log_probabilities describe different policies")
    return mask


def known_probability_term(probabilities: Tensor, alpha: float = 1.0) -> Tensor:
    """Per-decision alpha/2 ||p||²; padded options already have probability zero."""
    return validate_alpha(alpha) * 0.5 * probabilities.square().sum(dim=1)


def population_objective(
    probabilities: Tensor,
    log_probabilities: Tensor,
    target_probabilities: Tensor,
    *,
    alpha: float = 1.0,
    option_mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Full-information mathematical reference, including the q² constant."""
    alpha = validate_alpha(alpha)
    mask = validate_policy(probabilities, log_probabilities, option_mask)
    if target_probabilities.shape != probabilities.shape or target_probabilities.device != probabilities.device:
        raise ValueError("targets must have the policy's [B, K] shape and device")
    validate_distribution(target_probabilities, mask)
    q = target_probabilities.detach()
    safe_logs = torch.where(mask, log_probabilities, 0.0)
    per_decision = alpha * 0.5 * (probabilities - q).square().sum(dim=1)
    per_decision = per_decision - (1.0 - alpha) * (q * safe_logs).sum(dim=1)
    if reduction == "none":
        return per_decision
    if reduction == "mean":
        return per_decision.mean()
    raise ValueError("reduction must be 'none' or 'mean'")
