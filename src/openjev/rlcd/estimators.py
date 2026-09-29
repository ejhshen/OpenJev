"""Shared sampling reductions and detached coefficients; no optimizer code."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .objectives import validate_alpha


@dataclass(frozen=True)
class RLLossResult:
    loss: Tensor
    per_decision_loss: Tensor
    diagnostics: dict[str, Tensor]
    estimator: str


def resolve_sample_mask(actions: Tensor, sample_mask: Tensor | None = None) -> Tensor:
    if actions.ndim != 2 or actions.dtype != torch.long or not actions.numel():
        raise ValueError("actions must be nonempty int64 [B, G]")
    mask = torch.ones_like(actions, dtype=torch.bool) if sample_mask is None else sample_mask
    if mask.shape != actions.shape or mask.dtype != torch.bool or mask.device != actions.device:
        raise ValueError("sample_mask must be bool [B, G] on the action device")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every decision must have at least one sampled action")
    return mask


def validate_samples(
    probabilities: Tensor,
    option_mask: Tensor,
    actions: Tensor,
    feedback: Tensor,
    behavior_action_probs: Tensor,
    sample_mask: Tensor | None = None,
) -> Tensor:
    mask = resolve_sample_mask(actions, sample_mask)
    if actions.shape[0] != probabilities.shape[0] or actions.device != probabilities.device:
        raise ValueError("actions must have the policy batch size and device")
    valid_actions = actions.masked_select(mask)
    if bool(((valid_actions < 0) | (valid_actions >= probabilities.shape[1])).any()):
        raise ValueError("sampled action is outside the complete option set")
    safe_actions = torch.where(mask, actions, 0)
    if not bool((option_mask.gather(1, safe_actions) | ~mask).all()):
        raise ValueError("a sampled action refers to a masked option")
    for name, values in (("feedback", feedback), ("behavior_action_probs", behavior_action_probs)):
        if values.shape != actions.shape or values.device != probabilities.device or not values.is_floating_point():
            raise ValueError(f"{name} must be floating [B, G] on the policy device")
        if not bool(torch.isfinite(values.masked_select(mask)).all()):
            raise ValueError(f"{name} contains a nonfinite valid sample")
    c = feedback.masked_select(mask)
    if not bool(((c == 0) | (c == 1)).all()):
        raise ValueError("this feedback protocol requires binary correctness")
    mu = behavior_action_probs.masked_select(mask)
    if not bool(((mu > 0) & (mu <= 1)).all()):
        raise ValueError("selected behavior probabilities must be in (0, 1]")
    return mask


def gather_actions(values: Tensor, actions: Tensor, sample_mask: Tensor) -> Tensor:
    gathered = values.gather(1, torch.where(sample_mask, actions, 0))
    return torch.where(sample_mask, gathered, 0.0)


def decision_mean(values: Tensor, sample_mask: Tensor) -> Tensor:
    """Each decision has weight one, irrespective of its number of samples."""
    if values.shape != sample_mask.shape:
        raise ValueError("sample values and mask must have identical shapes")
    counts = sample_mask.sum(dim=1)
    if not bool((counts > 0).all()):
        raise ValueError("cannot reduce a decision with zero samples")
    return torch.where(sample_mask, values, 0.0).sum(dim=1) / counts


def leave_one_out(values: Tensor, sample_mask: Tensor) -> Tensor:
    """An independent-other-actions baseline; G=1 has baseline zero."""
    if values.shape != sample_mask.shape:
        raise ValueError("sample values and mask must have identical shapes")
    safe = torch.where(sample_mask, values, 0.0)
    counts = sample_mask.sum(dim=1, keepdim=True)
    baseline = (safe.sum(dim=1, keepdim=True) - safe) / (counts - 1).clamp_min(1)
    return torch.where(sample_mask & (counts > 1), baseline, 0.0)


def baseline_values(values: Tensor, sample_mask: Tensor, baseline: str) -> Tensor:
    if baseline == "loo":
        return leave_one_out(values, sample_mask).detach()
    if baseline == "none":
        return torch.zeros_like(values)
    raise ValueError("baseline must be 'loo' or 'none'; no reward whitening is applied")


def importance_weights(action_probs: Tensor, behavior_action_probs: Tensor, sample_mask: Tensor) -> Tensor:
    safe_mu = torch.where(sample_mask, behavior_action_probs.detach(), 1.0)
    weights = action_probs.detach() / safe_mu
    if not bool(torch.isfinite(weights.masked_select(sample_mask)).all()):
        raise FloatingPointError("nonfinite importance weight; no clipping or self-normalization is applied")
    return torch.where(sample_mask, weights, 0.0).detach()


def full_score_rewards(action_probs: Tensor, feedback: Tensor, sample_mask: Tensor, alpha: float) -> Tensor:
    """Detached R = alpha(c-p_a) + (1-alpha)c/p_a, with no hidden floors."""
    alpha = validate_alpha(alpha)
    p = torch.where(sample_mask, action_probs.detach(), 1.0)
    c = torch.where(sample_mask, feedback.detach(), 0.0)
    if alpha < 1.0 and not bool((p > 0).all()):
        raise ValueError("log-score reward requires positive selected policy probabilities; no floor is inserted")
    rewards = alpha * (c - p)
    if alpha < 1.0:
        rewards = rewards + (1.0 - alpha) * c / p
    if not bool(torch.isfinite(rewards.masked_select(sample_mask)).all()):
        raise FloatingPointError("nonfinite full-score reward; change numerical precision or sampling explicitly")
    return torch.where(sample_mask, rewards, 0.0).detach()
