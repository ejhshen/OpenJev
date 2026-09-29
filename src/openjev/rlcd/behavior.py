"""Explicit behavior policies and independent sampling with replacement."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .estimators import gather_actions
from .objectives import validate_distribution


@dataclass(frozen=True)
class SampledActions:
    actions: Tensor
    behavior_action_probs: Tensor
    sample_mask: Tensor
    behavior_probabilities: Tensor


def mixture_behavior(probabilities: Tensor, *, epsilon: float = 0.0, option_mask: Tensor | None = None) -> Tensor:
    mask = validate_distribution(probabilities, option_mask)
    if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be finite and in [0, 1]")
    uniform = mask.to(probabilities.dtype) / mask.sum(dim=1, keepdim=True)
    return ((1.0 - epsilon) * probabilities.detach() + epsilon * uniform).detach()


def sample_actions(
    probabilities: Tensor,
    group_sizes: int | Tensor,
    *,
    epsilon: float = 0.0,
    option_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> SampledActions:
    """G must be selected before observing this episode's feedback.

    The caller checkpoints ``generator.get_state()`` with its rollout state.
    Duplicate actions are retained. Padding is not a queried action.
    """
    mu = mixture_behavior(probabilities, epsilon=epsilon, option_mask=option_mask)
    if isinstance(group_sizes, int) and not isinstance(group_sizes, bool):
        sizes = torch.full((probabilities.shape[0],), group_sizes, dtype=torch.long, device=probabilities.device)
    elif isinstance(group_sizes, Tensor):
        if group_sizes.dtype != torch.long or group_sizes.shape != (probabilities.shape[0],):
            raise ValueError("group_sizes must be int64 [B]")
        sizes = group_sizes.to(probabilities.device)
    else:
        raise ValueError("group_sizes must be a positive integer or int64 [B]")
    if not bool((sizes > 0).all()):
        raise ValueError("every group size must be positive")
    max_g = int(sizes.max().item())
    actions = torch.multinomial(mu, max_g, replacement=True, generator=generator)
    sample_mask = torch.arange(max_g, device=probabilities.device)[None, :] < sizes[:, None]
    actions = torch.where(sample_mask, actions, -1)
    return SampledActions(actions, gather_actions(mu, actions, sample_mask), sample_mask, mu)
