"""Detached descriptive statistics; reward is not a probability-quality metric."""

from __future__ import annotations

import torch
from torch import Tensor

from .estimators import decision_mean


def feedback_diagnostics(
    actions: Tensor,
    feedback: Tensor,
    sample_mask: Tensor,
    option_mask: Tensor,
    *,
    coefficients: Tensor | None = None,
    importance: Tensor | None = None,
) -> dict[str, Tensor]:
    with torch.no_grad():
        # A valid draw is new if no earlier valid draw took the same action.
        g = actions.shape[1]
        earlier = torch.arange(g, device=actions.device)[None, :] < torch.arange(g, device=actions.device)[:, None]
        same = actions[:, :, None] == actions[:, None, :]
        seen = (same & earlier[None, :, :] & sample_mask[:, None, :]).any(dim=2)
        unique = (sample_mask & ~seen).sum(dim=1)
        values = {
            "feedback_queries": sample_mask.sum(),
            "mean_group_size": sample_mask.sum(dim=1).float().mean(),
            "correct_feedback_rate": decision_mean(feedback, sample_mask).mean(),
            "mean_unique_actions": unique.float().mean(),
            "mean_action_coverage": (unique / option_mask.sum(dim=1)).mean(),
        }
        if coefficients is not None:
            zero = ((coefficients == 0) | ~sample_mask).all(dim=1)
            values["zero_coefficient_group_fraction"] = zero.float().mean()
            centered = coefficients - decision_mean(coefficients, sample_mask)[:, None]
            values["within_group_coefficient_variance"] = decision_mean(centered.square(), sample_mask).mean()
        if importance is not None:
            values["mean_importance"] = decision_mean(importance, sample_mask).mean()
            values["max_importance"] = importance.masked_select(sample_mask).max()
        return {key: value.detach() for key, value in values.items()}


def policy_kl(old_probabilities: Tensor, old_log_probabilities: Tensor, log_probabilities: Tensor, option_mask: Tensor) -> Tensor:
    """Exact categorical KL(old || new), averaged equally over decisions."""
    with torch.no_grad():
        old_logs = torch.where(option_mask, old_log_probabilities, 0.0)
        new_logs = torch.where(option_mask, log_probabilities, 0.0)
        return (old_probabilities * (old_logs - new_logs)).sum(dim=1).mean().detach()
