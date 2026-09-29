"""REINFORCE-Analysis with an exact probability term and sampled outcome feedback.

These surrogate *values* are not the population objective or an RL performance
metric. Only their specified derivatives estimate the probability-loss gradient.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .diagnostics import feedback_diagnostics
from .estimators import (
    RLLossResult, baseline_values, decision_mean, full_score_rewards,
    gather_actions, importance_weights, validate_samples,
)
from .objectives import known_probability_term, validate_alpha, validate_policy


def analytic_probability_term_loss(
    probabilities: Tensor,
    log_probabilities: Tensor,
    actions: Tensor,
    feedback: Tensor,
    behavior_action_probs: Tensor,
    *,
    alpha: float = 1.0,
    option_mask: Tensor | None = None,
    sample_mask: Tensor | None = None,
    baseline: str = "loo",
) -> RLLossResult:
    """Compute alpha/2 ||p||² exactly and estimate only the outcome term."""
    alpha = validate_alpha(alpha)
    option_mask = validate_policy(probabilities, log_probabilities, option_mask)
    sample_mask = validate_samples(probabilities, option_mask, actions, feedback, behavior_action_probs, sample_mask)
    p_a = gather_actions(probabilities, actions, sample_mask)
    logp_a = gather_actions(log_probabilities, actions, sample_mask)
    safe_mu = torch.where(sample_mask, behavior_action_probs.detach(), 1.0)
    c = torch.where(sample_mask, feedback.detach(), 0.0)
    outcome_coefficient = (((alpha * p_a.detach() + 1.0 - alpha) * c) / safe_mu).detach()
    if not bool(torch.isfinite(outcome_coefficient.masked_select(sample_mask)).all()):
        raise FloatingPointError("nonfinite outcome coefficient; no inverse-probability cap is applied")
    importance = importance_weights(p_a, behavior_action_probs, sample_mask)
    # This baseline term has zero expected score-gradient. Subtracting an
    # unweighted LOO baseline from an off-policy coefficient would not.
    coefficients = (outcome_coefficient - importance * baseline_values(outcome_coefficient, sample_mask, baseline)).detach()
    known_term = known_probability_term(probabilities, alpha)
    per_decision = known_term - decision_mean(coefficients * logp_a, sample_mask)
    diagnostics = feedback_diagnostics(actions, feedback, sample_mask, option_mask, coefficients=coefficients, importance=importance)
    diagnostics["known_probability_term"] = known_term.detach().mean()
    return RLLossResult(per_decision.mean(), per_decision, diagnostics, "analytic_probability_term_v1")
