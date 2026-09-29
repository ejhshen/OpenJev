"""Distributed reduction for whole-decision cross entropy."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.distributed as dist

from openjev.training.supervised import decision_cross_entropy


def global_weight_denominator(local_weights: Tensor) -> Tensor:
    if local_weights.ndim != 1 or not local_weights.numel():
        raise ValueError("local weights must contain complete decisions for the whole update")
    if not bool(torch.isfinite(local_weights).all()) or bool((local_weights < 0).any()):
        raise ValueError("decision weights must be finite and nonnegative")
    total = local_weights.detach().double().sum()
    if dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if not bool(total > 0):
        raise ValueError("global decision weight must be positive")
    return total


def distributed_supervised_loss(log_probs: Tensor, batch: dict, global_denominator: Tensor) -> tuple[Tensor, Tensor]:
    """FSDP averages rank gradients, so each local numerator receives W/D.

    D covers the entire global accumulation window, not this microbatch. Every
    rank performs the same number of microsteps. The returned numerator is for
    reporting and must not be averaged a second time by a trainer.
    """
    losses = decision_cross_entropy(log_probs, batch["targets"], batch["option_mask"], reduction="none")
    weights = batch.get("sample_weight")
    if weights is None:
        numerator = losses.sum()
    else:
        if weights.shape != losses.shape or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("sample weights must be finite nonnegative [B]")
        # A rank/microstep may have zero weight even when the complete global
        # accumulation window has positive weight. Its backward still runs.
        numerator = (torch.where(weights > 0, losses, 0.0) * weights).sum()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return numerator * (world_size / global_denominator).to(numerator.dtype), numerator.detach()
