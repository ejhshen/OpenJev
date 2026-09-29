"""A decision actor without token-LM or generation-specific PPO assumptions."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from openjev.compute import LEGACY_COMPUTE, model_compute_contract

from .supervised import distributed_supervised_loss


class OpenJevDecisionActor:
    """The boundary is a complete decision batch; N branches never becomes B.

    This facade reuses verl's FSDP utilities rather than inheriting a token-LM
    actor whose initialization and losses require unrelated rollout machinery.
    """

    def __init__(self, model, optimizer=None, *, compute_dtype=torch.bfloat16):
        self.model = model
        self.optimizer = optimizer
        self.compute_dtype = compute_dtype

    def forward(self, batch):
        if batch["option_mask"].shape[0] != len(batch["decision_ids"]):
            raise ValueError("actor batch metadata must have leading decision dimension B")
        if batch["decision_ptr"].numel() != len(batch["decision_ids"]) + 1:
            raise ValueError("decision_ptr must retain every complete decision")
        use_autocast = (model_compute_contract(self.model) == LEGACY_COMPUTE
                        and batch["input_ids"].is_cuda
                        and self.compute_dtype in (torch.bfloat16, torch.float16))
        context = torch.autocast("cuda", dtype=self.compute_dtype) if use_autocast else nullcontext()
        with context:
            return self.model(batch)

    def supervised_backward(self, batch, global_denominator):
        output = self.forward(batch)
        loss, numerator = distributed_supervised_loss(output.log_probs, batch, global_denominator)
        loss.backward()
        return {"numerator": numerator, "decisions": len(batch["decision_ids"]),
                "expanded_tokens": int(batch["attention_mask"].sum().item()),
                "padded_tokens": batch["input_ids"].numel()}

    def optimizer_step(self, max_grad_norm: float):
        if self.optimizer is None:
            raise RuntimeError("optimizer is required for a training update")
        from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_

        grad_norm = fsdp2_clip_grad_norm_(self.model.parameters(), max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()
        return grad_norm
