"""One trainable model shared by expanded training and prefix-based serving."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from openjev.compute import compute_context, model_compute_contract

from .head import DecisionHead
from .pooling import group_branch_features, pool_branch_endpoints
from .schema import DecisionFeatures, DecisionOutput


class OpenJevDecisionModel(nn.Module):
    def __init__(
        self, adapter: nn.Module, head: DecisionHead, *,
        branch_microbatch_size: int | None = None, checkpoint_branches: bool = False,
    ) -> None:
        super().__init__()
        if adapter.get_hidden_size() != head.input_dim:
            raise ValueError("adapter hidden size and head input_dim must match")
        if branch_microbatch_size is not None and branch_microbatch_size < 1:
            raise ValueError("branch_microbatch_size must be positive")
        self.adapter = adapter
        self.head = head
        self.branch_microbatch_size = branch_microbatch_size
        self.checkpoint_branches = checkpoint_branches
        self.branch_batch_multiple = 1

    @property
    def config(self):
        return self.openjev_config

    def can_generate(self):
        return False

    def padded_branch_count(self, count):
        # A small shape vocabulary bounds upstream JIT variants. Small/long batches
        # use finer buckets so additional rows remain below 25% (and carry no loss).
        multiple = min(self.branch_batch_multiple, 1 if count < 8 else 2 if count < 16 else 4 if count < 32 else self.branch_batch_multiple)
        return (count + multiple - 1) // multiple * multiple

    def encode_expanded(self, batch: Mapping[str, Tensor]) -> DecisionFeatures:
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or not len(input_ids):
            raise ValueError("expanded inputs must be non-empty [N,L] tensors")
        position_ids = batch.get("position_ids")
        if position_ids is not None and position_ids.shape != input_ids.shape:
            raise ValueError("generic position_ids must have shape [N,L]")
        chunk_size = self.branch_microbatch_size or len(input_ids)
        questions, options = [], []
        for start in range(0, len(input_ids), chunk_size):
            stop = min(start + chunk_size, len(input_ids))
            ids, mask = input_ids[start:stop], attention_mask[start:stop]
            positions = None if position_ids is None else position_ids[start:stop]
            q_end, o_end = batch["question_end"][start:stop], batch["option_end"][start:stop]

            # Passing endpoints as arguments avoids a mutable loop-closure during
            # backward recomputation. Only pooled features survive each chunk.
            def encode_chunk(ids, mask, positions, q_end, o_end):
                real_count = len(ids)
                extra = self.padded_branch_count(real_count) - real_count
                run_ids, run_mask, run_positions = ids, mask, positions
                if extra:
                    run_ids = torch.cat((ids, ids[:1].expand(extra, -1)), 0)
                    run_mask = torch.cat((mask, mask[:1].expand(extra, -1)), 0)
                    if positions is not None:
                        run_positions = torch.cat((positions, positions[:1].expand(extra, -1)), 0)
                with compute_context(model_compute_contract(self), "backbone", ids.device):
                    hidden = self.adapter.forward_hidden(run_ids, run_mask, run_positions)[:real_count]
                return pool_branch_endpoints(hidden, q_end, o_end, mask)

            if self.checkpoint_branches and self.training and torch.is_grad_enabled():
                q, e = checkpoint(
                    encode_chunk, ids, mask, positions, q_end, o_end,
                    use_reentrant=False,
                )
            else:
                q, e = encode_chunk(ids, mask, positions, q_end, o_end)
            questions.append(q)
            options.append(e)
        return group_branch_features(
            torch.cat(questions), torch.cat(options),
            batch["decision_ptr"], batch["option_mask"],
        )

    def score_features(
        self, question_features: Tensor, option_features: Tensor, option_mask: Tensor,
        *, temperature: float = 1.0,
    ) -> DecisionOutput:
        with compute_context(model_compute_contract(self), "head", question_features.device):
            return self.head(question_features, option_features, option_mask, temperature=temperature)

    def forward(self, batch: Mapping[str, Tensor], *, temperature: float = 1.0) -> DecisionOutput:
        features = self.encode_expanded(batch)
        return self.score_features(
            features.question_features, features.option_features, features.option_mask,
            temperature=temperature,
        )
