"""Qwen3.5 differentiable text forward and equal-length prefix execution."""

from pathlib import Path
from uuid import uuid4

import torch

from openjev.models.base import DecisionBackboneAdapter
from .prefix_state import Qwen35PrefixState, fork_prefix


class Qwen35Adapter(DecisionBackboneAdapter):
    def __init__(self, backbone, *, recurrent_cache_dtype="native"):
        super().__init__()
        if recurrent_cache_dtype not in {"native", "float32"}:
            raise ValueError("recurrent_cache_dtype must be native or float32")
        self.backbone = backbone
        self._state_owner = uuid4().hex
        self.recurrent_cache_dtype = recurrent_cache_dtype

    @classmethod
    def from_prepared(cls, path, *, device="cpu", dtype="bfloat16", attn_implementation="sdpa", recurrent_cache_dtype="native"):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        model = Qwen3_5TextModel.from_pretrained(Path(path) / "backbone", dtype=dtype,
                                               attn_implementation=attn_implementation,
                                               local_files_only=True)
        return cls(model.to(device), recurrent_cache_dtype=recurrent_cache_dtype)

    def get_hidden_size(self):
        return self.backbone.config.hidden_size

    def forward_hidden(self, input_ids, attention_mask, position_ids=None):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, use_cache=False).last_hidden_state

    @torch.inference_mode()
    def prefill(self, input_ids, attention_mask=None):
        if self.training:
            raise ValueError("prefix execution requires eval mode")
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("prefill expects nonempty [batch, tokens]")
        if attention_mask is not None and (attention_mask.shape != input_ids.shape or not bool(attention_mask.all())):
            raise ValueError("Qwen prefix execution currently requires unpadded equal lengths")
        cache = None
        if self.recurrent_cache_dtype == "float32":
            from .experimental_cache import make_fp32_recurrent_cache
            cache = make_fp32_recurrent_cache(self.backbone.config)
        result = self.backbone(input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
        return Qwen35PrefixState(result.past_key_values, input_ids.shape[1], input_ids.shape[0], self._state_owner)

    @torch.inference_mode()
    def fork_state(self, state, branch_indices):
        self._validate_state(state)
        return fork_prefix(state, branch_indices)

    def _validate_state(self, state):
        if self.training or state.owner != self._state_owner:
            raise ValueError("state belongs to another model or adapter is training")

    @torch.inference_mode()
    def continue_from(self, state, suffix_tokens, lengths=None):
        self._validate_state(state)
        if suffix_tokens.ndim != 2 or suffix_tokens.shape[0] != state.batch_size or suffix_tokens.shape[1] == 0:
            raise ValueError("suffix shape does not match prefix batch")
        if lengths is not None and not bool((torch.as_tensor(lengths) == suffix_tokens.shape[1]).all()):
            raise ValueError("Qwen continuation requires unpadded equal suffix lengths")
        positions = torch.arange(state.length, state.length + suffix_tokens.shape[1], device=suffix_tokens.device)
        positions = positions.unsqueeze(0).expand(state.batch_size, -1)
        mask = torch.ones((state.batch_size, state.length + suffix_tokens.shape[1]), device=suffix_tokens.device, dtype=torch.long)
        result = self.backbone(input_ids=suffix_tokens, attention_mask=mask, position_ids=positions,
                               past_key_values=state.cache, use_cache=True)
        updated = Qwen35PrefixState(result.past_key_values, state.length + suffix_tokens.shape[1], state.batch_size, state.owner)
        return result.last_hidden_state, updated
