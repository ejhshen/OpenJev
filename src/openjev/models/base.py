"""Small contracts shared by assembly, training and runtime."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn


@dataclass(frozen=True)
class BackendCapabilities:
    training_forward: bool = True
    causal_prefix: bool = True
    prefix_reuse: bool = False
    fork: bool = False
    multi_token_continuation: bool = False
    cache_equal_lengths_only: bool = True


@dataclass(frozen=True)
class TrainingLayout:
    decoder_blocks: tuple[nn.Module, ...]
    checkpoint_blocks: tuple[nn.Module, ...]


class DecisionBackboneAdapter(nn.Module, ABC):
    @abstractmethod
    def forward_hidden(self, input_ids, attention_mask, position_ids=None) -> torch.Tensor:
        """Differentiable hidden states [branches, tokens, hidden]."""

    @abstractmethod
    def get_hidden_size(self) -> int:
        pass

    def prefill(self, input_ids, attention_mask=None):
        raise NotImplementedError("this adapter does not support prefix reuse")

    def fork_state(self, state, branch_indices):
        raise NotImplementedError("this adapter does not support state branching")

    def continue_from(self, state, suffix_tokens, lengths=None):
        raise NotImplementedError("this adapter does not support continuation")


@dataclass(frozen=True)
class ModelBackendSpec:
    backend_id: str
    adapter_api_version: int
    match_source: Callable[[dict], bool]
    prepare_source: Callable[..., dict]
    build_adapter: Callable[..., DecisionBackboneAdapter]
    capabilities: Callable[..., BackendCapabilities]
    training_layout: Callable[[DecisionBackboneAdapter], TrainingLayout]


# Family implementations own the concrete cache representation.
PrefixStateHandle = Any
