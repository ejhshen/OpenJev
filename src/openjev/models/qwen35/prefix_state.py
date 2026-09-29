"""Independent branches of the Transformers 5.12 Qwen hybrid cache."""

from copy import copy, deepcopy
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Qwen35PrefixState:
    cache: object
    length: int
    batch_size: int
    owner: str


def fork_prefix(state, branch_indices):
    indices = torch.as_tensor(branch_indices, dtype=torch.long)
    if indices.ndim != 1 or indices.numel() == 0 or torch.any(indices < 0) or torch.any(indices >= state.batch_size):
        raise ValueError("branch indices must be nonempty and within the parent batch")
    cache = copy(state.cache)
    for name, value in vars(state.cache).items():
        if name != "layers":
            setattr(cache, name, deepcopy(value))
    cache.layers = []
    device_indices = {}
    for layer in state.cache.layers:
        child = copy(layer)
        for name, value in vars(layer).items():
            if isinstance(value, torch.Tensor):
                if name in {"keys", "values", "conv_states", "recurrent_states"} and value.numel():
                    if value.shape[0] != state.batch_size:
                        raise ValueError(f"cache batch mismatch in {name}")
                    if value.device not in device_indices:
                        device_indices[value.device] = indices.to(value.device)
                    value = value.index_select(0, device_indices[value.device])
                else:
                    value = value.clone()
            else:
                value = deepcopy(value)
            setattr(child, name, value)
        if hasattr(child, "max_batch_size"):
            child.max_batch_size = indices.numel()
        cache.layers.append(child)
    return Qwen35PrefixState(cache, state.length, indices.numel(), state.owner)


def cache_tensors(state):
    for i, layer in enumerate(state.cache.layers):
        for name, value in vars(layer).items():
            if isinstance(value, torch.Tensor):
                yield f"{i}.{name}", value


def cache_bytes(state):
    return sum(t.numel() * t.element_size() for _, t in cache_tensors(state))
