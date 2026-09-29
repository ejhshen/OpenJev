"""Explicit eager-mode experiment preserving FP32 DeltaNet recurrent state.

The native Transformers cache chooses recurrent storage from the convolution
dtype. This variant preserves the kernel's FP32 result at its first write; it
does not recover precision by upcasting an already stored BF16 state.
"""


def make_fp32_recurrent_cache(config):
    import torch
    from transformers.cache_utils import DynamicCache, LinearAttentionLayer

    class FP32RecurrentLayer(LinearAttentionLayer):
        is_compileable = False

        def lazy_initialization(self, conv_states=None, recurrent_states=None):
            if conv_states is not None:
                super().lazy_initialization(conv_states=conv_states)
            if recurrent_states is not None:
                self.recurrent_states = torch.zeros_like(recurrent_states, dtype=torch.float32)
                self.is_recurrent_states_initialized = True

    cache = DynamicCache(config=config)
    for index, layer in enumerate(cache.layers):
        if type(layer) is LinearAttentionLayer:
            cache.layers[index] = FP32RecurrentLayer(config)
    return cache
