"""Select verified upstream kernels without replacing their implementation."""
import importlib.metadata

def configure_upstream_fla():
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    if importlib.metadata.version('fla-core') != '0.5.2' or qwen.chunk_gated_delta_rule is None:
        raise RuntimeError('this execution recipe requires the isolated fla-core==0.5.2 overlay')
    # Keep norm and convolution unchanged while validating the delta path independently.
    qwen.FusedRMSNormGated = None
    return {'delta': 'fla-core-0.5.2', 'norm': 'transformers', 'conv': 'torch'}
