"""Family bindings stay separate from the decision architecture."""

from openjev.models.base import BackendCapabilities, ModelBackendSpec, TrainingLayout


def match_source(config):
    if config.get("model_type") == "qwen3_5_text":
        text = config
    elif config.get("model_type") == "qwen3_5":
        text = config.get("text_config", {})
    else:
        return False
    return text.get("model_type") == "qwen3_5_text" and text.get("hidden_size", 0) > 0


def prepare_source(*args, **kwargs):
    from .extract_text import prepare_text
    return prepare_text(*args, **kwargs)


def build_adapter(*args, **kwargs):
    from .adapter import Qwen35Adapter
    return Qwen35Adapter.from_prepared(*args, **kwargs)


def capabilities(*, verified_prefix=True):
    return BackendCapabilities(prefix_reuse=verified_prefix, fork=verified_prefix,
                               multi_token_continuation=verified_prefix)


def training_layout(adapter):
    blocks = tuple(adapter.backbone.layers)
    return TrainingLayout(blocks, blocks)


BACKEND_SPEC = ModelBackendSpec("qwen35", 1, match_source, prepare_source,
                               build_adapter, capabilities, training_layout)
