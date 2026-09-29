"""Family bindings stay separate from the decision architecture."""

from openjev.models.base import BackendCapabilities, ModelBackendSpec, TrainingLayout


def build_adapter(*args, **kwargs):
    from .adapter import Qwen35Adapter
    return Qwen35Adapter.from_prepared(*args, **kwargs)


def capabilities(*, verified_prefix=True):
    return BackendCapabilities(prefix_reuse=verified_prefix, fork=verified_prefix,
                               multi_token_continuation=verified_prefix)


def training_layout(adapter):
    blocks = tuple(adapter.backbone.layers)
    return TrainingLayout(blocks, blocks)


BACKEND_SPEC = ModelBackendSpec("qwen35", 1, build_adapter, capabilities, training_layout)
