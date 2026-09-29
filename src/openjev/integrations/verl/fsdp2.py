"""Bind the family TrainingLayout to the installed verl FSDP2 implementation."""

from __future__ import annotations

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from openjev.models.registry import get_backend
from openjev.compute import BF16_BACKBONE_FP32_HEAD, model_compute_contract

_output_registered = False


def configure_fsdp2(model, *, gradient_checkpointing: bool = True,
                    compute_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                    reshard_after_forward: bool = True, checkpoint_stride: int = 1):
    from verl.utils.fsdp_utils import apply_fsdp2

    global _output_registered
    if not _output_registered:
        from torch.utils._pytree import register_dataclass
        from openjev.models.decision.schema import DecisionOutput

        register_dataclass(DecisionOutput)
        _output_registered = True

    config = model.openjev_config
    spec = get_backend(config.backend_id, config.adapter_api_version)
    layout = spec.training_layout(model.adapter)
    if not layout.decoder_blocks:
        raise ValueError("backend supplies no decoder blocks for FSDP2")
    decoder_classes = {type(block).__name__ for block in layout.decoder_blocks}
    checkpoint_ids = {id(block) for block in layout.checkpoint_blocks[::checkpoint_stride]}
    if not checkpoint_ids.issubset({id(module) for module in model.modules()}):
        raise ValueError("TrainingLayout contains modules outside the decision model")
    # Exactly one backbone forward per microstep on every rank. Differing K is
    # allowed; differing numbers of branch-chunk collectives are not.
    model.branch_microbatch_size = None
    model.checkpoint_branches = False
    if gradient_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
        )
        from functools import partial

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda module: id(module) in checkpoint_ids,
        )
    import torch.distributed as dist

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    contract = model_compute_contract(model)
    kwargs = {"mesh": mesh, "mp_policy": MixedPrecisionPolicy(
        param_dtype=compute_dtype, reduce_dtype=reduce_dtype,
        # FSDP-only activation casts would differ from the exported backbone:
        # its residual streams can be FP32 even with BF16 weights + autocast.
        cast_forward_inputs=contract != BF16_BACKBONE_FP32_HEAD,
    ), "reshard_after_forward": reshard_after_forward}
    if contract == BF16_BACKBONE_FP32_HEAD:
        if compute_dtype != torch.bfloat16 or reduce_dtype != torch.float32:
            raise ValueError("explicit compute contract requires BF16 backbone and FP32 reduction")
        from torch.distributed.fsdp import fully_shard
        # Shard the head separately before the installed helper shards decoder,
        # embedding and root groups. Root excludes this already-managed group.
        fully_shard(model.head, mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32,
                                                   cast_forward_inputs=False),
                    reshard_after_forward=reshard_after_forward)
        classes = sorted(decoder_classes)
    else:
        # Preserve the original grouping and compute precision for old resumes.
        classes = sorted(decoder_classes | {type(model.head).__name__})
    apply_fsdp2(model, kwargs, {"wrap_policy": {"transformer_layer_cls_to_wrap": classes}})
    return model, {"strategy": "fsdp2", "decoder_classes": sorted(decoder_classes),
                   "head_wrap_class": type(model.head).__name__, "gradient_checkpointing": gradient_checkpointing, "checkpoint_stride": checkpoint_stride,
                   "compute_dtype": str(compute_dtype), "reduce_dtype": str(reduce_dtype),
                   "compute_contract": contract,
                   "head_compute_dtype": str(torch.float32 if contract == BF16_BACKBONE_FP32_HEAD else compute_dtype),
                   "cast_forward_inputs": contract != BF16_BACKBONE_FP32_HEAD,
                   "reshard_after_forward": reshard_after_forward, "branch_microbatch_size": None, "mesh_size": dist.get_world_size()}
