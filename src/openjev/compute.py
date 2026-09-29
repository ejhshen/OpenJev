"""Versioned model arithmetic, shared by training and inference.

Legacy artifacts keep caller-controlled autocast. New contracts are opt-in and
bound by the artifact configuration hash, including calibration provenance.
"""

from contextlib import nullcontext
import json
from pathlib import Path

LEGACY_COMPUTE = "legacy-v0"
BF16_BACKBONE_FP32_HEAD = "bf16-backbone-fp32-head-v1"


def validate_compute_contract(value):
    if value not in (LEGACY_COMPUTE, BF16_BACKBONE_FP32_HEAD):
        raise ValueError(f"unsupported compute contract: {value!r}")
    return value


def model_compute_contract(model):
    config = getattr(model, "openjev_config", None)
    return validate_compute_contract(getattr(config, "compute_contract", LEGACY_COMPUTE))


def compute_context(contract, scope, device):
    validate_compute_contract(contract)
    if scope not in ("backbone", "head"):
        raise ValueError("compute scope must be backbone or head")
    if contract == LEGACY_COMPUTE:
        return nullcontext()
    import torch

    device_type = torch.device(device).type
    if device_type not in ("cuda", "cpu"):
        raise ValueError("the explicit compute contract supports CUDA and CPU only")
    return torch.autocast(device_type, dtype=torch.bfloat16, enabled=scope == "backbone")


def effective_training_compute_contract(recipe):
    declared = getattr(recipe, "compute_contract", None)
    if declared is not None:
        return validate_compute_contract(declared)
    config = json.loads((Path(recipe.model_artifact) / "openjev_config.json").read_text())
    return validate_compute_contract(config.get("compute_contract", LEGACY_COMPUTE))


def validate_inference_compute(config, dtype):
    contract = validate_compute_contract(config.compute_contract)
    if contract == BF16_BACKBONE_FP32_HEAD and str(dtype).removeprefix("torch.") != "bfloat16":
        raise ValueError("bf16-backbone-fp32-head-v1 inference requires BF16 backbone loading")
    return contract
