"""Versioned model artifacts, with explicit completeness and file hashes."""

import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finish_manifest(output, metadata):
    output = Path(output)
    files = {str(p.relative_to(output)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
             for p in sorted(output.rglob("*")) if p.is_file() and p.name not in {"manifest.json", "COMPLETE"}}
    manifest = {"artifact_format_version": 1, **metadata, "files": files}
    write_json(output / "manifest.json", manifest)
    (output / "COMPLETE").write_text(sha256_file(output / "manifest.json") + "\n")
    return manifest


def load_manifest(path, *, verify_hashes=False):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if manifest.get("artifact_format_version") != 1:
        raise ValueError("unsupported artifact format")
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"incomplete artifact: {path}")
    if (path / "COMPLETE").read_text().strip() != sha256_file(path / "manifest.json"):
        raise ValueError("manifest completeness hash mismatch")
    for name, info in manifest["files"].items():
        entry = path / name
        if not entry.is_relative_to(path) or ".." in Path(name).parts:
            raise ValueError(f"invalid artifact path: {name}")
        if not entry.is_file() or entry.stat().st_size != info["bytes"]:
            raise ValueError(f"artifact file missing or wrong size: {name}")
        if verify_hashes and sha256_file(entry) != info["sha256"]:
            raise ValueError(f"artifact hash mismatch: {name}")
    return manifest


def save_decision_artifact(model, tokenizer, config, output, *, stage, provenance=None, temperature=1.0):
    from safetensors.torch import save_file
    import math
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if stage == "calibrated-serving":
        raise ValueError("use calibration.export to bind calibration to model weights and held-out data")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model.adapter.backbone.save_pretrained(output / "backbone", safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output / "tokenizer")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()},
              str(output / "decision_head.safetensors"))
    write_json(output / "openjev_config.json", config.to_dict())
    write_json(output / "calibration.json", {"temperature": temperature, "fitted": stage == "calibrated-serving"})
    return finish_manifest(output, {"stage": stage, "backend_id": config.backend_id,
                                    "adapter_api_version": config.adapter_api_version,
                                    "provenance": provenance or {}})


def load_calibration(path, manifest):
    """Read small calibration metadata and verify its weight/data bindings."""
    path = Path(path)
    entry = manifest["files"].get("calibration.json")
    if entry is None or sha256_file(path / "calibration.json") != entry["sha256"]:
        raise ValueError("calibration metadata hash mismatch")
    calibration = read_json(path / "calibration.json")
    temperature = calibration.get("temperature")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    if manifest["stage"] == "calibrated-serving":
        model = calibration.get("model", {})
        data = calibration.get("data", {})
        fit = calibration.get("fit", {})
        provenance = manifest.get("provenance", {})
        weights = {name: info for name, info in manifest["files"].items() if name.endswith(".safetensors")}
        if not weights or model.get("weight_files") != weights:
            raise ValueError("calibration/model weight binding mismatch")
        if calibration.get("fitted") is not True or fit.get("temperature") != temperature:
            raise ValueError("serving artifact requires a fitted, consistent temperature")
        if (not model.get("artifact_manifest_sha256")
                or model["artifact_manifest_sha256"] != provenance.get("base_artifact_manifest_sha256")
                or not data.get("manifest_sha256")
                or data["manifest_sha256"] != provenance.get("calibration_data_manifest_sha256")
                or data.get("split") != "calibration"):
            raise ValueError("calibration provenance binding mismatch")
        from openjev.compute import LEGACY_COMPUTE, validate_compute_contract
        declared = validate_compute_contract(read_json(path / "openjev_config.json").get("compute_contract", LEGACY_COMPUTE))
        if model.get("compute_contract", LEGACY_COMPUTE) != declared:
            raise ValueError("calibration/model compute contract mismatch")
        if declared != LEGACY_COMPUTE and calibration.get("execution", {}).get("compute_contract") != declared:
            raise ValueError("calibration execution compute contract mismatch")
        if (model.get("backend_id") != manifest["backend_id"]
                or model.get("adapter_api_version") != manifest["adapter_api_version"]
                or model.get("model_config_sha256") != sha256_file(path / "openjev_config.json")):
            raise ValueError("calibration/model configuration mismatch")
    return calibration


def load_decision_artifact(path, *, device="cpu", dtype="bfloat16", verify_hashes=False):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from openjev.config import OpenJevConfig
    from openjev.models.registry import get_backend
    from openjev.models.decision.head import DecisionHead
    from openjev.models.decision.model import OpenJevDecisionModel
    path = Path(path)
    manifest = load_manifest(path, verify_hashes=verify_hashes)
    if manifest["stage"] == "text-prepared":
        raise ValueError("prepared backbone has no decision head; run jevify")
    calibration = load_calibration(path, manifest)
    config = OpenJevConfig.from_dict(read_json(path / "openjev_config.json"))
    if (config.backend_id, config.adapter_api_version) != (manifest["backend_id"], manifest["adapter_api_version"]):
        raise ValueError("config/manifest backend mismatch")
    if config.kernel_backend == "fla":
        from openjev.models.qwen35.kernels import configure_upstream_fla
        configure_upstream_fla()
    adapter = get_backend(config.backend_id, config.adapter_api_version).build_adapter(path, device=device, dtype=dtype)
    if adapter.get_hidden_size() != config.hidden_size:
        raise ValueError("head/backbone hidden size mismatch")
    head = DecisionHead(config.hidden_size, **config.head).to(device=device)
    head.load_state_dict(load_file(str(path / "decision_head.safetensors")), strict=True)
    model = OpenJevDecisionModel(adapter, head)
    model.openjev_config = config
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    return model, tokenizer, config, calibration
