"""Resolve a complete decision artifact from disk or the Hugging Face Hub."""
from pathlib import Path

def resolve_artifact(model, revision=None):
    path = Path(model).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(str(model), revision=revision, allow_patterns=[
        "backbone/*", "tokenizer/*", "decision_head.safetensors",
        "openjev_config.json", "calibration.json", "manifest.json", "COMPLETE",
    ])
