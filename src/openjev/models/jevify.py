"""Assemble a pretrained backbone and a newly initialized decision head."""

from dataclasses import replace

from openjev.artifacts import load_manifest, save_decision_artifact
from openjev.config import OpenJevConfig
from openjev.models.registry import get_backend


def jevify(prepared, output, *, config=None, device="cpu", dtype="bfloat16"):
    from transformers import AutoTokenizer
    from openjev.models.decision.head import DecisionHead
    from openjev.models.decision.model import OpenJevDecisionModel
    from pathlib import Path
    manifest = load_manifest(prepared)
    if manifest["stage"] != "text-prepared":
        raise ValueError("jevify expects a prepared text backbone")
    spec = get_backend(manifest["backend_id"], manifest["adapter_api_version"])
    adapter = spec.build_adapter(prepared, device=device, dtype=dtype)
    config = config or OpenJevConfig(backend_id=spec.backend_id)
    if config.backend_id != spec.backend_id:
        raise ValueError("recipe and prepared backend do not match")
    # Record the effective initialization of a new artifact even when the
    # recipe omits it. Loading an existing artifact never takes this path.
    head_config = {"score_init_gain": 0.1, **config.head}
    config = replace(config, hidden_size=adapter.get_hidden_size(), head=head_config)
    head = DecisionHead(config.hidden_size, **config.head).to(device=device)
    model = OpenJevDecisionModel(adapter, head)
    model.openjev_config = config
    tokenizer = AutoTokenizer.from_pretrained(Path(prepared) / "tokenizer", local_files_only=True)
    return save_decision_artifact(model, tokenizer, config, output, stage="jevified-init",
                                  provenance={"prepared": str(prepared), "source": manifest.get("source")})
