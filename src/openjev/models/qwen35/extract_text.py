"""Extract Qwen3.5 text tensors without constructing the multimodal model."""

import shutil
from collections import defaultdict
from pathlib import Path

from openjev.artifacts import finish_manifest, read_json, sha256_file, write_json


def prepare_text(source, output, *, revision=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty: {output}")
    config = read_json(source / "config.json")
    full = config.get("model_type") == "qwen3_5"
    text_config = config["text_config"] if full else config
    if text_config.get("model_type") != "qwen3_5_text":
        raise ValueError("source is not a Qwen3.5 text model")
    prefix = "model.language_model." if full else ""
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        source_map = read_json(index_path)["weight_map"]
    else:
        with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as f:
            source_map = {name: "model.safetensors" for name in f.keys()}
    groups = defaultdict(list)
    inventory = {}
    for key, filename in sorted(source_map.items()):
        mapped = key.removeprefix(prefix)
        if key.startswith(prefix) and (mapped == "embed_tokens.weight" or mapped == "norm.weight" or mapped.startswith("layers.")):
            groups[filename].append((key, mapped))
            inventory[key] = {"action": "retained", "target": mapped}
        elif key == "lm_head.weight" and text_config.get("tie_word_embeddings", config.get("tie_word_embeddings", False)):
            inventory[key] = {"action": "alias", "target": "embed_tokens.weight"}
        elif key.startswith(("model.visual.", "visual.", "model.mtp.", "mtp.", "lm_head.")):
            inventory[key] = {"action": "excluded", "reason": "non-text-decision output or vision/MTP"}
        else:
            raise ValueError(f"unmapped source tensor: {key}")
    retained = {item["target"] for item in inventory.values() if item["action"] == "retained"}
    if "embed_tokens.weight" not in retained or "norm.weight" not in retained:
        raise ValueError("source lacks required embedding/norm")
    for layer in range(text_config["num_hidden_layers"]):
        if not any(name.startswith(f"layers.{layer}.") for name in retained):
            raise ValueError(f"source lacks decoder layer {layer}")
    (output / "backbone").mkdir(parents=True)
    (output / "tokenizer").mkdir()
    out_map, total_bytes, parameters = {}, 0, 0
    for i, (filename, mapping) in enumerate(sorted(groups.items()), 1):
        tensors = {}
        with safe_open(str(source / filename), framework="pt", device="cpu") as f:
            for original, target in mapping:
                tensor = f.get_tensor(original)
                tensors[target] = tensor
                total_bytes += tensor.numel() * tensor.element_size()
                parameters += tensor.numel()
                inventory[original].update(shape=list(tensor.shape), dtype=str(tensor.dtype))
        out_name = f"model-{i:05d}-of-{len(groups):05d}.safetensors"
        save_file(tensors, str(output / "backbone" / out_name), metadata={"format": "pt"})
        out_map.update({name: out_name for name in tensors})
        del tensors
    # If present, tied aliases must match the retained input embedding.
    aliases = [k for k, item in inventory.items() if item["action"] == "alias"]
    embedding_key = prefix + "embed_tokens.weight"
    for key in aliases:
        with safe_open(str(source / source_map[key]), framework="pt", device="cpu") as f:
            alias = f.get_tensor(key)
        with safe_open(str(source / source_map[embedding_key]), framework="pt", device="cpu") as f:
            embedding = f.get_tensor(embedding_key)
        if not torch.equal(alias, embedding):
            raise ValueError(f"declared tied embedding differs: {key}")
        del alias, embedding
    clean = Qwen3_5TextConfig(**text_config)
    clean.architectures = ["Qwen3_5TextModel"]
    clean.save_pretrained(output / "backbone")
    write_json(output / "backbone" / "model.safetensors.index.json", {"metadata": {"total_size": total_bytes}, "weight_map": out_map})
    tokenizer_files = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja"}
    for name in tokenizer_files:
        if (source / name).is_file():
            shutil.copyfile(source / name, output / "tokenizer" / name)
    if not (output / "tokenizer" / "tokenizer_config.json").is_file():
        raise ValueError("source tokenizer_config.json missing")
    write_json(output / "source_config.json", config)
    write_json(output / "tensor_inventory.json", inventory)
    source_manifest = source / "openjev_source_manifest.json"
    return finish_manifest(output, {"stage": "text-prepared", "backend_id": "qwen35", "adapter_api_version": 1,
                                    "source": {"path": str(source), "revision": revision, "config_sha256": sha256_file(source / "config.json"),
                                               "manifest_sha256": sha256_file(source_manifest) if source_manifest.exists() else None},
                                    "parameter_count": parameters, "weight_bytes": total_bytes,
                                    "retained_tensors": len(out_map), "source_tensors": len(source_map)})
