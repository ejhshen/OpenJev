"""Installed verl sharded checkpoints plus decision-stream completeness metadata."""

from __future__ import annotations

from pathlib import Path

import torch.distributed as dist

from openjev.artifacts import finish_manifest, load_manifest, read_json, sha256_file, write_json
from openjev.compute import LEGACY_COMPUTE, effective_training_compute_contract


def training_contract(recipe, *, rlcd: bool = False):
    """One contract builder shared by training, resume, and artifact export."""
    keys = ("global_decision_batch", "microbatch_decisions", "backbone_lr", "head_lr", "weight_decay",
            "warmup_steps", "max_grad_norm", "seed", "max_branch_length", "max_decision_tokens",
            "pad_to_multiple_of", "gradient_checkpointing", "compute_dtype", "reduce_dtype", "backbone_storage_dtype")
    value = {"model_manifest_sha256": sha256_file(Path(recipe.model_artifact) / "manifest.json"),
             "dataset_manifest_sha256": sha256_file(recipe.train_manifest), "train_split": recipe.train_split,
             "world_size": dist.get_world_size(), **{key: getattr(recipe, key) for key in keys}}
    compute_contract = effective_training_compute_contract(recipe)
    if compute_contract != LEGACY_COMPUTE:
        value["compute_contract"] = compute_contract
    for key, default in [('checkpoint_stride', 1), ('branch_batch_multiple', 1), ('kernel_backend', 'default'), ('reshard_after_forward', True),
                         ('retain_parameters_between_microsteps', False), ('defer_gradient_sync', False)]:
        if getattr(recipe, key, default) != default:
            value[key] = getattr(recipe, key)
    if recipe.microbatch_decisions != 1:
        value['max_padded_tokens_per_gpu'] = recipe.max_padded_tokens_per_gpu
    if rlcd:
        value.update(stage="decision-rlcd", oracle_sha256=sha256_file(recipe.oracle_path))
        value.update({key: getattr(recipe, key) for key in
                      ("algorithm", "estimator", "alpha", "epsilon", "group_size", "baseline", "ppo_epochs", "ppo_clip_epsilon")})
    return value


class DecisionCheckpointManager:
    def __init__(self, engine, *, training_contract: dict):
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

        self.engine = engine
        self.training_contract = training_contract
        self.loaded_extra_state = None
        self.manager = FSDPCheckpointManager(
            model=engine.model, optimizer=engine.optimizer,
            lr_scheduler=engine.scheduler, processing_class=engine.tokenizer,
            checkpoint_contents=["model", "optimizer", "extra"],
        )

    def save(self, path: str | Path, *, step: int, sampler_state: dict, extra_state: dict | None = None):
        path = Path(path)
        status = [None]
        if dist.get_rank() == 0:
            if path.exists() and any(path.iterdir()):
                status[0] = f"refusing to overwrite checkpoint: {path}"
            else:
                path.mkdir(parents=True, exist_ok=True)
        dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise FileExistsError(status[0])
        dist.barrier()
        self.manager.save_checkpoint(str(path), global_step=step, max_ckpt_to_keep=None)
        if extra_state is not None:
            write_json(path / f"openjev_rank_state_{dist.get_rank():04d}.json", extra_state)
        dist.barrier()
        status = [None]
        if dist.get_rank() == 0:
            try:
                write_json(path / "openjev_training_state.json", {
                    "format_version": 1, "global_step": step, "world_size": dist.get_world_size(),
                    "sampler": sampler_state, "training_contract": self.training_contract,
                    "optimizer_state_dtypes": self.engine.optimizer_dtype_summary(),
                })
                finish_manifest(path, {"stage": "training-checkpoint", "global_step": step,
                                       "world_size": dist.get_world_size(), "backend_id": self.engine.config.backend_id,
                                       "adapter_api_version": self.engine.config.adapter_api_version})
            except Exception as failure:
                status[0] = f"{type(failure).__name__}: {failure}"
        dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError("checkpoint finalization failed: " + status[0])
        dist.barrier()
        return path

    def load(self, path: str | Path, sampler) -> int:
        path = Path(path)
        # Rank zero verifies every shard once; all ranks receive the validated
        # metadata or the same error before entering checkpoint collectives.
        payload = [None]
        if dist.get_rank() == 0:
            try:
                manifest = load_manifest(path, verify_hashes=True)
                state = read_json(path / "openjev_training_state.json")
                if manifest["stage"] != "training-checkpoint" or state.get("format_version") != 1:
                    raise ValueError("not a supported decision training checkpoint")
                if state["world_size"] != dist.get_world_size():
                    raise ValueError("this per-rank checkpoint requires the original world size")
                if state["training_contract"] != self.training_contract:
                    raise ValueError("checkpoint training contract differs from this recipe/data/artifact")
                payload[0] = {"state": state}
            except Exception as error:
                payload[0] = {"error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(payload, src=0)
        if "error" in payload[0]:
            raise ValueError(payload[0]["error"])
        state = payload[0]["state"]
        sampler.load_state_dict(state["sampler"])
        self.manager.load_checkpoint(str(path), del_local_after_load=False)
        extra_path = path / f"openjev_rank_state_{dist.get_rank():04d}.json"
        self.loaded_extra_state = read_json(extra_path) if extra_path.is_file() else None
        dist.barrier()
        return int(state["global_step"])
