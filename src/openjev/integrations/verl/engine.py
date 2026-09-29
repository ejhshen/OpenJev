"""Training facade: complete decisions, equal collective counts, exact weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist

from openjev.artifacts import load_decision_artifact
from openjev.compute import effective_training_compute_contract, validate_compute_contract
from openjev.data.collator import DecisionCollator, collate_serialized, pack_decisions, split_to_microsteps
from openjev.models.decision.serializer import DecisionSerializer
from openjev.training.supervised import decision_cross_entropy

from .actor import OpenJevDecisionActor
from .fsdp2 import configure_fsdp2
from .supervised import global_weight_denominator


@dataclass
class SFTRecipe:
    model_artifact: str
    train_manifest: str
    output_dir: str
    max_steps: int = 3
    train_split: str = "train"
    eval_manifest: str | None = None
    eval_split: str = "dev"
    global_decision_batch: int | None = None
    microbatch_decisions: int = 1
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    seed: int = 42
    max_branch_length: int = 2048
    max_decision_tokens: int = 32768
    max_padded_tokens_per_gpu: int | None = None
    pad_to_multiple_of: int = 1
    freeze_backbone: bool = False
    gradient_checkpointing: bool = True
    checkpoint_stride: int = 1
    checkpoint_every: int = 0
    checkpoint_at_end: bool = True
    eval_every: int = 0
    eval_max_decisions: int = 32
    log_every: int = 1
    strategy: str = "fsdp2"
    compute_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    backbone_storage_dtype: str = "float32"
    compute_contract: str | None = None

    branch_batch_multiple: int = 1
    token_cache: str | None = None
    kernel_backend: str = "default"
    reshard_after_forward: bool = True
    retain_parameters_between_microsteps: bool = False
    defer_gradient_sync: bool = False

    def validate(self, world_size: int) -> None:
        if self.compute_contract is not None:
            validate_compute_contract(self.compute_contract)
        if self.global_decision_batch is None:
            self.global_decision_batch = world_size
        if self.strategy != "fsdp2" or self.compute_dtype != "bfloat16" or self.reduce_dtype != "float32":
            raise ValueError("first validated strategy is fsdp2 with BF16 compute and FP32 reduction")
        if self.backbone_storage_dtype not in ("bfloat16", "float32"):
            raise ValueError("backbone_storage_dtype must be bfloat16 or float32")
        if self.branch_batch_multiple not in (1, 8):
            raise ValueError("branch_batch_multiple must be 1 or 8")
        if self.kernel_backend not in {"default", "fla"}:
            raise ValueError("unsupported kernel backend")
        for name in ("max_steps", "global_decision_batch", "microbatch_decisions", "branch_batch_multiple", "checkpoint_stride", "max_branch_length", "max_decision_tokens", "pad_to_multiple_of", "eval_max_decisions", "log_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.global_decision_batch % world_size:
            raise ValueError("global_decision_batch must be divisible by world_size")
        for name in ("warmup_steps", "checkpoint_every", "eval_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.max_padded_tokens_per_gpu is not None and self.max_padded_tokens_per_gpu < 1:
            raise ValueError("max_padded_tokens_per_gpu must be positive")
        for name in ("backbone_lr", "head_lr", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


class DecisionStream:
    """A global shuffled stream; ranks take disjoint equal slices of each update.

    An update may cross an epoch boundary. No label or option padding changes
    the number of real decisions. The identical cursor is checkpointed on all
    ranks; world size and batch size belong to the checkpoint contract.
    """

    def __init__(self, examples, *, seed: int, dataset_hash: str):
        if not examples:
            raise ValueError("training data is empty")
        self.examples, self.seed = examples, seed
        self.epoch, self.cursor, self.total_seen = 0, 0, 0
        self.fingerprint = hashlib.sha256(json.dumps([dataset_hash, [row.id for row in examples], seed]).encode()).hexdigest()
        self._order = self._shuffle()

    def _shuffle(self):
        order = list(range(len(self.examples)))
        random.Random(self.seed + self.epoch).shuffle(order)
        return order

    def next_global(self, count: int):
        indices = []
        while len(indices) < count:
            if self.cursor == len(self._order):
                self.epoch += 1
                self.cursor = 0
                self._order = self._shuffle()
            size = min(count - len(indices), len(self._order) - self.cursor)
            indices.extend(self._order[self.cursor:self.cursor + size])
            self.cursor += size
        self.total_seen += count
        return indices

    def state_dict(self):
        return {"format_version": 1, "fingerprint": self.fingerprint, "epoch": self.epoch,
                "cursor": self.cursor, "total_seen": self.total_seen}

    def load_state_dict(self, state):
        if state.get("format_version") != 1 or state.get("fingerprint") != self.fingerprint:
            raise ValueError("sampler checkpoint does not match the training dataset/order")
        if any(not isinstance(state.get(key), int) or state[key] < 0 for key in ("epoch", "cursor", "total_seen")):
            raise ValueError("invalid sampler checkpoint counters")
        if state["cursor"] > len(self.examples):
            raise ValueError("sampler cursor exceeds this dataset")
        self.epoch, self.cursor, self.total_seen = state["epoch"], state["cursor"], state["total_seen"]
        self._order = self._shuffle()


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=value.is_pinned()) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


class DecisionTrainingEngine:
    def __init__(self, recipe: SFTRecipe, device):
        self.recipe, self.device = recipe, device
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        if recipe.kernel_backend == 'fla':
            from openjev.models.qwen35.kernels import configure_upstream_fla
            configure_upstream_fla()
        model, tokenizer, config, _ = load_decision_artifact(recipe.model_artifact, device="cpu", dtype=recipe.backbone_storage_dtype)
        config = replace(config, compute_contract=effective_training_compute_contract(recipe), kernel_backend=recipe.kernel_backend)
        model.openjev_config = config
        model.branch_batch_multiple = recipe.branch_batch_multiple
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        if recipe.freeze_backbone:
            model.adapter.requires_grad_(False)
        self.parameter_dtypes = {}
        for parameter in model.parameters():
            key = str(parameter.dtype)
            self.parameter_dtypes[key] = self.parameter_dtypes.get(key, 0) + parameter.numel()
        self.model, self.fsdp_metadata = configure_fsdp2(model, gradient_checkpointing=recipe.gradient_checkpointing, reshard_after_forward=recipe.reshard_after_forward, checkpoint_stride=recipe.checkpoint_stride)
        if self.rank == 0:
            print(json.dumps({"event":"trainable_parameters","freeze_backbone":recipe.freeze_backbone,"backbone_trainable":sum(p.numel() for p in self.model.adapter.parameters() if p.requires_grad),"head_trainable":sum(p.numel() for p in self.model.head.parameters() if p.requires_grad)}),flush=True)
        self.tokenizer, self.config = tokenizer, config
        serializer_args = dict(config.serializer)
        serializer_args.pop("version", None)
        serializer_args.update(max_branch_length=recipe.max_branch_length, max_decision_tokens=recipe.max_decision_tokens)
        self.serializer = DecisionSerializer(tokenizer, **serializer_args)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.collator = DecisionCollator(self.serializer, pad_id, max_branch_length=recipe.max_branch_length, pad_to_multiple_of=recipe.pad_to_multiple_of)
        self.token_cache = None
        if recipe.token_cache:
            from openjev.data.token_cache import TokenCache
            self.token_cache = TokenCache(recipe.token_cache, self.serializer)
        self.optimizer = torch.optim.AdamW([
            {"params": [p for p in self.model.adapter.parameters() if p.requires_grad], "lr": 0.0 if recipe.freeze_backbone else recipe.backbone_lr, "name": "backbone"},
            {"params": list(self.model.head.parameters()), "lr": recipe.head_lr, "name": "head"},
        ], weight_decay=recipe.weight_decay)

        def multiplier(step):
            if recipe.warmup_steps and step < recipe.warmup_steps:
                return (step + 1) / recipe.warmup_steps
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, multiplier)
        self.actor = OpenJevDecisionActor(self.model, self.optimizer)

    def collate_cpu(self, examples):
        batch = self.collator(examples)
        budget = self.recipe.max_padded_tokens_per_gpu
        if budget is not None and batch["input_ids"].numel() > budget:
            raise ValueError("a complete decision exceeds max_padded_tokens_per_gpu; use another length/K bucket")
        return batch

    def prepare_microbatches(self, local_examples):
        """Validate all local microsteps before any rank enters model collectives."""
        batches, error = [], None
        try:
            batches = [self.collate_cpu([example]) for example in local_examples]
        except Exception as failure:
            error = f"rank {self.rank}: {type(failure).__name__}: {failure}"
        statuses = [None] * self.world_size
        dist.all_gather_object(statuses, {"error": error, "microsteps": len(local_examples)})
        errors = [status["error"] for status in statuses if status["error"] is not None]
        if errors:
            raise ValueError("collective microbatch validation failed: " + "; ".join(errors))
        if len({status["microsteps"] for status in statuses}) != 1:
            raise ValueError("ranks have different forward/backward microstep counts")
        return batches

    def validate_items(self, items):
        pass

    def prepare_groups(self, items):
        error = None; groups = []
        try:
            encode = self.token_cache or self.serializer
            self.validate_items(items)
            entries = [(role, row, encode(row.decision)) for role, row in items]
            # A single long decision may exceed the microbatch target, never the recipe's logical limit.
            groups = pack_decisions(entries, self.recipe.microbatch_decisions,
                                    self.recipe.max_padded_tokens_per_gpu,
                                    pad_multiple=self.recipe.pad_to_multiple_of)
        except Exception as exc:
            error = repr(exc)
        states = [None] * self.world_size
        dist.all_gather_object(states, {'error': error, 'groups': len(groups), 'decisions': len(items)})
        if any(x['error'] for x in states):
            raise RuntimeError(f'batch preparation failed: {states}')
        count = max(x['groups'] for x in states)
        if not count or any(x['decisions'] < count for x in states):
            raise ValueError('ranks need enough real decisions for the shared microstep count')
        return split_to_microsteps(groups, count)

    def collate_group(self, group):
        batch = collate_serialized([x[2] for x in group], self.collator.pad_token_id,
                                  max_branch_length=self.recipe.max_branch_length,
                                  pad_to_multiple_of=self.recipe.pad_to_multiple_of)
        if group[0][0] != 'primary':
            targets = torch.zeros_like(batch['option_mask'], dtype=torch.float32)
            for i, (_, row, _) in enumerate(group):
                targets[i, :len(row.decision.options)] = torch.tensor(row.target.vector(row.decision))
            batch['targets'] = targets
        batch['sample_weight'] = torch.tensor([x[1].sample_weight for x in group], dtype=torch.float32)
        return {k: v.pin_memory() if isinstance(v, torch.Tensor) and self.device.type == 'cuda' else v
                for k, v in batch.items()}

    def run_update(self, items, objective):
        """One shared optimizer update. objective returns a weighted decision-loss sum."""
        self.model.train(); torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device); start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function('openjev.prepare'):
            groups = self.prepare_groups(items)
            denominator = global_weight_denominator(torch.tensor([x[1].sample_weight for x in items], device=self.device))
        numerator = torch.zeros((), device=self.device, dtype=torch.float64)
        tokens = padded = physical = branches = 0
        for index, group in enumerate(groups):
            last = index == len(groups)-1
            self.model.set_requires_gradient_sync(last or not self.recipe.defer_gradient_sync)
            self.model.set_reshard_after_backward(last or not self.recipe.retain_parameters_between_microsteps)
            with torch.profiler.record_function('openjev.collate_h2d'):
                cpu = self.collate_group(group)
                tokens += sum(len(b.input_ids) for x in group for b in x[2].branches)
                padded += cpu['input_ids'].numel(); branches += len(cpu['input_ids'])
                physical += self.model.padded_branch_count(len(cpu['input_ids']))*cpu['input_ids'].shape[1]
                batch = move_batch(cpu, self.device)
            with torch.profiler.record_function('openjev.forward'):
                output = self.actor.forward(batch)
            with torch.profiler.record_function('openjev.objective'):
                loss_sum = objective(output, batch, group)
                numerator += loss_sum.detach().double()
            with torch.profiler.record_function('openjev.backward'):
                (loss_sum * (self.world_size / denominator).to(loss_sum.dtype)).backward()
        self.model.set_requires_gradient_sync(True)
        with torch.profiler.record_function('openjev.optimizer'):
            norm = self.actor.optimizer_step(self.recipe.max_grad_norm); self.scheduler.step()
        if self.recipe.freeze_backbone:
            assert not self.optimizer.param_groups[0]["params"]
            assert all(not p.requires_grad and p.grad is None for p in self.model.adapter.parameters())
        stats = torch.tensor([tokens, padded, branches, len(items), physical], device=self.device, dtype=torch.float64)
        dist.all_reduce(stats); dist.all_reduce(numerator)
        torch.cuda.synchronize(self.device)
        seconds = torch.tensor(time.perf_counter()-start, device=self.device, dtype=torch.float64)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        mem = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()], device=self.device)
        dist.all_reduce(mem, op=dist.ReduceOp.MAX)
        return {'nll': float(numerator/denominator), 'seconds': float(seconds),
                'global_decisions': int(stats[3]), 'decisions_per_second': float(stats[3]/seconds),
                'expanded_tokens': int(stats[0]), 'padded_tokens': int(stats[1]), 'branches': int(stats[2]),
                'padding_fraction': float(1-stats[0]/stats[1]), 'physical_branch_tokens': int(stats[4]),
                'physical_padding_fraction': float(1-stats[0]/stats[4]), 'microsteps_per_rank': len(groups),
                'zero_weight_microbatches_global': 0, 'max_allocated_gib': float(mem[0])/2**30,
                'max_reserved_gib': float(mem[1])/2**30, 'gradient_norm': float(norm),
                'backbone_lr': self.optimizer.param_groups[0]['lr'], 'head_lr': self.optimizer.param_groups[1]['lr']}

    def train_step(self, local_examples):
        def objective(output, batch, group):
            losses = decision_cross_entropy(output.log_probs, batch['targets'], batch['option_mask'], reduction='none')
            return (losses * batch['sample_weight']).sum()
        return self.run_update([('supervised', x) for x in local_examples], objective)

    @torch.no_grad()
    def evaluate(self, examples, *, prediction_path: Path | None = None):
        self.model.eval()
        count = min(len(examples), self.recipe.eval_max_decisions)
        if not count:
            raise ValueError("evaluation split is empty")
        total = torch.zeros(6, device=self.device, dtype=torch.float64)
        records = []
        items = []
        for j in range((count+self.world_size-1)//self.world_size):
            i = j*self.world_size+self.rank
            items.append(('supervised' if i < count else 'evaldummy', examples[i if i < count else 0]))
        groups = self.prepare_groups(items)
        for group in groups:
            batch = move_batch(self.collate_group(group), self.device)
            output = self.actor.forward(batch)
            losses = decision_cross_entropy(output.log_probs, batch['targets'], batch['option_mask'], reduction='none')
            predicted = output.probabilities.argmax(-1); target = batch['targets'].argmax(-1)
            brier = (output.probabilities-batch['targets']).square().sum(-1)
            valid = torch.tensor([x[0] != 'evaldummy' for x in group], device=self.device)
            losses = losses*valid; brier = brier*valid
            weights = batch['sample_weight']*valid
            total += torch.stack((losses.double().sum(), ((predicted==target)&valid).double().sum(),
                                  brier.double().sum(), valid.double().sum(),
                                  (losses.double()*weights).sum(), weights.double().sum()))
            probabilities = output.probabilities.cpu().tolist(); targets = batch['targets'].cpu().tolist()
            for i, (role, example, _) in enumerate(group):
                if role == 'evaldummy':
                    continue
                k = len(example.decision.options)
                records.append({'id': example.id, 'group_id': example.group_id,
                                'option_ids': list(batch['option_ids'][i]),
                                'probabilities': probabilities[i][:k], 'target_probabilities': targets[i][:k]})
        dist.all_reduce(total)
        if prediction_path is not None:
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            with prediction_path.open("w") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.model.train()
        return {"n": int(total[3].item()), "nll": (total[0] / total[3]).item(),
                "accuracy": (total[1] / total[3]).item(), "brier": (total[2] / total[3]).item(),
                "weighted_nll": (total[4] / total[5]).item() if total[5] > 0 else None}

    def optimizer_dtype_summary(self):
        result = {}
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    result.setdefault(key, set()).add(str(value.dtype))
        return {key: sorted(values) for key, values in result.items()}
