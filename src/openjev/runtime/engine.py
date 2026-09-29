"""Decision serving with shared-prefix execution and an expanded reference mode."""

from pathlib import Path
from threading import Lock
import time

import torch

from openjev.api.typesafe import InvalidDecisionRequest, format_answer, parse_request
from openjev.artifacts import load_decision_artifact, sha256_file
from openjev.compute import validate_inference_compute
from openjev.data.collator import collate_serialized
from openjev.models.decision.serializer import DecisionSerializer
from openjev.models.registry import get_backend
from .prefix import predict_shared_prefix


class DecisionEngine:
    def __init__(self, artifact, *, device="cuda:0", dtype="bfloat16", alias="OpenJev-4B",
                 branch_microbatch_size=8, max_request_tokens=65536, execution_mode="auto"):
        if execution_mode not in {"auto", "expanded", "shared_prefix"}:
            raise ValueError("execution_mode must be auto, expanded or shared_prefix")
        if branch_microbatch_size < 1:
            raise ValueError("branch_microbatch_size must be positive")
        from openjev.runtime.hub import resolve_artifact
        artifact = resolve_artifact(artifact)
        self.device = torch.device(device)
        self.model, self.tokenizer, self.config, calibration = load_decision_artifact(artifact, device=device, dtype=dtype)
        self.compute_contract = validate_inference_compute(self.config, dtype)
        capabilities = get_backend(self.config.backend_id, self.config.adapter_api_version).capabilities()
        supports_prefix = (capabilities.causal_prefix and capabilities.prefix_reuse
                           and capabilities.fork and capabilities.multi_token_continuation)
        if execution_mode == "shared_prefix" and not supports_prefix:
            raise ValueError(f"backend {self.config.backend_id!r} does not support shared-prefix inference")
        self.execution_mode = ("shared_prefix" if supports_prefix else "expanded") if execution_mode == "auto" else execution_mode
        self.model.eval()
        self.model.branch_microbatch_size = branch_microbatch_size
        self.temperature = float(calibration["temperature"])
        args = dict(self.config.serializer)
        args.pop("version")
        self.serializer = DecisionSerializer(self.tokenizer, **args)
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        self.model_id = "openjev-" + sha256_file(Path(artifact) / "manifest.json")[:16]
        self.alias = alias
        self.max_request_tokens = max_request_tokens
        self.lock = Lock()

    @torch.inference_mode()
    def predict(self, body):
        try:
            questions = parse_request(body, model_ids={self.alias, self.model_id})
            serialized = [self.serializer(question.decision) for question in questions]
        except (ValueError, TypeError) as exc:
            raise InvalidDecisionRequest(str(exc)) from exc
        # Keep expanded_input_tokens as the unpadded expanded reference budget.
        # executed_input_tokens includes any padding actually sent to the model.
        logical_tokens = len(serialized[0].state_ids) + sum(item.logical_token_count - len(item.state_ids) for item in serialized)
        if logical_tokens > self.max_request_tokens:
            raise InvalidDecisionRequest(f"request uses {logical_tokens} tokens, exceeds {self.max_request_tokens}; truncation is disabled")
        answers = {}
        expanded_tokens = sum(len(branch.input_ids) for item in serialized for branch in item.branches)
        executed_tokens = 0
        with self.lock:
            start = time.perf_counter()
            if self.execution_mode == "shared_prefix":
                outputs, executed_tokens = predict_shared_prefix(
                    self.model, serialized, device=self.device,
                    branch_microbatch_size=self.model.branch_microbatch_size,
                    pad_token_id=self.pad_id,
                    temperature=self.temperature,
                )
            else:
                outputs = []
                for encoded in serialized:
                    batch = collate_serialized([encoded], self.pad_id)
                    batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                    outputs.append(self.model(batch, temperature=self.temperature))
                    executed_tokens += batch["input_ids"].numel()
            for question, output in zip(questions, outputs):
                probabilities = output.probabilities[0].cpu().tolist()
                answers[question.decision.id] = format_answer(question, probabilities)
            elapsed = time.perf_counter() - start
        return {"model": self.model_id, "answers": answers,
                "usage": {"input_tokens": logical_tokens, "output_tokens": 0,
                          "expanded_input_tokens": expanded_tokens,
                          "executed_input_tokens": executed_tokens,
                          "reused_prefix_tokens": expanded_tokens - logical_tokens if self.execution_mode == "shared_prefix" else 0},
                "runtime": {"execution": self.execution_mode, "cache_hit": False,
                            "prefix_reuse": self.execution_mode == "shared_prefix",
                            "state_prefills": 1 if self.execution_mode == "shared_prefix" else 0,
                            "seconds": elapsed}}
