"""Arithmetic contract boundaries, backward flow and legacy compatibility."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from openjev.compute import (BF16_BACKBONE_FP32_HEAD, LEGACY_COMPUTE,
                             effective_training_compute_contract, validate_inference_compute)
from openjev.config import OpenJevConfig

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from openjev.models.decision.model import OpenJevDecisionModel
    from openjev.models.decision.head import DecisionHead


class ComputeConfigurationTests(unittest.TestCase):
    def test_missing_contract_keeps_legacy_and_bad_contract_is_rejected(self):
        self.assertEqual(OpenJevConfig.from_dict({}).compute_contract, LEGACY_COMPUTE)
        # Preserve the original positional constructor: backend, API, hidden size.
        self.assertEqual(OpenJevConfig("qwen35", 1, 48).hidden_size, 48)
        with self.assertRaisesRegex(ValueError, "compute contract"):
            OpenJevConfig(compute_contract="unknown")
        new = OpenJevConfig(compute_contract=BF16_BACKBONE_FP32_HEAD)
        self.assertEqual(OpenJevConfig.from_dict(new.to_dict()), new)
        self.assertEqual(validate_inference_compute(new, "bfloat16"), BF16_BACKBONE_FP32_HEAD)
        with self.assertRaisesRegex(ValueError, "BF16 backbone"):
            validate_inference_compute(new, "float32")
        validate_inference_compute(OpenJevConfig(), "float32")

    def test_training_inherits_or_explicitly_overrides_artifact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "openjev_config.json"
            path.write_text(json.dumps({}))
            recipe = SimpleNamespace(model_artifact=directory, compute_contract=None)
            self.assertEqual(effective_training_compute_contract(recipe), LEGACY_COMPUTE)
            recipe.compute_contract = BF16_BACKBONE_FP32_HEAD
            self.assertEqual(effective_training_compute_contract(recipe), BF16_BACKBONE_FP32_HEAD)
            recipe.compute_contract = None
            path.write_text(json.dumps({"compute_contract": BF16_BACKBONE_FP32_HEAD}))
            self.assertEqual(effective_training_compute_contract(recipe), BF16_BACKBONE_FP32_HEAD)


@unittest.skipUnless(HAS_TORCH, "torch is unavailable")
class ComputeScopeTests(unittest.TestCase):
    def make_model(self, contract):
        class Adapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(32, 12)
                self.projection = nn.Linear(12, 12)
                self.observed_dtype = None
                self.observed_autocast = None
            def get_hidden_size(self):
                return 12
            def forward_hidden(self, ids, mask, positions=None):
                self.observed_autocast = torch.is_autocast_enabled("cpu")
                value = self.projection(self.embedding(ids))
                self.observed_dtype = value.dtype
                return value
        torch.manual_seed(47)
        model = OpenJevDecisionModel(Adapter(), DecisionHead(12, set_dim=16, set_layers=1, set_heads=4, ffn_dim=24))
        model.openjev_config = OpenJevConfig(hidden_size=12, compute_contract=contract)
        batch = {"input_ids": torch.tensor([[1,2,3],[1,2,4],[1,2,5]]),
                 "attention_mask": torch.ones(3,3,dtype=torch.bool),
                 "question_end": torch.tensor([1,1,1]), "option_end": torch.tensor([2,2,2]),
                 "decision_ptr": torch.tensor([0,3]), "option_mask": torch.ones(1,3,dtype=torch.bool)}
        return model, batch

    def test_canonical_backbone_amp_head_fp32_and_outer_context_invariance(self):
        model, batch = self.make_model(BF16_BACKBONE_FP32_HEAD)
        model.eval()
        head_dtypes = []
        hook = model.head.question_projection.register_forward_hook(lambda _m, _i, output: head_dtypes.append(output.dtype))
        with torch.no_grad():
            a = model(batch)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                b = model(batch)
        hook.remove()
        self.assertTrue(model.adapter.observed_autocast)
        self.assertEqual(model.adapter.observed_dtype, torch.bfloat16)
        self.assertEqual(head_dtypes, [torch.float32, torch.float32])
        torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
        torch.testing.assert_close(a.probabilities, b.probabilities, rtol=0, atol=0)

    def test_canonical_full_model_gradients_remain_connected_and_fp32_stored(self):
        model, batch = self.make_model(BF16_BACKBONE_FP32_HEAD)
        output = model(batch)
        (-output.log_probs[0, 1]).backward()
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            if name.endswith("weight"):
                self.assertGreater(float(parameter.grad.norm()), 0, name)

    def test_legacy_keeps_caller_controlled_context(self):
        model, batch = self.make_model(LEGACY_COMPUTE)
        model.eval()
        head_dtypes = []
        hook = model.head.question_projection.register_forward_hook(lambda _m, _i, output: head_dtypes.append(output.dtype))
        with torch.no_grad():
            model(batch)
            self.assertFalse(model.adapter.observed_autocast)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                model(batch)
        hook.remove()
        self.assertEqual(head_dtypes, [torch.float32, torch.bfloat16])


if __name__ == "__main__":
    unittest.main()
