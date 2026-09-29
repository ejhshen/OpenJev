"""Exercise the actual engine with a causal model and deliberately mutable caches."""

import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from openjev.models.base import BackendCapabilities, DecisionBackboneAdapter
    from openjev.models.decision.head import DecisionHead
    from openjev.models.decision.model import OpenJevDecisionModel
    from openjev.runtime.engine import DecisionEngine

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        bos_token_id = 2

        def encode(self, text, **kwargs):
            return [3 + ord(c) % 125 for c in text]

    class MutableCausalAdapter(DecisionBackboneAdapter):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.register_buffer("embedding", torch.arange(128 * 8).reshape(128, 8).remainder(17).float())

        def get_hidden_size(self):
            return 8

        def forward_hidden(self, input_ids, attention_mask, position_ids=None):
            self.calls.append(("expanded", input_ids.clone()))
            values = self.embedding[input_ids] * attention_mask.unsqueeze(-1)
            return values.cumsum(1) / attention_mask.cumsum(1).clamp_min(1).unsqueeze(-1)

        def prefill(self, input_ids, attention_mask=None):
            self.calls.append(("prefill", input_ids.clone()))
            return {"sum": self.embedding[input_ids].sum(1), "length": input_ids.shape[1]}

        def fork_state(self, state, branch_indices):
            return {"sum": state["sum"][branch_indices].clone(), "length": state["length"]}

        def continue_from(self, state, suffix_tokens, lengths=None):
            self.calls.append(("continue", suffix_tokens.clone()))
            sums = self.embedding[suffix_tokens].cumsum(1) + state["sum"][:, None]
            count = torch.arange(1, suffix_tokens.shape[1] + 1) + state["length"]
            hidden = sums / count[None, :, None]
            state["sum"].copy_(sums[:, -1])
            state["length"] += suffix_tokens.shape[1]
            return hidden, state


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class RuntimePrefixTests(unittest.TestCase):
    def setUp(self):
        self.model = OpenJevDecisionModel(
            MutableCausalAdapter(), DecisionHead(8, set_dim=8, set_heads=2, ffn_dim=16, init_seed=31),
        ).eval()
        self.config = SimpleNamespace(
            compute_contract="legacy-v0", backend_id="test", adapter_api_version=1,
            serializer={"version": "test"},
        )
        self.body = {"state": "An invoice was paid twice; customer requests a refund.", "questions": {
            "route": {"type": "choice", "instructions": "Who handles this?", "criteria": {
                "billing": "Payments", "support": "Software", "sales": "New subscriptions", "other": None}},
            "paid": {"type": "noul", "instructions": "Is this about a payment?"},
            "urgency": {"type": "score", "instructions": "Assess urgency and deadline.",
                        "criteria": ["None", "Soon", "Immediately"]},
        }}

    def engine(self, mode="auto", supports_prefix=True, microbatch=2):
        caps = BackendCapabilities(prefix_reuse=supports_prefix, fork=supports_prefix,
                                   multi_token_continuation=supports_prefix)
        with patch("openjev.runtime.hub.resolve_artifact", side_effect=lambda x:x), patch("openjev.runtime.engine.load_decision_artifact", return_value=(
            self.model, Tokenizer(), self.config, {"temperature": 1.7},
        )), patch("openjev.runtime.engine.sha256_file", return_value="a" * 64), patch(
            "openjev.runtime.engine.get_backend", return_value=SimpleNamespace(capabilities=lambda: caps),
        ):
            return DecisionEngine("unused", device="cpu", execution_mode=mode, branch_microbatch_size=microbatch)

    def assert_answers_close(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            if isinstance(a[key], dict):
                self.assert_answers_close(a[key], b[key])
            elif isinstance(a[key], float):
                self.assertAlmostEqual(a[key], b[key], places=6)
            else:
                self.assertEqual(a[key], b[key])

    def test_shared_tree_matches_expanded_and_processes_each_segment_once(self):
        expected = self.engine("expanded").predict(self.body)
        self.model.adapter.calls.clear()
        actual = self.engine().predict(self.body)
        self.assert_answers_close(actual["answers"], expected["answers"])
        calls = self.model.adapter.calls
        self.assertEqual(sum(kind == "prefill" for kind, _ in calls), 1)
        self.assertFalse(any(kind == "expanded" for kind, _ in calls))
        self.assertEqual(sum(ids.numel() for _, ids in calls), actual["usage"]["executed_input_tokens"])
        self.assertGreaterEqual(actual["usage"]["executed_input_tokens"], actual["usage"]["input_tokens"])
        self.assertLess(actual["usage"]["executed_input_tokens"], expected["usage"]["executed_input_tokens"])
        self.assertEqual(actual["usage"]["expanded_input_tokens"], expected["usage"]["expanded_input_tokens"])
        self.assertTrue(all(ids.shape[0] <= 2 for _, ids in calls))
        self.assertTrue(any(ids.shape[0] == 2 for _, ids in calls), "options should be batched")
        self.assertEqual(actual["runtime"]["execution"], "shared_prefix")
        self.assertFalse(actual["runtime"]["cache_hit"])

    def test_question_order_options_order_microbatch_and_request_isolation(self):
        engine = self.engine()
        expected = engine.predict(self.body)["answers"]
        reversed_body = {**self.body, "questions": dict(reversed(list(self.body["questions"].items())))}
        self.assert_answers_close(engine.predict(reversed_body)["answers"], expected)
        for key, question in self.body["questions"].items():
            body = {"state": self.body["state"], "questions": {key: question}}
            self.assert_answers_close(engine.predict(body)["answers"], {key: expected[key]})
        q = self.body["questions"]["route"]
        reversed_q = {**q, "criteria": dict(reversed(list(q["criteria"].items())))}
        reordered = {**self.body, "questions": {"route": reversed_q}}
        self.assert_answers_close(engine.predict(reordered)["answers"], {"route": expected["route"]})
        engine.predict({**self.body, "state": "Completely different state with a different length."})
        self.assert_answers_close(engine.predict(self.body)["answers"], expected)
        self.assert_answers_close(self.engine(microbatch=1).predict(self.body)["answers"], expected)

    def test_equal_length_questions_batch_without_cross_question_contamination(self):
        questions = {f"q{i}": {"type": "choice", "instructions": f"Pick team {i}.",
                              "criteria": {"one": "Yes", "two": "A much longer criterion"}}
                     for i in range(5)}
        body = {**self.body, "questions": questions}
        expected = self.engine("expanded").predict(body)["answers"]
        engine = self.engine(microbatch=3)
        self.model.adapter.calls.clear()
        actual = engine.predict(body)["answers"]
        self.assert_answers_close(actual, expected)
        self.assertTrue(any(ids.shape[0] == 3 for _, ids in self.model.adapter.calls))
        for key, question in questions.items():
            single = engine.predict({**body, "questions": {key: question}})["answers"]
            self.assert_answers_close(single, {key: expected[key]})

    def test_backend_selection_and_request_validation(self):
        self.assertEqual(self.engine(supports_prefix=False).execution_mode, "expanded")
        self.assertEqual(self.engine("expanded").execution_mode, "expanded")
        with self.assertRaisesRegex(ValueError, "does not support"):
            self.engine("shared_prefix", supports_prefix=False)
        for kwargs in ({"mode": "typo"}, {"microbatch": 0}):
            with self.assertRaises(ValueError):
                self.engine(**kwargs)
        engine = self.engine()
        engine.max_request_tokens = 1
        self.model.adapter.calls.clear()
        with self.assertRaises(ValueError):
            engine.predict(self.body)
        self.assertEqual(self.model.adapter.calls, [])

    def test_option_cardinality_and_no_gradients(self):
        for count in (1, 255):
            body = {"state": "", "questions": {"q": {"type": "choice", "instructions": "",
                    "criteria": {f"option_{i}": None for i in range(count)}}}}
            actual = self.engine().predict(body)
            expected = self.engine("expanded").predict(body)
            self.assert_answers_close(actual["answers"], expected["answers"])
            self.assertAlmostEqual(sum(actual["answers"]["q"]["probabilities"].values()), 1, places=6)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_hybrid_cache_forks_have_independent_storage(self):
        from openjev.models.qwen35.prefix_state import Qwen35PrefixState, fork_prefix
        layer = SimpleNamespace(**{name: torch.randn(1, 3, 4) for name in (
            "keys", "values", "conv_states", "recurrent_states",
        )}, metadata={"positions": [1, 2]})
        parent = Qwen35PrefixState(SimpleNamespace(layers=[layer]), 4, 1, "owner")
        child = fork_prefix(parent, [0, 0])
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            original = getattr(layer, name).clone()
            forked = getattr(child.cache.layers[0], name)
            forked[0].zero_()
            torch.testing.assert_close(getattr(layer, name), original)
            torch.testing.assert_close(forked[1], original[0])
        child.cache.layers[0].metadata["positions"].append(3)
        self.assertEqual(layer.metadata["positions"], [1, 2])
