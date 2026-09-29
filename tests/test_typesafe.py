import unittest

from openjev.api.typesafe import format_answer, parse_request
from openjev.models.decision.serializer import DecisionSerializer


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


class TypeSafeMappingTests(unittest.TestCase):
    def test_three_primitives_and_structured_fields(self):
        body = {"state": {"message": "duplicate charge"}, "questions": {
            "route": {"type": "choice", "instructions": ["Route"], "criteria": {"billing": {"handles": "refunds"}, "other": None}},
            "refund": {"type": "noul", "instructions": "Refund?", "criteria": {"true": "Asked for refund", "false": "No request"}},
            "urgency": {"type": "score", "instructions": "Rate urgency", "criteria": ["Can wait", {"impact": "Blocking"}]},
        }}
        questions = parse_request(body, model_ids={"openjev-latest"})
        self.assertEqual(format_answer(questions[0], [.8, .2])["choice"], "billing")
        self.assertEqual(format_answer(questions[1], [.3, .7]), {"type": "noul", "noul": .7})
        self.assertEqual(format_answer(questions[2], [.25, .75])["score"], .75)
        self.assertEqual(format_answer(questions[2], [.25, .75])["legend"]["1"], {"impact": "Blocking"})

    def test_question_ids_and_score_indices_do_not_enter_text(self):
        base = {"type": "score", "instructions": "Rate", "criteria": ["Low", "High"]}
        a = parse_request({"state": "x", "questions": {"sensitive-id": base}}, model_ids=set())[0]
        b = parse_request({"state": "x", "questions": {"different-id": base}}, model_ids=set())[0]
        serializer = DecisionSerializer(Tokenizer())
        self.assertEqual(serializer(a.decision).branches, serializer(b.decision).branches)
        self.assertEqual([item.name for item in a.decision.options], ["Low", "High"])
        self.assertNotIn("sensitive-id", bytes(serializer(a.decision).branches[0].input_ids).decode())

    def test_partial_or_nonfinite_probabilities_are_rejected(self):
        q = parse_request({"state": "x", "questions": {"q": {"type": "noul", "instructions": "true?"}}}, model_ids=set())[0]
        for p in ([1.0], [.9, .9], [float("nan"), 0], [-.1, 1.1]):
            with self.subTest(p=p), self.assertRaises(RuntimeError):
                format_answer(q, p)

    def test_single_score_level_returns_zero_index_and_full_confidence(self):
        body = {"state": "fixed", "questions": {"score": {
            "type": "score", "instructions": "Apply the only level.", "criteria": ["Only level"]}}}
        question = parse_request(body, model_ids=set())[0]
        answer = format_answer(question, [1.0])
        self.assertEqual(answer, {"type": "score", "probabilities": {"0": 1.0},
                                  "confidence": 1.0, "score": 0.0, "legend": {"0": "Only level"}})

    def test_unknown_model_and_bad_criteria_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown model"):
            parse_request({"state": "x", "model": "wrong", "questions": {}}, model_ids={"right"})
        for question in ({"type": "choice", "instructions": "q", "criteria": []},
                         {"type": "score", "instructions": "q", "criteria": []},
                         {"type": "noul", "instructions": "q", "criteria": {"maybe": "x"}}):
            with self.subTest(question=question), self.assertRaises(ValueError):
                parse_request({"state": "x", "questions": {"q": question}}, model_ids=set())
