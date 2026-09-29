import unittest

from openjev.models.decision.schema import Decision, Option
from openjev.models.decision.serializer import DecisionSerializer


class CharacterTokenizer:
    """Every closing marker spans several tokens; IDs are reversible."""

    bos_token_id = 1

    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("serializer must own BOS insertion")
        return [ord(character) + 2 for character in text]


def decode(ids):
    return "".join(chr(token - 2) for token in ids if token != 1)


class SerializerTests(unittest.TestCase):
    def setUp(self):
        self.serializer = DecisionSerializer(CharacterTokenizer())
        self.decision = Decision(
            "d1", "用户要求退款。", "Where should this go?",
            (Option("a", "Billing", "Handles refunds"), Option("b", "Technical")),
        )

    def test_segments_spans_and_optional_criteria(self):
        encoded = self.serializer.serialize(self.decision)
        self.assertEqual(encoded.option_ids, ("a", "b"))
        self.assertIn("<CRITERIA>Handles refunds</CRITERIA>", decode(encoded.option_token_ids[0]))
        self.assertNotIn("CRITERIA", decode(encoded.option_token_ids[1]))
        for branch, option_ids in zip(encoded.branches, encoded.option_token_ids):
            self.assertEqual(branch.input_ids, encoded.state_ids + encoded.question_ids + option_ids)
            self.assertEqual(branch.question_end, len(encoded.prefix_ids) - 1)
            self.assertEqual(branch.option_end, len(branch.input_ids) - 1)
            self.assertTrue(decode(branch.input_ids[:branch.question_end + 1]).endswith("</QUESTION>\n"))
            self.assertTrue(decode(branch.input_ids[:branch.option_end + 1]).endswith("</OPTION>"))
        self.assertEqual(
            encoded.logical_token_count,
            len(encoded.prefix_ids) + sum(map(len, encoded.option_token_ids)),
        )

    def test_ids_and_score_values_do_not_enter_semantics(self):
        renamed = Decision(
            "another-id", self.decision.state, self.decision.question,
            (Option("hidden-1", "Billing", "Handles refunds"), Option("hidden-2", "Technical")),
            score_values=(100.0, -10.0),
        )
        self.assertEqual(
            self.serializer(self.decision).branches, self.serializer(renamed).branches
        )

    def test_option_permutation_only_permutes_branches(self):
        permuted = Decision("d2", self.decision.state, self.decision.question, self.decision.options[::-1])
        original = self.serializer(self.decision)
        changed = self.serializer(permuted)
        self.assertEqual(original.prefix_ids, changed.prefix_ids)
        self.assertEqual(original.branches[::-1], changed.branches)

    def test_explicit_bos_once_per_branch(self):
        encoded = DecisionSerializer(CharacterTokenizer(), add_bos_token=True)(self.decision)
        for branch in encoded.branches:
            self.assertEqual(branch.input_ids[0], 1)
            self.assertEqual(branch.input_ids.count(1), 1)

    def test_over_budget_is_rejected_without_truncation(self):
        encoded = self.serializer(self.decision)
        with self.assertRaisesRegex(ValueError, "logical tokens"):
            DecisionSerializer(
                CharacterTokenizer(), max_decision_tokens=encoded.logical_token_count - 1
            )(self.decision)
        with self.assertRaisesRegex(ValueError, "branch budget"):
            DecisionSerializer(CharacterTokenizer(), max_branch_length=10)(self.decision)

    def test_cardinality_and_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "1 to 255"):
            Decision("d", "", "", ())
        with self.assertRaisesRegex(ValueError, "unique"):
            Decision("d", "", "", (Option("same", "A"), Option("same", "B")))
        one = Decision("d", "", "", [Option("only", "Yes")])
        self.assertEqual(len(self.serializer(one).branches), 1)


if __name__ == "__main__":
    unittest.main()
