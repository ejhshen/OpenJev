"""One tokenization path for expanded branches and shared-prefix execution."""

from dataclasses import dataclass
from typing import Protocol

from .schema import Decision


class Tokenizer(Protocol):
    bos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class SerializedBranch:
    input_ids: tuple[int, ...]
    question_end: int
    option_end: int


@dataclass(frozen=True)
class SerializedDecision:
    decision_id: str
    option_ids: tuple[str, ...]
    state_ids: tuple[int, ...]
    question_ids: tuple[int, ...]
    option_token_ids: tuple[tuple[int, ...], ...]
    branches: tuple[SerializedBranch, ...]
    logical_token_count: int

    @property
    def prefix_ids(self) -> tuple[int, ...]:
        return self.state_ids + self.question_ids


class DecisionSerializer:
    version = "serializer-v1"

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        add_bos_token: bool = False,
        max_branch_length: int = 32768,
        max_decision_tokens: int = 32768,
    ) -> None:
        if max_branch_length < 1 or max_decision_tokens < 1:
            raise ValueError("token budgets must be positive")
        self.tokenizer = tokenizer
        self.add_bos_token = add_bos_token
        self.max_branch_length = max_branch_length
        self.max_decision_tokens = max_decision_tokens
        if add_bos_token and getattr(tokenizer, "bos_token_id", None) is None:
            raise ValueError("explicit BOS insertion requires tokenizer.bos_token_id")

    def _encode(self, text: str) -> tuple[int, ...]:
        ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
        if not ids:
            raise ValueError("tokenizer returned no tokens for a non-empty segment")
        if any(not isinstance(token, int) or token < 0 for token in ids):
            raise ValueError("tokenizer must return non-negative integer token IDs")
        return ids

    def serialize(self, decision: Decision) -> SerializedDecision:
        # The newlines belong to their segment and are identical in both paths.
        state_ids = self._encode(f"<STATE>{decision.state}</STATE>\n")
        if self.add_bos_token:
            state_ids = (self.tokenizer.bos_token_id,) + state_ids
        question_ids = self._encode(f"<QUESTION>{decision.question}</QUESTION>\n")
        prefix_ids = state_ids + question_ids
        options = []
        for option in decision.options:
            text = f"<OPTION><NAME>{option.name}</NAME>"
            if option.criteria is not None:
                text += f"<CRITERIA>{option.criteria}</CRITERIA>"
            options.append(self._encode(text + "</OPTION>"))
        logical_count = len(prefix_ids) + sum(map(len, options))
        if logical_count > self.max_decision_tokens:
            raise ValueError(
                f"decision {decision.id!r} uses {logical_count} logical tokens; "
                f"budget is {self.max_decision_tokens} (no automatic truncation)"
            )
        branches = tuple(
            SerializedBranch(
                input_ids=prefix_ids + option_ids,
                question_end=len(prefix_ids) - 1,
                option_end=len(prefix_ids) + len(option_ids) - 1,
            )
            for option_ids in options
        )
        if any(len(branch.input_ids) > self.max_branch_length for branch in branches):
            raise ValueError(
                f"decision {decision.id!r} exceeds branch budget "
                f"{self.max_branch_length} (no automatic truncation)"
            )
        return SerializedDecision(
            decision_id=decision.id,
            option_ids=tuple(option.id for option in decision.options),
            state_ids=state_ids,
            question_ids=question_ids,
            option_token_ids=tuple(options),
            branches=branches,
            logical_token_count=logical_count,
        )

    __call__ = serialize
