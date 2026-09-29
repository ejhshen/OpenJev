"""Public decision records and tensor outputs, without a torch import at load time."""

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class Option:
    id: str
    name: str
    criteria: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("option id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("option name must be a non-empty string")
        if self.criteria is not None and not isinstance(self.criteria, str):
            raise TypeError("option criteria must be text or None")


@dataclass(frozen=True)
class Decision:
    id: str
    state: str
    question: str
    options: tuple[Option, ...]
    score_values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("decision id must be a non-empty string")
        if not isinstance(self.state, str) or not isinstance(self.question, str):
            raise TypeError("state and question must be serialized text")
        object.__setattr__(self, "options", tuple(self.options))
        if not 1 <= len(self.options) <= 255:
            raise ValueError("a decision requires 1 to 255 options")
        if not all(isinstance(option, Option) for option in self.options):
            raise TypeError("options must contain Option instances")
        if len({option.id for option in self.options}) != len(self.options):
            raise ValueError("option ids must be unique within a decision")
        if self.score_values is not None:
            values = tuple(self.score_values)
            if len(values) != len(self.options):
                raise ValueError("score_values must align with options")
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError("score_values must be finite numbers")
            object.__setattr__(self, "score_values", values)


@dataclass(frozen=True)
class DecisionFeatures:
    question_features: "Tensor"
    option_features: "Tensor"
    option_mask: "Tensor"


@dataclass(frozen=True)
class DecisionOutput:
    logits: "Tensor"
    log_probs: "Tensor"
    probabilities: "Tensor"
    option_mask: "Tensor"
