"""The public processed-data contract, independent of torch or datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from openjev.models.decision.schema import Decision, Option
from .distribution import normalize_distribution_row


@dataclass(frozen=True)
class DecisionContext:
    """Actor-visible decision inputs, with no target or latent outcome field."""

    decision: Decision
    group_id: str
    source: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    primitive: str = "choice"
    sample_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id is required for split isolation")
        if self.primitive not in ("choice", "noul", "score"):
            raise ValueError("primitive must be choice, noul or score")
        if self.primitive == "noul" and len(self.decision.options) != 2:
            raise ValueError("noul contexts require exactly two options")
        if self.primitive == "score" and self.decision.score_values is None:
            raise ValueError("score contexts require score_values")
        if (isinstance(self.sample_weight, bool) or not isinstance(self.sample_weight, (int, float))
                or not math.isfinite(self.sample_weight) or self.sample_weight < 0):
            raise ValueError("sample_weight must be finite and nonnegative")
        if not isinstance(self.source, Mapping) or not isinstance(self.provenance, Mapping):
            raise ValueError("source and provenance must be objects")
        object.__setattr__(self, "source", dict(self.source))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def id(self) -> str:
        return self.decision.id

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> DecisionContext:
        if not isinstance(row, Mapping):
            raise ValueError("context row must be an object")
        row = normalize_distribution_row(row, actor=True)
        if any(key in row for key in ("target", "target_distribution", "posterior", "hard_label", "oracle")):
            raise ValueError("actor context must not contain target/oracle fields; use the separate context export")
        try:
            decision = Decision(row["id"], row["state"], row["question"],
                                tuple(Option(item["id"], item["name"], item.get("criteria")) for item in row["options"]),
                                row.get("score_values"))
            return cls(decision, row["group_id"], row.get("source", {}), row.get("provenance", {}),
                       row.get("primitive", "choice"), row.get("sample_weight", 1.0))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed context row: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        result = {"id": self.id, "group_id": self.group_id, "state": self.decision.state,
                  "question": self.decision.question, "options": [
                      {"id": option.id, "name": option.name, "criteria": option.criteria} for option in self.decision.options],
                  "source": dict(self.source), "provenance": dict(self.provenance),
                  "primitive": self.primitive, "sample_weight": self.sample_weight}
        if self.decision.score_values is not None:
            result["score_values"] = list(self.decision.score_values)
        return result


@dataclass(frozen=True)
class DecisionTarget:
    kind: str
    option_id: str | None = None
    probabilities: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.kind == "hard":
            if not isinstance(self.option_id, str) or not self.option_id:
                raise ValueError("hard target requires a nonempty option_id")
            if self.probabilities is not None:
                raise ValueError("hard target cannot also contain probabilities")
        elif self.kind == "soft":
            if self.option_id is not None or not self.probabilities:
                raise ValueError("soft target requires probabilities and no option_id")
            probs = dict(self.probabilities)
            if any(not isinstance(k, str) or not k for k in probs):
                raise ValueError("probability keys must be nonempty option IDs")
            if any(isinstance(p, bool) or not isinstance(p, (int, float))
                   or not math.isfinite(p) or p < 0 for p in probs.values()):
                raise ValueError("probabilities must be finite nonnegative numbers")
            if not math.isclose(math.fsum(probs.values()), 1.0, abs_tol=1e-6, rel_tol=0):
                raise ValueError("soft target probabilities must sum to 1; no implicit renormalization")
            object.__setattr__(self, "probabilities", probs)
        else:
            raise ValueError(f"unknown target kind: {self.kind!r}")

    def vector(self, decision: Decision) -> tuple[float, ...]:
        ids = tuple(option.id for option in decision.options)
        if self.kind == "hard":
            if self.option_id not in ids:
                raise ValueError(f"target {self.option_id!r} is not a declared option")
            return tuple(float(key == self.option_id) for key in ids)
        assert self.probabilities is not None
        if set(self.probabilities) != set(ids):
            raise ValueError("soft target must name exactly the declared options, including zero probabilities")
        return tuple(float(self.probabilities[key]) for key in ids)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DecisionTarget:
        if not isinstance(value, Mapping):
            raise ValueError("target must be an object")
        return cls(kind=value.get("kind", ""), option_id=value.get("option_id"),
                   probabilities=value.get("probabilities"))

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "hard":
            return {"kind": "hard", "option_id": self.option_id}
        return {"kind": "soft", "probabilities": dict(self.probabilities or {})}


@dataclass(frozen=True)
class DecisionExample:
    decision: Decision
    target: DecisionTarget
    group_id: str
    source: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    primitive: str = "choice"
    sample_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id is required for split isolation")
        if self.primitive not in ("choice", "noul", "score"):
            raise ValueError("primitive must be choice, noul or score")
        if self.primitive == "noul" and len(self.decision.options) != 2:
            raise ValueError("noul examples require exactly two options")
        if self.primitive == "score" and self.decision.score_values is None:
            raise ValueError("score examples require score_values")
        if (isinstance(self.sample_weight, bool) or not isinstance(self.sample_weight, (int, float))
                or not math.isfinite(self.sample_weight) or self.sample_weight < 0):
            raise ValueError("sample_weight must be finite and nonnegative")
        if not isinstance(self.source, Mapping) or not isinstance(self.provenance, Mapping):
            raise ValueError("source and provenance must be objects")
        self.target.vector(self.decision)
        object.__setattr__(self, "source", dict(self.source))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def id(self) -> str:
        return self.decision.id

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> DecisionExample:
        if not isinstance(row, Mapping):
            raise ValueError("dataset row must be an object")
        row = normalize_distribution_row(row)
        try:
            options = tuple(Option(id=item["id"], name=item["name"], criteria=item.get("criteria"))
                            for item in row["options"])
            decision = Decision(id=row["id"], state=row["state"], question=row["question"],
                                options=options, score_values=row.get("score_values"))
            return cls(decision=decision, target=DecisionTarget.from_dict(row["target"]),
                       group_id=row["group_id"], source=row.get("source", {}),
                       provenance=row.get("provenance", {}), primitive=row.get("primitive", "choice"),
                       sample_weight=row.get("sample_weight", 1.0))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed decision row: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id, "group_id": self.group_id, "state": self.decision.state,
            "question": self.decision.question,
            "options": [{"id": option.id, "name": option.name, "criteria": option.criteria}
                        for option in self.decision.options],
            "target": self.target.to_dict(), "source": dict(self.source),
            "provenance": dict(self.provenance), "primitive": self.primitive,
            "sample_weight": self.sample_weight,
        }
        if self.decision.score_values is not None:
            result["score_values"] = list(self.decision.score_values)
        return result
