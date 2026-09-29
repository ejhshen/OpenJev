"""TypeSafe /v1/systemone JSON mapping to the one decision primitive.

Wire contract: https://docs.typesafe.ai/primitives/{choice,noul,score}.
Question identifiers and Score level indices never enter model text.
"""

from dataclasses import dataclass
import json
import math

from openjev.models.decision.schema import Decision, Option


class InvalidDecisionRequest(ValueError):
    """A caller request fails the protocol or token budget contract."""


def structured_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    raise ValueError("content must be a string, object or array")


@dataclass(frozen=True)
class TypedQuestion:
    decision: Decision
    primitive: str
    legend: dict | None = None


def parse_request(body, *, model_ids):
    if not isinstance(body, dict) or "state" not in body:
        raise ValueError("request requires state and questions")
    if body.get("model") is not None and body["model"] not in model_ids:
        raise ValueError("unknown model")
    state = structured_text(body["state"])
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty object")
    result = []
    for key, question in questions.items():
        if not isinstance(key, str) or not key or not isinstance(question, dict):
            raise ValueError("questions require nonempty string IDs and question objects")
        instructions = structured_text(question.get("instructions"))
        primitive, criteria = question.get("type"), question.get("criteria")
        legend = None
        if primitive == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValueError("choice criteria must map 1 to 255 unique option names to descriptions")
            options = tuple(Option(name, name, None if description is None else structured_text(description))
                            for name, description in criteria.items())
        elif primitive == "noul":
            criteria = {} if criteria is None else criteria
            if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
                raise ValueError("noul criteria supports only false and true descriptions")
            options = tuple(Option(key, name, None if criteria.get(key) is None else structured_text(criteria[key]))
                            for key, name in (("false", "False"), ("true", "True")))
        elif primitive == "score":
            if not isinstance(criteria, list) or not 1 <= len(criteria) <= 10:
                raise ValueError("score criteria requires 1 to 10 level descriptions")
            options = tuple(Option(str(index), structured_text(description)) for index, description in enumerate(criteria))
            legend = {str(index): description for index, description in enumerate(criteria)}
        else:
            raise ValueError("question type must be choice, noul or score")
        decision = Decision(key, state, instructions, options,
                            tuple(float(i) for i in range(len(options))) if primitive == "score" else None)
        result.append(TypedQuestion(decision, primitive, legend))
    return result


def format_answer(question, probabilities):
    probabilities = [float(p) for p in probabilities]
    if len(probabilities) != len(question.decision.options):
        raise RuntimeError("model returned an incomplete probability vector")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities) or not math.isclose(sum(probabilities), 1, abs_tol=2e-6):
        raise RuntimeError("model returned an invalid probability distribution")
    if question.primitive == "noul":
        return {"type": "noul", "noul": probabilities[1]}
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    confidence = 1.0 if len(probabilities) == 1 else max(0.0, min(1.0, 1 - entropy / math.log(len(probabilities))))
    answer = {"type": question.primitive,
              "probabilities": {option.id: p for option, p in zip(question.decision.options, probabilities)},
              "confidence": confidence}
    if question.primitive == "choice":
        answer["choice"] = question.decision.options[max(range(len(probabilities)), key=probabilities.__getitem__)].name
    else:
        answer.update(score=sum(i * p for i, p in enumerate(probabilities)), legend=question.legend)
    return answer
