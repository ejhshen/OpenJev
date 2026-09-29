"""Reproducible selected-action feedback; standard library only.

The environment owns oracle data and episode outcomes. Actor-facing handles and
feedback contain no posterior or outcome label. Its checkpoint is environment
state, not an actor batch; restoring it preserves outstanding episode outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class OutcomeSpec:
    num_options: int
    hard_label: int | None = None
    posterior: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.num_options, int) or isinstance(self.num_options, bool) or self.num_options < 1:
            raise ValueError("num_options must be a positive integer")
        if (self.hard_label is None) == (self.posterior is None):
            raise ValueError("provide exactly one hard_label or posterior")
        if self.hard_label is not None:
            if not isinstance(self.hard_label, int) or isinstance(self.hard_label, bool) or not 0 <= self.hard_label < self.num_options:
                raise ValueError("hard_label must index a valid option")
        if self.posterior is not None:
            values = tuple(float(value) for value in self.posterior)
            if len(values) != self.num_options or any(not math.isfinite(p) or p < 0 for p in values):
                raise ValueError("posterior must contain one finite nonnegative probability per option")
            if not math.isclose(sum(values), 1.0, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("posterior must sum to one; it is never renormalized")
            object.__setattr__(self, "posterior", values)


@dataclass(frozen=True)
class EpisodeHandle:
    episode_id: str
    context_id: str
    num_options: int


class BanditFeedbackEnvironment:
    """Offline single-outcome simulator with independent episodes and fixed Y.

    Repeated actions within an episode retain identical correctness and count
    as repeated queries. New outcomes are drawn only by ``start_episode``.
    """

    def __init__(self, oracles: Mapping[str, OutcomeSpec], *, seed: int = 0) -> None:
        if not oracles or any(not isinstance(key, str) or not isinstance(value, OutcomeSpec) for key, value in oracles.items()):
            raise ValueError("oracles must map context IDs to OutcomeSpec values")
        self._oracles = dict(oracles)
        encoded = json.dumps({key: asdict(value) for key, value in sorted(oracles.items())}, sort_keys=True, separators=(",", ":"))
        self._oracle_fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        self._rng = random.Random(seed)
        self._episodes: dict[str, tuple[str, int]] = {}
        self._next_episode = 0
        self._queries = 0

    @property
    def query_count(self) -> int:
        return self._queries

    def start_episode(self, context_id: str, *, episode_id: str | None = None) -> EpisodeHandle:
        spec = self._oracles[context_id]
        if episode_id is None:
            while f"episode-{self._next_episode}" in self._episodes:
                self._next_episode += 1
            episode_id = f"episode-{self._next_episode}"
            self._next_episode += 1
        elif not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a nonempty string")
        if episode_id in self._episodes:
            if self._episodes[episode_id][0] != context_id:
                raise ValueError("an episode ID cannot be reassigned to another context")
            return EpisodeHandle(episode_id, context_id, spec.num_options)
        if spec.hard_label is not None:
            outcome = spec.hard_label
        else:
            draw, cumulative = self._rng.random(), 0.0
            # Roundoff at the cumulative endpoint must not assign an outcome
            # whose declared probability is zero.
            outcome = max(index for index, value in enumerate(spec.posterior) if value > 0)  # type: ignore[union-attr]
            for index, probability in enumerate(spec.posterior):  # type: ignore[union-attr]
                cumulative += probability
                if draw < cumulative:
                    outcome = index
                    break
        self._episodes[episode_id] = (context_id, outcome)
        return EpisodeHandle(episode_id, context_id, spec.num_options)

    def feedback(self, episode_id: str, actions: Sequence[int]) -> tuple[float, ...]:
        context_id, outcome = self._episodes[episode_id]
        size = self._oracles[context_id].num_options
        actions = tuple(actions)
        if not actions:
            raise ValueError("a feedback query needs at least one action")
        if any(not isinstance(a, int) or isinstance(a, bool) or not 0 <= a < size for a in actions):
            raise ValueError("actions must index this episode's option set")
        self._queries += len(actions)
        return tuple(float(action == outcome) for action in actions)

    def feedback_many(self, episode_ids: Sequence[str], action_groups: Sequence[Sequence[int]]) -> tuple[tuple[float, ...], ...]:
        if len(episode_ids) != len(action_groups):
            raise ValueError("one action group is required for each episode")
        return tuple(self.feedback(episode_id, actions) for episode_id, actions in zip(episode_ids, action_groups))

    def close_episode(self, episode_id: str) -> None:
        del self._episodes[episode_id]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "oracle_fingerprint": self._oracle_fingerprint,
            "rng_state": self._rng.getstate(),
            "next_episode": self._next_episode,
            "queries": self._queries,
            "episodes": {key: {"context_id": context, "outcome": outcome} for key, (context, outcome) in self._episodes.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1 or state.get("oracle_fingerprint") != self._oracle_fingerprint:
            raise ValueError("feedback state version or oracle dataset does not match")
        next_episode, queries = state["next_episode"], state["queries"]
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (next_episode, queries)):
            raise ValueError("feedback checkpoint counters must be nonnegative integers")
        episodes = {}
        for episode_id, record in state["episodes"].items():
            context, outcome = record["context_id"], record["outcome"]
            if not isinstance(episode_id, str) or not episode_id or context not in self._oracles:
                raise ValueError("invalid checkpoint episode or context")
            if not isinstance(outcome, int) or isinstance(outcome, bool) or not 0 <= outcome < self._oracles[context].num_options:
                raise ValueError("invalid checkpoint outcome")
            episodes[episode_id] = (context, outcome)

        def tuples(value: Any) -> Any:
            return tuple(tuples(x) for x in value) if isinstance(value, (list, tuple)) else value

        rng = random.Random()
        rng.setstate(tuples(state["rng_state"]))
        # Assign only after validating the complete checkpoint.
        self._rng, self._episodes = rng, episodes
        self._next_episode, self._queries = next_episode, queries
