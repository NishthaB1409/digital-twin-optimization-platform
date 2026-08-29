"""A uniform interface over everything that can drive the environment.

A trained PPO agent and a textbook dispatching rule have to be scored the same
way, on the same episodes, or the comparison means nothing. Both become a
:class:`Policy`: something that maps an observation to four weights.

A classical rule is just a policy that ignores its observation -- which is
precisely the weakness the RL agent is supposed to exploit.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..digital_twin.dispatch import CLASSICAL_RULES, N_FEATURES


@runtime_checkable
class Policy(Protocol):
    """Maps an observation to a dispatch weight vector."""

    name: str

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return the four dispatch weights to use for the next interval."""
        ...

    def reset(self) -> None:
        """Called at the start of each episode. Stateless policies ignore it."""
        ...


class ConstantPolicy:
    """Holds one weight vector for the whole episode.

    Every classical dispatching rule is one of these. They cannot react to a
    queue building at the bottleneck, because they never look at the state.
    """

    def __init__(self, name: str, weights: Sequence[float]) -> None:
        arr = np.asarray(weights, dtype=np.float32).reshape(-1)
        if arr.shape != (N_FEATURES,):
            raise ValueError(
                f"policy {name!r}: expected {N_FEATURES} weights, got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"policy {name!r}: weights must be finite")
        self.name = name
        self.weights = np.clip(arr, -1.0, 1.0)

    def __repr__(self) -> str:
        return f"ConstantPolicy({self.name!r}, {np.round(self.weights, 2).tolist()})"

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.weights

    def reset(self) -> None:
        return None


class RandomPolicy:
    """Uniform random weights each step. The floor any agent must clear."""

    def __init__(self, seed: int | None = None, name: str = "random") -> None:
        self.name = name
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return f"RandomPolicy(seed={self._seed})"

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=N_FEATURES).astype(np.float32)

    def reset(self) -> None:
        # Re-seed per episode so a policy comparison is reproducible.
        self._rng = np.random.default_rng(self._seed)


class SB3Policy:
    """Wraps a trained Stable-Baselines3 model.

    ``deterministic=True`` by default: at evaluation time we want the policy's
    actual choice, not a sample from its exploration distribution.
    """

    def __init__(self, model, name: str = "ppo", deterministic: bool = True) -> None:
        self.model = model
        self.name = name
        self.deterministic = deterministic

    def __repr__(self) -> str:
        return f"SB3Policy({self.name!r}, deterministic={self.deterministic})"

    def act(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def reset(self) -> None:
        return None

    @classmethod
    def load(cls, path, name: str = "ppo", **kwargs) -> "SB3Policy":
        from stable_baselines3 import PPO

        return cls(PPO.load(path), name=name, **kwargs)


#: Best fixed weight vector found by random search, selected on seeds
#: 3000-3015 and then validated on *held-out* seeds 4000-4039, where it beats
#: SPT by +3.53 return (75% win rate, p=0.005).
#:
#: Day 1 reported a different vector at +12.4%, but that number was selection
#: bias: the search picked the best of 150 candidates scored on only 5 seeds,
#: and was then "validated" on a superset containing those same 5. On truly
#: held-out seeds that vector beats SPT by +0.4% (p=0.88) -- i.e. not at all.
#: Any future search over this space must select and judge on disjoint seeds.
#:
#: The shape is interpretable and matches the ATC rule from the scheduling
#: literature: mostly shortest-processing-time, with a real slack term and a
#: slight bias toward jobs that have waited.
BEST_KNOWN_BLEND = (0.77, 0.28, 0.14, -0.25)


def classical_policies(include_blend: bool = True) -> list[ConstantPolicy]:
    """Every textbook rule as a policy, ready to benchmark against."""
    policies = [
        ConstantPolicy(name, weights)
        for name, weights in sorted(CLASSICAL_RULES.items())
    ]
    if include_blend:
        policies.append(ConstantPolicy("blend", BEST_KNOWN_BLEND))
    return policies
