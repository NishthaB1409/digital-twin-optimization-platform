"""The composite dispatching rule the RL agent will eventually steer.

Every waiting job is scored as a weighted sum of four features, and the lowest
score is dispatched. The weight vector is mutable, which is the whole point:
Day 1 sets it by hand, Day 3 lets PPO move it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .entities import Job

FEATURE_NAMES: tuple[str, ...] = (
    "processing_time",
    "slack",
    "remaining_work",
    "waiting_time",
)
N_FEATURES = len(FEATURE_NAMES)

#: Weight vectors that reproduce the textbook rules. Handy as RL baselines and
#: as fixtures in the tests.
CLASSICAL_RULES: dict[str, tuple[float, float, float, float]] = {
    "spt": (1.0, 0.0, 0.0, 0.0),        # shortest processing time
    "lpt": (-1.0, 0.0, 0.0, 0.0),       # longest processing time
    "min_slack": (0.0, 1.0, 0.0, 0.0),  # least slack first
    "lwkr": (0.0, 0.0, 1.0, 0.0),       # least work remaining
    "mwkr": (0.0, 0.0, -1.0, 0.0),      # most work remaining
    "fifo": (0.0, 0.0, 0.0, -1.0),      # longest wait first
    "lifo": (0.0, 0.0, 0.0, 1.0),
}

_EPS = 1e-12


def min_max_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each column into [0, 1] across the rows of ``matrix``.

    Columns that are constant collapse to zero rather than dividing by zero,
    which correctly makes a feature carry no signal when every job in the queue
    agrees on it.
    """
    lo = matrix.min(axis=0)
    hi = matrix.max(axis=0)
    span = hi - lo
    out = np.zeros_like(matrix)
    varying = span > _EPS
    out[:, varying] = (matrix[:, varying] - lo[varying]) / span[varying]
    return out


class CompositeDispatchRule:
    """Scores queued jobs as ``w . x`` and dispatches the minimum.

    Features are min-max normalised across the *current* queue, so the four
    weights are directly comparable to each other and the rule does not care
    what unit time is measured in. Normalising at dispatch time (rather than
    against global constants) is what keeps a single weight vector meaningful
    at both a 2-job and a 40-job queue.

    With weights in [-1, 1] the rule spans the classical heuristics -- see
    :data:`CLASSICAL_RULES`.
    """

    def __init__(
        self,
        weights: Sequence[float] = CLASSICAL_RULES["spt"],
    ) -> None:
        self._weights = np.zeros(N_FEATURES, dtype=float)
        self.weights = weights

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{name}={value:+.2f}"
            for name, value in zip(FEATURE_NAMES, self._weights)
        )
        return f"{type(self).__name__}({pairs})"

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------
    @property
    def weights(self) -> np.ndarray:
        """A copy, so callers cannot mutate the rule by accident."""
        return self._weights.copy()

    @weights.setter
    def weights(self, values: Sequence[float]) -> None:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape != (N_FEATURES,):
            raise ValueError(
                f"expected {N_FEATURES} weights {FEATURE_NAMES}, "
                f"got shape {np.asarray(values).shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"weights must all be finite, got {arr!r}")
        self._weights = arr

    @classmethod
    def from_name(cls, name: str) -> "CompositeDispatchRule":
        """Build one of the :data:`CLASSICAL_RULES` by name."""
        try:
            return cls(CLASSICAL_RULES[name])
        except KeyError:
            raise KeyError(
                f"unknown rule {name!r}; choose from {sorted(CLASSICAL_RULES)}"
            ) from None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def features(self, queue: Sequence[Job], now: float) -> np.ndarray:
        """Raw (unnormalised) feature matrix, one row per queued job."""
        return np.array(
            [
                (
                    job.current_processing_time,
                    job.slack(now),
                    job.remaining_work,
                    job.waiting_time(now),
                )
                for job in queue
            ],
            dtype=float,
        )

    def scores(self, queue: Sequence[Job], now: float) -> np.ndarray:
        return min_max_normalise(self.features(queue, now)) @ self._weights

    def select(self, queue: Sequence[Job], now: float) -> Job:
        """Pick the next job to run. Ties break toward the earlier arrival."""
        if not queue:
            raise ValueError("cannot dispatch from an empty queue")
        if len(queue) == 1:
            return queue[0]
        return queue[int(np.argmin(self.scores(queue, now)))]
