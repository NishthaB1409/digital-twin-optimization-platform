"""A live control loop: the twin runs in real time and the policy steers it.

This is the deployment architecture a real plant would use, with a simulated
factory standing in for the physical one. The pieces are the same:

* the floor advances on its own, whether or not anyone is watching;
* every ``decision_interval`` simulated hours a controller reads the state,
  asks the policy for weights, and writes them back;
* observers subscribe to a stream of state rather than polling for it.

Swapping the simulated floor for OPC-UA or an MES feed replaces
:class:`LiveFactory`'s inner loop and nothing else -- the controller, the
policy, and the stream stay as they are.

Safety is deliberate here. A plant will not accept a policy that can emit
anything at all, so the controller clamps what it applies and falls back to a
known-good rule if the policy raises. A scheduler that occasionally does
something merely adequate is fine; one that occasionally does something wild is
not deployable.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..agents.policies import BEST_KNOWN_BLEND, ConstantPolicy
from ..digital_twin.dispatch import FEATURE_NAMES, N_FEATURES
from ..digital_twin.factory import FactoryModel
from ..env.factory_env import observe_factory
from ..utils.config import FactoryConfig

#: Simulated hours advanced per loop iteration. Small enough that the stream
#: looks continuous, large enough not to burn CPU on empty advances.
TICK_HOURS = 1.0

#: If the policy fails, fall back to this. Chosen because it is the strongest
#: fixed rule we have measured, not because it is convenient.
FALLBACK_WEIGHTS = BEST_KNOWN_BLEND


@dataclass
class LiveEvent:
    """Something worth telling an observer about."""

    clock_hours: float
    kind: str
    detail: str


@dataclass
class LiveConfig:
    """How fast the clock runs and how often the policy is consulted."""

    #: Simulated hours per real second. 20 means a 400-hour run takes ~20s.
    time_scale: float = 20.0
    decision_interval: float = 8.0
    seed: int = 1000
    #: Stop the policy emitting anything extreme, whatever it predicts.
    max_abs_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.time_scale <= 0:
            raise ValueError("time_scale must be > 0")
        if self.decision_interval <= 0:
            raise ValueError("decision_interval must be > 0")


class LiveFactory:
    """Runs the twin on a wall clock, steered by a policy.

    The simulation owns a background thread; HTTP handlers only ever read a
    snapshot taken under a lock, so a slow observer can never stall the floor.
    """

    def __init__(
        self,
        config: FactoryConfig,
        policy,
        live_config: LiveConfig | None = None,
        horizon_hours: float | None = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.live = live_config or LiveConfig()
        self.horizon = (
            horizon_hours
            if horizon_hours is not None
            else 2.0 * config.n_jobs / config.arrival_rate
        )
        self._slack_scale = float(
            np.mean([family.planned_work for family in config.families])
        )

        self.model = FactoryModel(config, seed=self.live.seed)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._weights = np.array(FALLBACK_WEIGHTS, dtype=float)
        self._decisions = 0
        self._fallbacks = 0
        self._events: list[LiveEvent] = []
        self._started_at: float | None = None
        self._finished = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            raise RuntimeError("already running")
        self.model = FactoryModel(self.config, seed=self.live.seed)
        self.model.reset()
        self.model.set_weights(self._weights)
        self._stop.clear()
        self._decisions = 0
        self._fallbacks = 0
        self._events = []
        self._finished = False
        self._started_at = time.time()
        self._log("start", f"seed {self.live.seed}, {self.config.n_jobs} jobs")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        next_decision = 0.0
        try:
            while not self._stop.is_set() and not self.model.is_complete:
                if self.model.now >= next_decision:
                    self._decide()
                    next_decision += self.live.decision_interval

                with self._lock:
                    before = len(self.model.completed)
                    self.model.advance(self.model.now + TICK_HOURS)
                    finished = len(self.model.completed) - before
                if finished:
                    self._log("progress", f"{finished} job(s) completed")

                # Real time is only spent sleeping, never holding the lock.
                time.sleep(TICK_HOURS / self.live.time_scale)
        finally:
            with self._lock:
                self._finished = self.model.is_complete
            self._log(
                "finish" if self._finished else "stop",
                f"{len(self.model.completed)}/{self.config.n_jobs} jobs done",
            )

    def _decide(self) -> None:
        """Consult the policy, clamp what comes back, apply it."""
        with self._lock:
            observation = observe_factory(
                self.model, self.config, self.horizon, self._slack_scale
            )
        try:
            action = np.asarray(self.policy.act(observation), dtype=float).reshape(-1)
            if action.shape != (N_FEATURES,) or not np.all(np.isfinite(action)):
                raise ValueError(f"policy returned {action!r}")
            limit = self.live.max_abs_weight
            action = np.clip(action, -limit, limit)
        except Exception as error:  # noqa: BLE001 -- never let the floor stall
            self._fallbacks += 1
            action = np.array(FALLBACK_WEIGHTS, dtype=float)
            self._log("fallback", f"policy failed ({error}); using the fixed rule")

        norm = float(np.linalg.norm(action))
        if norm > 1e-6:
            action = action / norm

        with self._lock:
            self.model.set_weights(action)
            self._weights = action
            self._decisions += 1

    def _log(self, kind: str, detail: str) -> None:
        with self._lock:
            self._events.append(LiveEvent(self.model.now, kind, detail))
            del self._events[:-40]  # keep the tail only

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """A consistent view of the floor. Safe to call from any thread."""
        with self._lock:
            model = self.model
            stations = [
                {
                    "name": name,
                    "queue_length": station.queue_length,
                    "busy_machines": station.busy_machines,
                    "capacity": station.capacity,
                    "utilisation": (
                        station.busy_time / (station.capacity * model.now)
                        if model.now > 0
                        else 0.0
                    ),
                    "contested_fraction": station.contested_fraction,
                }
                for name, station in model.stations.items()
            ]
            completed = model.completed
            on_time = (
                sum(1 for job in completed if job.is_on_time) / len(completed)
                if completed
                else 0.0
            )
            return {
                "running": self.running,
                "finished": self._finished,
                "clock_hours": model.now,
                "elapsed_seconds": (
                    time.time() - self._started_at if self._started_at else 0.0
                ),
                "time_scale": self.live.time_scale,
                "jobs_total": self.config.n_jobs,
                "jobs_completed": len(completed),
                "jobs_in_progress": len(model.wip),
                "stations": stations,
                "weights": dict(zip(FEATURE_NAMES, [float(w) for w in self._weights])),
                "policy": getattr(self.policy, "name", "unknown"),
                "decisions": self._decisions,
                "fallbacks": self._fallbacks,
                "on_time_rate": on_time,
                "weighted_tardiness": model.total_weighted_tardiness,
                "events": [
                    {"clock_hours": e.clock_hours, "kind": e.kind, "detail": e.detail}
                    for e in self._events[-8:]
                ],
            }
