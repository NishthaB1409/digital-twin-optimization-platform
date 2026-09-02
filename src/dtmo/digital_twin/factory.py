"""The digital twin: builds the line, releases jobs, runs the clock."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np
import simpy

from .dispatch import N_FEATURES, CompositeDispatchRule
from .entities import Job
from .kpis import KPIs, compute_kpis
from .stations import Station

if TYPE_CHECKING:  # import only for typing -- a runtime import would make
    # dtmo.utils.config <-> dtmo.digital_twin circular, since config imports
    # entities and this package's __init__ imports factory.
    from ..utils.config import FactoryConfig


class FactoryModel:
    """A 6-station line driven by one mutable dispatch weight vector.

    A run is fully determined by ``(config, weights, seed)``, which is what
    makes this usable as an RL environment: the agent changes only the weights,
    so any KPI difference is attributable to that.

    Two ways to drive it:

    * :meth:`run` -- build and simulate to completion, return KPIs. This is the
      Day 1 batch path, and what the classical-rule benchmarks use.
    * :meth:`reset` + :meth:`advance` -- step the clock forward in slices,
      retuning :attr:`dispatcher` weights between slices. This is what the
      Gymnasium environment drives, and it is the reason the weight vector is
      mutable rather than fixed at construction.
    """

    def __init__(
        self,
        config: "FactoryConfig",
        weights: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.seed = config.seed if seed is None else seed
        # One rule per station, not one for the line. Stations differ a lot
        # here -- Surface Treatment has a single machine at ~89% load while
        # Machining has three at ~72% -- and the rule that suits a bottleneck
        # is not obviously the rule that suits a station with slack.
        # `set_weights` broadcasts a single 4-vector across all of them, so the
        # classical rules and the shared-weight agent behave exactly as before.
        initial = config.dispatch_weights if weights is None else weights
        self.dispatchers: dict[str, CompositeDispatchRule] = {
            spec.name: CompositeDispatchRule(initial) for spec in config.stations
        }

        self.env: simpy.Environment | None = None
        self.stations: dict[str, Station] = {}
        self.jobs: list[Job] = []
        self.completed: list[Job] = []

    def __repr__(self) -> str:
        return f"FactoryModel(seed={self.seed}, {self.dispatcher!r})"

    @property
    def dispatcher(self) -> CompositeDispatchRule:
        """The first station's rule.

        With shared weights every station holds the same vector, so this is
        the line's rule. With per-station weights it is only the first one --
        read :attr:`dispatchers` instead.
        """
        return next(iter(self.dispatchers.values()))

    # ------------------------------------------------------------------
    # Job generation
    # ------------------------------------------------------------------
    def _sample_processing_time(self, mean: float, rng: np.random.Generator) -> float:
        """Lognormal draw with the given mean and the configured CV.

        Lognormal rather than normal because processing times are positive and
        right-skewed -- a job can run long, but never finish in negative time.
        """
        cv = self.config.processing_cv
        if cv <= 0:
            return float(mean)
        sigma = math.sqrt(math.log(1.0 + cv * cv))
        mu = math.log(mean) - 0.5 * sigma * sigma
        return float(rng.lognormal(mu, sigma))

    def _build_jobs(self, rng: np.random.Generator) -> list[Job]:
        cfg = self.config
        families = cfg.families
        mix = np.array([family.mix for family in families], dtype=float)
        mix /= mix.sum()

        jobs: list[Job] = []
        clock = 0.0
        for job_id in range(cfg.n_jobs):
            clock += float(rng.exponential(1.0 / cfg.arrival_rate))
            family = families[int(rng.choice(len(families), p=mix))]
            proc_times = tuple(
                self._sample_processing_time(op.mean_time, rng)
                for op in family.route
            )
            # Due dates come off *planned* work, not the realised draw -- a
            # planner setting dates up front cannot see the noise.
            due_date = clock + family.due_factor * family.planned_work
            jobs.append(
                Job(
                    job_id=job_id,
                    family=family,
                    release_time=clock,
                    due_date=due_date,
                    proc_times=proc_times,
                )
            )
        return jobs

    # ------------------------------------------------------------------
    # Simulation -- stepwise interface
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        """Rebuild the line and the job set, leaving the clock at zero."""
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)
        self.env = simpy.Environment()
        self.jobs = self._build_jobs(rng)
        self.completed = []
        self.stations = {
            spec.name: Station(
                env=self.env,
                spec=spec,
                dispatcher=self.dispatchers[spec.name],
                on_operation_complete=self._route,
            )
            for spec in self.config.stations
        }
        self.env.process(self._release_jobs())

    def advance(self, until: float) -> None:
        """Run the clock forward to ``until``.

        A target at or before the current time is a no-op rather than an error,
        so a caller stepping on a fixed grid does not have to special-case the
        moment the line finishes early.
        """
        if self.env is None:
            raise RuntimeError("call reset() before advance()")
        if until <= self.env.now:
            return
        self.env.run(until=until)

    def run(self, until: float | None = None) -> KPIs:
        """Build and simulate to completion. Safe to call repeatedly."""
        self.reset()
        # With no `until`, SimPy stops when the event queue drains -- idle
        # machines parked on an event that will never fire schedule nothing.
        self.env.run(until=until)
        return self.kpis()

    def _release_jobs(self):
        for job in self.jobs:
            delay = job.release_time - self.env.now
            if delay > 0:
                yield self.env.timeout(delay)
            self._route(job)

    def _route(self, job: Job) -> None:
        """Send a job to its next station, or retire it."""
        if job.is_done:
            job.completion_time = self.env.now
            self.completed.append(job)
        else:
            self.stations[job.current_station].submit(job)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def now(self) -> float:
        return 0.0 if self.env is None else float(self.env.now)

    @property
    def is_complete(self) -> bool:
        return bool(self.jobs) and len(self.completed) == len(self.jobs)

    @property
    def released(self) -> list[Job]:
        """Jobs whose release time has passed."""
        now = self.now
        return [job for job in self.jobs if job.release_time <= now]

    @property
    def wip(self) -> list[Job]:
        """Released but not yet finished -- the jobs actually on the floor."""
        return [job for job in self.released if job.completion_time is None]

    @property
    def makespan(self) -> float:
        if not self.completed:
            return 0.0
        return max(job.completion_time for job in self.completed)

    @property
    def total_weighted_tardiness(self) -> float:
        """Weighted tardiness realised so far.

        Only completed jobs contribute: a late job's tardiness is not known
        until it actually finishes.
        """
        return sum(job.weighted_tardiness for job in self.completed)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def kpis(self) -> KPIs:
        """KPIs for the run so far. Closes the queue-time integrals first."""
        end = self.makespan or self.now
        for station in self.stations.values():
            station.finalise(end)
        return compute_kpis(self.jobs, self.completed, self.stations)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Retune the dispatch rules. Takes effect at the very next dispatch.

        Accepts either 4 weights, applied to every station, or 4 per station in
        configured station order. Broadcasting is what keeps a classical rule
        meaningful on a per-station line: SPT everywhere is still SPT.
        """
        values = np.asarray(weights, dtype=float).reshape(-1)
        n_stations = len(self.dispatchers)
        if values.size == N_FEATURES:
            for rule in self.dispatchers.values():
                rule.weights = values
            return
        if values.size == N_FEATURES * n_stations:
            for index, rule in enumerate(self.dispatchers.values()):
                rule.weights = values[index * N_FEATURES : (index + 1) * N_FEATURES]
            return
        raise ValueError(
            f"expected {N_FEATURES} weights (shared) or "
            f"{N_FEATURES * n_stations} ({n_stations} stations x {N_FEATURES}), "
            f"got {values.size}"
        )

    @property
    def station_weights(self) -> dict[str, np.ndarray]:
        """Current weight vector per station."""
        return {name: rule.weights for name, rule in self.dispatchers.items()}
