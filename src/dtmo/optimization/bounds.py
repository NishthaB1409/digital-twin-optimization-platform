"""Combinatorial lower bounds on what any schedule could possibly achieve.

Day 3 ended with PPO tied against SPT, which on its own is ambiguous: either
the agent underperformed and there is room left, or SPT is already near the
ceiling and there was never much to win. A lower bound settles that. Nothing --
no dispatch rule, no RL policy, no exact solver -- can beat these numbers, so
the gap between the best policy and the bound is all the room that exists.

These are *relaxations*: each drops some constraint of the real problem, which
can only make the optimum better, so the result is guaranteed not to exceed the
true optimum. They are cheap and exact to compute. The LP relaxation in
:mod:`dtmo.optimization.lp` is tighter but far more expensive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..digital_twin.entities import Job, StationSpec


@dataclass(frozen=True)
class Bounds:
    """Lower bounds for one instance. No schedule can do better than these."""

    weighted_tardiness: float
    makespan: float
    #: Which relaxation produced the makespan bound, for reporting.
    makespan_source: str
    n_jobs: int
    #: Jobs that cannot possibly meet their due date, whatever the schedule.
    n_doomed: int

    def gap(self, achieved_weighted_tardiness: float) -> float:
        """Fraction of a policy's cost that is *not* forced by the instance.

        Returns the share of the achieved cost that better scheduling could in
        principle remove. 0.0 means the policy is provably optimal; 0.5 means
        at most half the cost is schedulable away.
        """
        if achieved_weighted_tardiness <= 0:
            return 0.0
        excess = achieved_weighted_tardiness - self.weighted_tardiness
        return max(0.0, excess / achieved_weighted_tardiness)

    def summary(self) -> str:
        return (
            f"weighted tardiness >= {self.weighted_tardiness:.1f}  "
            f"makespan >= {self.makespan:.1f} ({self.makespan_source})  "
            f"{self.n_doomed}/{self.n_jobs} jobs cannot meet their date"
        )


def earliest_completion(job: Job) -> float:
    """When a job could finish if it never queued anywhere.

    Its release time plus its own work, ignoring every other job in the
    factory. No schedule can finish it sooner, because a job cannot be in two
    places at once and its operations are strictly ordered.
    """
    return job.release_time + job.total_work


def unavoidable_tardiness(job: Job) -> float:
    """Lateness that is baked into the instance for this job.

    If a job's release plus its own processing already runs past its due date,
    it is late no matter what the scheduler does. Dropping all contention
    between jobs is the relaxation here.
    """
    return max(0.0, earliest_completion(job) - job.due_date)


def unavoidable_weighted_tardiness(jobs: Iterable[Job]) -> float:
    """Lower bound on total weighted tardiness for the whole instance."""
    return sum(job.family.weight * unavoidable_tardiness(job) for job in jobs)


def station_work(jobs: Iterable[Job]) -> dict[str, float]:
    """Total processing hours each station must absorb."""
    totals: dict[str, float] = {}
    for job in jobs:
        for operation, duration in zip(job.family.route, job.proc_times):
            totals[operation.station] = totals.get(operation.station, 0.0) + duration
    return totals


def _head_and_tail(job: Job, station: str) -> tuple[float, float]:
    """Work strictly before, and strictly after, a job's visit to ``station``.

    A job that has not reached a station yet still owes its upstream work, and
    once past it still owes the downstream work. Both are unavoidable.
    """
    head = 0.0
    tail = 0.0
    seen = False
    for operation, duration in zip(job.family.route, job.proc_times):
        if operation.station == station:
            seen = True
            continue
        if seen:
            tail += duration
        else:
            head += duration
    return head, tail


def station_makespan_bound(
    jobs: Sequence[Job], station: StationSpec
) -> float:
    """Lower bound on makespan from one station's capacity.

    The station cannot start before its earliest arrival, must absorb all of
    its work at ``capacity`` jobs in parallel, and whatever finishes last still
    owes its downstream operations. Machine idle time only makes this worse, so
    it is a valid floor.
    """
    visiting = [
        (job, *_head_and_tail(job, station.name))
        for job in jobs
        if any(op.station == station.name for op in job.family.route)
    ]
    if not visiting:
        return 0.0

    work = sum(
        duration
        for job, _, _ in visiting
        for operation, duration in zip(job.family.route, job.proc_times)
        if operation.station == station.name
    )
    earliest_arrival = min(job.release_time + head for job, head, _ in visiting)
    smallest_tail = min(tail for _, _, tail in visiting)
    return earliest_arrival + work / station.capacity + smallest_tail


def route_makespan_bound(jobs: Iterable[Job]) -> float:
    """Lower bound from the single longest job. It alone takes this long."""
    return max((earliest_completion(job) for job in jobs), default=0.0)


def compute_bounds(
    jobs: Sequence[Job], stations: Sequence[StationSpec]
) -> Bounds:
    """All bounds for one instance, taking the strongest of each kind."""
    if not jobs:
        raise ValueError("cannot bound an empty instance")

    route_bound = route_makespan_bound(jobs)
    best_makespan = route_bound
    source = "longest route"
    for station in stations:
        candidate = station_makespan_bound(jobs, station)
        if candidate > best_makespan:
            best_makespan = candidate
            source = f"{station.name} capacity"

    return Bounds(
        weighted_tardiness=unavoidable_weighted_tardiness(jobs),
        makespan=best_makespan,
        makespan_source=source,
        n_jobs=len(jobs),
        n_doomed=sum(1 for job in jobs if unavoidable_tardiness(job) > 0),
    )


def bounds_for_seed(config, seed: int) -> Bounds:
    """Build one instance from a seed and bound it, without simulating."""
    from ..digital_twin.factory import FactoryModel

    model = FactoryModel(config, seed=seed)
    model.reset()  # generates the job set; the clock never advances
    return compute_bounds(model.jobs, config.stations)
