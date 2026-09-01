"""A linear-programming lower bound on weighted tardiness, solved with HiGHS.

The combinatorial bound in :mod:`dtmo.optimization.bounds` drops all contention
between jobs, which makes it far too weak here -- on this line it lands about
140x below what any real policy achieves, because nearly all tardiness comes
from queueing rather than from jobs being individually impossible.

This module keeps the part that matters. For one station at a time it solves a
*time-indexed transportation relaxation*:

* time is cut into slots of ``slot_hours``;
* ``x[j, t]`` is how much of job j's work at this station happens in slot t;
* the station may do at most ``capacity * slot_hours`` of work per slot, and no
  single job may occupy more than one machine at once;
* each unit of work is charged at the tardiness rate of the slot it lands in.

Two relaxations make it a valid bound rather than an exact answer: work may be
split arbitrarily (preemption, and across machines), and every *other* station's
capacity is ignored. Both only make the optimum better, so the LP value can
never exceed the true optimum.

Because each station is relaxed independently, every station yields its own
valid bound -- so the strongest one wins. They are not summed; that would count
the same jobs several times over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack

from ..digital_twin.entities import Job, StationSpec
from .bounds import _head_and_tail


@dataclass(frozen=True)
class StationRelaxation:
    """The LP bound contributed by one station."""

    station: str
    bound: float
    n_jobs: int
    n_slots: int
    solved: bool
    message: str = ""


@dataclass(frozen=True)
class LPBound:
    """Best station relaxation for one instance."""

    weighted_tardiness: float
    binding_station: str
    stations: tuple[StationRelaxation, ...]
    slot_hours: float

    def summary(self) -> str:
        return (
            f"LP weighted tardiness >= {self.weighted_tardiness:.1f} "
            f"(binding at {self.binding_station}, {self.slot_hours:g}h slots)"
        )


def _station_jobs(
    jobs: Sequence[Job], station: str
) -> list[tuple[Job, float, float, float]]:
    """(job, processing here, earliest arrival, latest useful finish)."""
    out = []
    for job in jobs:
        duration = sum(
            p
            for operation, p in zip(job.family.route, job.proc_times)
            if operation.station == station
        )
        if duration <= 0:
            continue
        head, tail = _head_and_tail(job, station)
        # It cannot arrive before its release plus upstream work, and finishing
        # here later than (due - downstream work) guarantees lateness.
        out.append((job, duration, job.release_time + head, job.due_date - tail))
    return out


def relax_station(
    jobs: Sequence[Job],
    station: StationSpec,
    slot_hours: float = 4.0,
) -> StationRelaxation:
    """Solve the transportation relaxation for one station."""
    entries = _station_jobs(jobs, station.name)
    if not entries:
        return StationRelaxation(station.name, 0.0, 0, 0, True, "no jobs visit")

    total_work = sum(duration for _, duration, _, _ in entries)
    latest_arrival = max(arrival for _, _, arrival, _ in entries)
    # A horizon this long always admits a feasible schedule: start everything
    # at the last arrival and run the station flat out.
    horizon = latest_arrival + total_work / station.capacity + slot_hours
    n_slots = int(math.ceil(horizon / slot_hours))
    n_jobs = len(entries)

    capacity_per_slot = station.capacity * slot_hours

    costs = np.zeros(n_jobs * n_slots)
    upper = np.zeros(n_jobs * n_slots)
    for i, (job, duration, arrival, latest) in enumerate(entries):
        weight = job.family.weight
        base = i * n_slots
        first_slot = int(math.floor(arrival / slot_hours))
        for t in range(first_slot, n_slots):
            # Charge at the slot's START, not its end. If a job still has work
            # in slot t then it completes no earlier than t * slot_hours, so
            # this understates its lateness -- which is what keeps the LP a
            # valid *lower* bound. Charging at the slot end overstates by up to
            # one slot per job and can push the value above the true optimum,
            # which shows up as the bound rising when slots are made coarser.
            earliest_finish = t * slot_hours
            # Rate per hour of processing, so a whole job's work charged in one
            # slot costs exactly that job's tardiness.
            lateness = max(0.0, earliest_finish - latest)
            costs[base + t] = weight * lateness / duration
            # A job occupies at most one machine, so at most slot_hours of it.
            upper[base + t] = min(slot_hours, duration)

    # Each job's work must all be scheduled somewhere.
    rows, cols = [], []
    for i in range(n_jobs):
        for t in range(n_slots):
            rows.append(i)
            cols.append(i * n_slots + t)
    a_eq = csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_jobs, n_jobs * n_slots)
    )
    b_eq = np.array([duration for _, duration, _, _ in entries])

    # The station cannot exceed its capacity in any slot.
    rows, cols = [], []
    for t in range(n_slots):
        for i in range(n_jobs):
            rows.append(t)
            cols.append(i * n_slots + t)
    a_ub = csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_slots, n_jobs * n_slots)
    )
    b_ub = np.full(n_slots, capacity_per_slot)

    result = linprog(
        c=costs,
        A_eq=a_eq,
        b_eq=b_eq,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=np.column_stack([np.zeros_like(upper), upper]),
        method="highs",
    )
    if not result.success:
        return StationRelaxation(
            station.name, 0.0, n_jobs, n_slots, False, result.message
        )
    return StationRelaxation(
        station.name, float(result.fun), n_jobs, n_slots, True
    )


def lp_bound(
    jobs: Sequence[Job],
    stations: Sequence[StationSpec],
    slot_hours: float = 4.0,
) -> LPBound:
    """Strongest single-station relaxation for this instance."""
    if not jobs:
        raise ValueError("cannot bound an empty instance")

    relaxations = tuple(
        relax_station(jobs, station, slot_hours) for station in stations
    )
    solved = [r for r in relaxations if r.solved]
    if not solved:
        raise RuntimeError(
            "every station relaxation failed: "
            + "; ".join(f"{r.station}: {r.message}" for r in relaxations)
        )
    best = max(solved, key=lambda r: r.bound)
    return LPBound(
        weighted_tardiness=best.bound,
        binding_station=best.station,
        stations=relaxations,
        slot_hours=slot_hours,
    )


def lp_bound_for_seed(config, seed: int, slot_hours: float = 4.0) -> LPBound:
    """Build one instance from a seed and bound it, without simulating."""
    from ..digital_twin.factory import FactoryModel

    model = FactoryModel(config, seed=seed)
    model.reset()
    return lp_bound(model.jobs, config.stations, slot_hours)


def best_weighted_tardiness_bound(
    jobs: Sequence[Job],
    stations: Sequence[StationSpec],
    slot_hours: float = 2.0,
) -> tuple[float, str]:
    """Strongest valid bound available, and which relaxation produced it.

    Neither relaxation dominates the other, because they keep different
    constraints. The combinatorial bound keeps each job's own route intact but
    drops all contention; the LP keeps one station's capacity but allows
    preemption and charges conservatively at slot boundaries.

    On a congested line the LP wins by a wide margin -- contention is where
    nearly all the tardiness comes from. On a lightly loaded one the LP can
    schedule everything on time and returns zero, while the combinatorial bound
    still catches jobs that are individually impossible. Both are valid lower
    bounds, so the larger of the two is also valid and is what to report.
    """
    from .bounds import unavoidable_weighted_tardiness

    combinatorial = unavoidable_weighted_tardiness(jobs)
    linear = lp_bound(jobs, stations, slot_hours).weighted_tardiness
    if linear >= combinatorial:
        return linear, "LP relaxation"
    return combinatorial, "unavoidable tardiness"
