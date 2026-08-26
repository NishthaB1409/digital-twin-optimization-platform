"""Domain objects for the factory digital twin.

The specs (StationSpec, Operation, FamilySpec) are frozen: they describe the
line and never change during a run. Job is mutable -- it carries the state that
moves through the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StationSpec:
    """Static description of one work centre."""

    name: str
    capacity: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("station name must be non-empty")
        if self.capacity < 1:
            raise ValueError(f"station {self.name!r}: capacity must be >= 1")


@dataclass(frozen=True)
class Operation:
    """One step in a product family's route."""

    station: str
    mean_time: float

    def __post_init__(self) -> None:
        if self.mean_time <= 0:
            raise ValueError(
                f"operation at {self.station!r}: mean_time must be > 0"
            )


@dataclass(frozen=True)
class FamilySpec:
    """A product family: a fixed route plus its commercial parameters."""

    name: str
    route: tuple[Operation, ...]
    weight: float       # tardiness weight -- how much lateness hurts
    mix: float          # share of released jobs
    due_factor: float   # due = release + due_factor * planned work

    def __post_init__(self) -> None:
        if not self.route:
            raise ValueError(f"family {self.name!r}: route must be non-empty")
        if self.weight <= 0:
            raise ValueError(f"family {self.name!r}: weight must be > 0")
        if self.mix <= 0:
            raise ValueError(f"family {self.name!r}: mix must be > 0")
        if self.due_factor <= 0:
            raise ValueError(f"family {self.name!r}: due_factor must be > 0")

    @property
    def planned_work(self) -> float:
        """Total standard processing time over the route."""
        return float(sum(op.mean_time for op in self.route))

    @property
    def n_operations(self) -> int:
        return len(self.route)


@dataclass
class Job:
    """One work order moving through the line.

    ``proc_times`` holds the *realised* processing time for each operation,
    sampled once at creation so a run is fully reproducible from its seed.
    """

    job_id: int
    family: FamilySpec
    release_time: float
    due_date: float
    proc_times: tuple[float, ...]

    op_index: int = 0
    queue_entry_time: float = 0.0
    completion_time: float | None = None
    # (station, start, finish) per completed operation
    op_log: list[tuple[str, float, float]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Position along the route
    # ------------------------------------------------------------------
    @property
    def is_done(self) -> bool:
        return self.op_index >= len(self.proc_times)

    @property
    def current_station(self) -> str:
        return self.family.route[self.op_index].station

    @property
    def current_processing_time(self) -> float:
        return self.proc_times[self.op_index]

    @property
    def remaining_work(self) -> float:
        """Processing time still to be done, including the current operation."""
        return float(sum(self.proc_times[self.op_index:]))

    @property
    def total_work(self) -> float:
        return float(sum(self.proc_times))

    @property
    def completion_fraction(self) -> float:
        return self.op_index / len(self.proc_times)

    # ------------------------------------------------------------------
    # Dispatch features
    # ------------------------------------------------------------------
    def slack(self, now: float) -> float:
        """Spare time before the due date if all remaining work ran back to back.

        Negative slack means the job is already too late to make its date.
        """
        return self.due_date - now - self.remaining_work

    def waiting_time(self, now: float) -> float:
        return now - self.queue_entry_time

    # ------------------------------------------------------------------
    # KPI helpers -- only meaningful once the job has finished
    # ------------------------------------------------------------------
    def _require_complete(self) -> float:
        if self.completion_time is None:
            raise ValueError(f"job {self.job_id} has not completed")
        return self.completion_time

    @property
    def flow_time(self) -> float:
        return self._require_complete() - self.release_time

    @property
    def tardiness(self) -> float:
        return max(0.0, self._require_complete() - self.due_date)

    @property
    def weighted_tardiness(self) -> float:
        return self.family.weight * self.tardiness

    @property
    def is_on_time(self) -> bool:
        return self._require_complete() <= self.due_date
