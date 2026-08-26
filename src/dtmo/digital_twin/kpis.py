"""KPI extraction from a finished simulation run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Iterable, Mapping

from .entities import Job


@dataclass(frozen=True)
class KPIs:
    """The six headline metrics, plus the per-station detail behind them."""

    jobs_released: int
    jobs_completed: int
    makespan: float
    throughput: float                 # completed jobs per hour
    mean_flow_time: float
    max_flow_time: float
    on_time_rate: float               # 0..1
    total_weighted_tardiness: float
    mean_tardiness: float
    max_tardiness: float
    mean_utilisation: float           # 0..1
    station_utilisation: Mapping[str, float]
    station_mean_queue: Mapping[str, float]

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """A human-readable block, for the Day 1 end-to-end run."""
        width = max(len(name) for name in self.station_utilisation) if self.station_utilisation else 10
        lines = [
            "=" * (width + 34),
            "FACTORY KPIs",
            "=" * (width + 34),
            f"  Jobs released           {self.jobs_released:>10d}",
            f"  Jobs completed          {self.jobs_completed:>10d}",
            f"  Makespan                {self.makespan:>10.1f} h",
            f"  Throughput              {self.throughput:>10.3f} jobs/h",
            f"  Mean flow time          {self.mean_flow_time:>10.1f} h",
            f"  Max flow time           {self.max_flow_time:>10.1f} h",
            f"  On-time rate            {self.on_time_rate:>10.1%}",
            f"  Mean tardiness          {self.mean_tardiness:>10.1f} h",
            f"  Max tardiness           {self.max_tardiness:>10.1f} h",
            f"  Total wgt. tardiness    {self.total_weighted_tardiness:>10.1f}",
            f"  Mean utilisation        {self.mean_utilisation:>10.1%}",
            "-" * (width + 34),
            f"  {'STATION'.ljust(width)}   {'UTIL':>7}  {'MEAN Q':>8}",
        ]
        for name, util in self.station_utilisation.items():
            queue = self.station_mean_queue.get(name, 0.0)
            lines.append(f"  {name.ljust(width)}   {util:>6.1%}  {queue:>8.2f}")
        lines.append("=" * (width + 34))
        return "\n".join(lines)


def compute_kpis(
    jobs: Iterable[Job],
    completed: Iterable[Job],
    stations: Mapping[str, "object"],
) -> KPIs:
    """Roll a finished run up into a :class:`KPIs` record.

    ``makespan`` is the last completion time, not the simulation clock -- the
    clock can drift past the final completion while idle machines wait on
    events that never fire.
    """
    jobs = list(jobs)
    completed = list(completed)
    n_completed = len(completed)

    if n_completed:
        makespan = max(job.completion_time for job in completed)
        flow_times = [job.flow_time for job in completed]
        tardiness = [job.tardiness for job in completed]
        mean_flow_time = fmean(flow_times)
        max_flow_time = max(flow_times)
        mean_tardiness = fmean(tardiness)
        max_tardiness = max(tardiness)
        on_time_rate = sum(job.is_on_time for job in completed) / n_completed
        total_weighted_tardiness = sum(job.weighted_tardiness for job in completed)
        throughput = n_completed / makespan if makespan > 0 else 0.0
    else:
        makespan = 0.0
        mean_flow_time = max_flow_time = 0.0
        mean_tardiness = max_tardiness = 0.0
        on_time_rate = 0.0
        total_weighted_tardiness = 0.0
        throughput = 0.0

    station_utilisation = {
        name: station.utilisation(makespan) for name, station in stations.items()
    }
    station_mean_queue = {
        name: station.mean_queue_length(makespan) for name, station in stations.items()
    }
    mean_utilisation = (
        fmean(station_utilisation.values()) if station_utilisation else 0.0
    )

    return KPIs(
        jobs_released=len(jobs),
        jobs_completed=n_completed,
        makespan=makespan,
        throughput=throughput,
        mean_flow_time=mean_flow_time,
        max_flow_time=max_flow_time,
        on_time_rate=on_time_rate,
        total_weighted_tardiness=total_weighted_tardiness,
        mean_tardiness=mean_tardiness,
        max_tardiness=max_tardiness,
        mean_utilisation=mean_utilisation,
        station_utilisation=station_utilisation,
        station_mean_queue=station_mean_queue,
    )
