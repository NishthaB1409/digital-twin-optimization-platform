"""SimPy work centre with a pluggable dispatch rule."""

from __future__ import annotations

from typing import Callable

import simpy

from .dispatch import CompositeDispatchRule
from .entities import Job, StationSpec


class Station:
    """A work centre with ``capacity`` identical parallel machines.

    Neither of SimPy's built-in resources fits here. ``Resource`` serves its
    queue FIFO, and ``PriorityResource`` freezes a job's priority at the moment
    it requests -- but slack and waiting time keep changing while a job sits in
    the queue, so a priority computed on arrival is stale by the time a machine
    frees up.

    So the queue is a plain list and each machine is its own SimPy process that
    re-scores the whole queue at the instant it becomes free. That is the
    correct semantics for a dispatching rule: the decision uses the state at
    the decision point.
    """

    def __init__(
        self,
        env: simpy.Environment,
        spec: StationSpec,
        dispatcher: CompositeDispatchRule,
        on_operation_complete: Callable[[Job], None],
    ) -> None:
        self.env = env
        self.name = spec.name
        self.capacity = spec.capacity
        self.dispatcher = dispatcher
        self._on_operation_complete = on_operation_complete

        self.queue: list[Job] = []
        self.busy_machines = 0
        self.busy_time = 0.0
        self.operations_completed = 0
        #: Dispatch decisions taken, and how many had a real choice to make.
        #: If `contested` is near zero the queue is never deep enough for the
        #: dispatch rule to matter -- tune the load before blaming the rule.
        self.dispatch_decisions = 0
        self.contested_decisions = 0
        #: Time-weighted queue length accumulator, for utilisation-style stats.
        self.queue_time_integral = 0.0
        self._last_queue_change = 0.0

        # One process per machine. `capacity` processes can each hold at most
        # one job, so the capacity constraint holds by construction.
        self._work_available = env.event()
        for _ in range(spec.capacity):
            env.process(self._machine())

    def __repr__(self) -> str:
        return (
            f"Station({self.name!r}, capacity={self.capacity}, "
            f"queued={len(self.queue)}, busy={self.busy_machines})"
        )

    # ------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------
    def submit(self, job: Job) -> None:
        """Put a job in this station's queue and wake an idle machine."""
        self._accrue_queue_time()
        job.queue_entry_time = self.env.now
        self.queue.append(job)
        self._signal_work()

    def _accrue_queue_time(self) -> None:
        self.queue_time_integral += len(self.queue) * (
            self.env.now - self._last_queue_change
        )
        self._last_queue_change = self.env.now

    def _signal_work(self) -> None:
        """Fire the current wake-up event and install a fresh one.

        Machines waiting on the old event resume; machines that go back to
        sleep afterwards wait on the new one. Swapping before firing keeps a
        machine from re-waiting on an already-triggered event and spinning.
        """
        event, self._work_available = self._work_available, self.env.event()
        if not event.triggered:
            event.succeed()

    @property
    def queue_length(self) -> int:
        return len(self.queue)

    @property
    def is_idle(self) -> bool:
        return self.busy_machines == 0

    def utilisation(self, horizon: float) -> float:
        """Fraction of available machine-hours that were spent processing."""
        if horizon <= 0:
            return 0.0
        return self.busy_time / (self.capacity * horizon)

    def mean_queue_length(self, horizon: float) -> float:
        if horizon <= 0:
            return 0.0
        return self.queue_time_integral / horizon

    # ------------------------------------------------------------------
    # Machine process
    # ------------------------------------------------------------------
    def _machine(self):
        while True:
            # `while`, not `if`: several machines wake on the same event but
            # only one of them gets the job that caused it.
            while not self.queue:
                yield self._work_available

            self.dispatch_decisions += 1
            if len(self.queue) > 1:
                self.contested_decisions += 1
            job = self.dispatcher.select(self.queue, self.env.now)
            self._accrue_queue_time()
            self.queue.remove(job)

            start = self.env.now
            duration = job.current_processing_time
            self.busy_machines += 1
            yield self.env.timeout(duration)
            self.busy_machines -= 1

            self.busy_time += duration
            self.operations_completed += 1
            job.op_log.append((self.name, start, self.env.now))
            job.op_index += 1
            self._on_operation_complete(job)

    def finalise(self, at: float) -> None:
        """Close the time-weighted queue integral at the end of a run.

        Without this the interval between the last queue change and the end of
        the horizon is silently dropped.
        """
        # Clamp: kpis() may be called mid-episode, where the last completion
        # time trails the clock. Without this a later call would subtract.
        at = max(at, self._last_queue_change)
        self.queue_time_integral += len(self.queue) * (at - self._last_queue_change)
        self._last_queue_change = at

    @property
    def contested_fraction(self) -> float:
        """Share of dispatches that chose between two or more waiting jobs."""
        if not self.dispatch_decisions:
            return 0.0
        return self.contested_decisions / self.dispatch_decisions
