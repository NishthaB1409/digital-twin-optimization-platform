"""Request and response models for the scheduling API.

The API speaks floor state, not observation vectors. A caller integrating with
an MES knows how many jobs are queued at Heat Treatment; it has no business
knowing that queue lengths are tanh-squashed at a scale of 5. Encoding is the
server's job, and it uses the same function the training environment does.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StationState(BaseModel):
    """What one work centre looks like right now."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Station name, matching the configured line")
    queue_length: int = Field(..., ge=0, description="Jobs waiting, not yet started")
    busy_machines: int = Field(..., ge=0, description="Machines currently processing")

    @field_validator("name")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("station name must not be blank")
        return value


class FloorState(BaseModel):
    """A snapshot of the line, as a scheduler would see it."""

    model_config = ConfigDict(extra="forbid")

    stations: list[StationState] = Field(..., min_length=1)
    clock_hours: float = Field(..., ge=0, description="Hours since the run started")
    jobs_completed: int = Field(0, ge=0)
    jobs_in_progress: int = Field(0, ge=0, description="Released but not finished")
    total_jobs: int = Field(..., gt=0, description="Jobs in the production plan")
    mean_slack_hours: float = Field(
        0.0,
        description=(
            "Mean spare time before due dates across work in progress. "
            "Negative means the average job can no longer make its date."
        ),
    )

    @field_validator("stations")
    @classmethod
    def _unique_names(cls, value: list[StationState]) -> list[StationState]:
        names = [station.name for station in value]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate station names: {names}")
        return value


class DispatchWeights(BaseModel):
    """The four numbers that drive the dispatch rule."""

    model_config = ConfigDict(extra="forbid")

    processing_time: float
    slack: float
    remaining_work: float
    waiting_time: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.processing_time,
            self.slack,
            self.remaining_work,
            self.waiting_time,
        )

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "DispatchWeights":
        if len(values) != 4:
            raise ValueError(f"expected 4 weights, got {len(values)}")
        return cls(
            processing_time=float(values[0]),
            slack=float(values[1]),
            remaining_work=float(values[2]),
            waiting_time=float(values[3]),
        )


class WeightsResponse(BaseModel):
    """What the policy decided, and enough context to audit it."""

    model_config = ConfigDict(extra="forbid")

    weights: DispatchWeights
    policy: str = Field(..., description="Which policy produced these weights")
    rule: str = Field(
        ...,
        description=(
            "Plain-language reading of the weights, e.g. 'favours short jobs, "
            "then urgent ones'"
        ),
    )
    observation: list[float] = Field(
        ..., description="The 16 features the policy actually saw"
    )


class SimulationRequest(BaseModel):
    """Run a what-if on the digital twin."""

    model_config = ConfigDict(extra="forbid")

    weights: DispatchWeights | None = Field(
        None, description="Explicit weights; omit to use `policy` instead"
    )
    policy: str | None = Field(
        None, description="A named rule such as 'spt', 'blend', or 'ppo'"
    )
    seed: int = Field(1000, description="Which job set to generate")
    n_jobs: int | None = Field(None, gt=0, le=1000)


class StationKPIs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    utilisation: float
    mean_queue_length: float


class SimulationResponse(BaseModel):
    """KPIs from one simulated run."""

    model_config = ConfigDict(extra="forbid")

    policy: str
    seed: int
    jobs_completed: int
    makespan_hours: float
    mean_flow_time_hours: float
    on_time_rate: float
    total_weighted_tardiness: float
    mean_utilisation: float
    stations: list[StationKPIs]
    #: Provable floor for this instance -- no schedule can beat it.
    weighted_tardiness_lower_bound: float | None = None


class LineInfo(BaseModel):
    """What line this server is configured for, and what is loaded."""

    # `model_path` is a deliberate field name -- it reads correctly in the API.
    # Clearing the protected namespace silences pydantic's `model_` warning
    # without renaming the field to something worse.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    stations: list[str]
    station_capacities: dict[str, int]
    families: list[str]
    n_jobs: int
    policy_loaded: str
    model_path: str | None
    available_policies: list[str]
    observation_features: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    version: str
    policy_loaded: str
