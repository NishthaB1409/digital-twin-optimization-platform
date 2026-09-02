"""FastAPI service that serves dispatch weights from a trained policy.

The deployment shape this assumes: a shop-floor system posts the current state
of the line every few hours and gets back four numbers to feed its dispatching
rule. That keeps the integration surface tiny -- the caller never has to model
jobs, and the service never has to hold state between requests.

Run it with::

    uvicorn dtmo.serving.app:app --reload
    DTMO_MODEL=runs/ppo_shaped/ppo_best uvicorn dtmo.serving.app:app
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from .. import __version__
from ..agents.policies import BEST_KNOWN_BLEND, ConstantPolicy, SB3Policy
from ..digital_twin.dispatch import CLASSICAL_RULES, FEATURE_NAMES
from ..digital_twin.factory import FactoryModel
from ..env.factory_env import OBS_DIM, encode_observation
from ..optimization.lp import best_weighted_tardiness_bound
from ..utils.config import load_config
from .schemas import (
    DispatchWeights,
    FloorState,
    HealthResponse,
    LineInfo,
    SimulationRequest,
    SimulationResponse,
    StationKPIs,
    WeightsResponse,
)

#: Where to find a trained agent. Absent is fine -- the service falls back to
#: the best known fixed rule rather than refusing to start, so a deployment
#: without a model still schedules sensibly.
MODEL_ENV_VAR = "DTMO_MODEL"
CONFIG_ENV_VAR = "DTMO_CONFIG"


@lru_cache(maxsize=1)
def get_config():
    return load_config(os.environ.get(CONFIG_ENV_VAR) or None)


@lru_cache(maxsize=1)
def get_policy() -> tuple[Any, str | None]:
    """Load the trained policy once, falling back to a fixed rule."""
    path = os.environ.get(MODEL_ENV_VAR)
    # Stable-Baselines3 saves to "<path>.zip" but loads from either spelling.
    if path and (Path(path).exists() or Path(f"{path}.zip").exists()):
        try:
            return SB3Policy.load(path, name="ppo"), path
        except Exception:  # noqa: BLE001 -- degraded service beats no service
            pass
    return ConstantPolicy("blend", BEST_KNOWN_BLEND), None


def named_policy(name: str):
    """Resolve a policy name, including the loaded agent."""
    if name == "ppo":
        policy, _ = get_policy()
        return policy
    if name == "blend":
        return ConstantPolicy("blend", BEST_KNOWN_BLEND)
    if name in CLASSICAL_RULES:
        return ConstantPolicy(name, CLASSICAL_RULES[name])
    raise HTTPException(
        status_code=404,
        detail=(
            f"unknown policy {name!r}; available: "
            f"{['ppo', 'blend', *sorted(CLASSICAL_RULES)]}"
        ),
    )


def describe_rule(weights: np.ndarray) -> str:
    """Plain-language reading of a weight vector.

    Lower score dispatches first, so a positive processing-time weight means
    short jobs go first. Worth returning: four signed floats are not something
    a scheduler can sanity-check at a glance.
    """
    readings = {
        "processing_time": ("favours short jobs", "favours long jobs"),
        "slack": ("favours urgent jobs", "favours slack jobs"),
        "remaining_work": ("favours nearly-finished jobs", "favours fresh jobs"),
        "waiting_time": ("favours recent arrivals", "favours long waiters"),
    }
    ranked = sorted(
        zip(FEATURE_NAMES, weights), key=lambda pair: -abs(pair[1])
    )
    parts = [
        readings[name][0 if value > 0 else 1]
        for name, value in ranked[:2]
        if abs(value) > 0.05
    ]
    return ", then ".join(parts) if parts else "no strong preference"


app = FastAPI(
    title="DTMO Scheduling API",
    version=__version__,
    description=(
        "Serves dispatch weights for a 6-station aerospace line. Post the state "
        "of the floor, get back the four weights to drive your dispatching rule."
    ),
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    policy, _ = get_policy()
    return HealthResponse(
        status="ok", version=__version__, policy_loaded=policy.name
    )


@app.get("/info", response_model=LineInfo, tags=["ops"])
def info() -> LineInfo:
    config = get_config()
    policy, path = get_policy()
    return LineInfo(
        stations=list(config.station_names),
        station_capacities={s.name: s.capacity for s in config.stations},
        families=[f.name for f in config.families],
        n_jobs=config.n_jobs,
        policy_loaded=policy.name,
        model_path=path,
        available_policies=["ppo", "blend", *sorted(CLASSICAL_RULES)],
        observation_features=OBS_DIM,
    )


@app.post("/weights", response_model=WeightsResponse, tags=["scheduling"])
def weights(state: FloorState, policy: str = "ppo") -> WeightsResponse:
    """Given the floor right now, return the weights to dispatch with."""
    config = get_config()
    expected = list(config.station_names)
    received = [station.name for station in state.stations]
    if received != expected:
        raise HTTPException(
            status_code=422,
            detail=(
                "stations must match the configured line, in order. "
                f"expected {expected}, got {received}"
            ),
        )

    capacities = {s.name: s.capacity for s in config.stations}
    for station in state.stations:
        if station.busy_machines > capacities[station.name]:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{station.name} reports {station.busy_machines} busy machines "
                    f"but has capacity {capacities[station.name]}"
                ),
            )

    horizon = 2.0 * config.n_jobs / config.arrival_rate
    slack_scale = float(np.mean([f.planned_work for f in config.families]))
    observation = encode_observation(
        queue_lengths=[s.queue_length for s in state.stations],
        busy_fractions=[
            s.busy_machines / capacities[s.name] for s in state.stations
        ],
        clock_hours=state.clock_hours,
        horizon_hours=horizon,
        jobs_completed=state.jobs_completed,
        jobs_in_progress=state.jobs_in_progress,
        total_jobs=state.total_jobs,
        mean_slack_hours=state.mean_slack_hours,
        slack_scale=slack_scale,
    )

    chosen = named_policy(policy)
    action = np.asarray(chosen.act(observation), dtype=float).reshape(-1)
    action = np.clip(action, -1.0, 1.0)
    norm = float(np.linalg.norm(action))
    if norm > 1e-6:
        # The rule is scale-invariant, so return the direction. Keeps responses
        # comparable across calls and matches what the env feeds the simulator.
        action = action / norm

    return WeightsResponse(
        weights=DispatchWeights.from_sequence(action),
        policy=chosen.name,
        rule=describe_rule(action),
        observation=[float(v) for v in observation],
    )


@app.post("/simulate", response_model=SimulationResponse, tags=["scheduling"])
def simulate(request: SimulationRequest) -> SimulationResponse:
    """Run the twin end to end and report KPIs, with a lower bound."""
    if request.weights is None and request.policy is None:
        raise HTTPException(
            status_code=422, detail="provide either `weights` or `policy`"
        )

    config = get_config()
    if request.n_jobs is not None:
        config = config.with_overrides(n_jobs=request.n_jobs)

    if request.weights is not None:
        vector = np.array(request.weights.as_tuple(), dtype=float)
        label = "custom"
    else:
        chosen = named_policy(request.policy)
        if hasattr(chosen, "weights"):
            vector = np.asarray(chosen.weights, dtype=float)
        else:
            # A state-dependent policy cannot be reduced to one vector, so run
            # the twin under its opening move rather than pretending otherwise.
            vector = np.asarray(
                chosen.act(np.zeros(OBS_DIM, dtype=np.float32)), dtype=float
            )
        label = chosen.name

    model = FactoryModel(config, weights=vector, seed=request.seed)
    kpis = model.run()

    bound = None
    try:
        bound, _ = best_weighted_tardiness_bound(model.jobs, config.stations, 4.0)
    except Exception:  # noqa: BLE001 -- the bound is a nicety, not the answer
        bound = None

    return SimulationResponse(
        policy=label,
        seed=request.seed,
        jobs_completed=kpis.jobs_completed,
        makespan_hours=kpis.makespan,
        mean_flow_time_hours=kpis.mean_flow_time,
        on_time_rate=kpis.on_time_rate,
        total_weighted_tardiness=kpis.total_weighted_tardiness,
        mean_utilisation=kpis.mean_utilisation,
        stations=[
            StationKPIs(
                name=name,
                utilisation=kpis.station_utilisation[name],
                mean_queue_length=kpis.station_mean_queue[name],
            )
            for name in kpis.station_utilisation
        ],
        weighted_tardiness_lower_bound=bound,
    )


@app.get("/policies", tags=["scheduling"])
def policies() -> dict[str, Any]:
    """Every policy this server can dispatch with."""
    loaded, path = get_policy()
    return {
        "loaded": loaded.name,
        "model_path": path,
        "classical": {
            name: dict(zip(FEATURE_NAMES, weights))
            for name, weights in sorted(CLASSICAL_RULES.items())
        },
        "tuned": {"blend": dict(zip(FEATURE_NAMES, BEST_KNOWN_BLEND))},
    }


# Registered last: live_routes imports get_config and named_policy from this
# module, so the import has to happen after they exist.
from .live_routes import router as live_router  # noqa: E402

app.include_router(live_router)
