"""YAML -> typed configuration objects.

Everything the twin needs to build a line lives in one YAML file, so a run is
described by ``(config file, weights, seed)`` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import yaml

from ..digital_twin.entities import FamilySpec, Operation, StationSpec

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "factory.yaml"


@dataclass(frozen=True)
class FactoryConfig:
    """Fully-resolved description of a line and the run to perform on it."""

    stations: tuple[StationSpec, ...]
    families: tuple[FamilySpec, ...]
    n_jobs: int
    arrival_rate: float
    processing_cv: float
    seed: int
    dispatch_weights: tuple[float, ...]

    @property
    def station_names(self) -> tuple[str, ...]:
        return tuple(station.name for station in self.stations)

    def validate(self) -> "FactoryConfig":
        """Fail loudly on a malformed line rather than mid-simulation."""
        names = self.station_names
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate station names in {names}")
        known = set(names)
        for family in self.families:
            for op in family.route:
                if op.station not in known:
                    raise ValueError(
                        f"family {family.name!r} routes through unknown station "
                        f"{op.station!r}; known stations are {sorted(known)}"
                    )
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1, got {self.n_jobs}")
        if self.arrival_rate <= 0:
            raise ValueError(f"arrival_rate must be > 0, got {self.arrival_rate}")
        if self.processing_cv < 0:
            raise ValueError(f"processing_cv must be >= 0, got {self.processing_cv}")
        return self

    def expected_utilisation(self) -> dict[str, float]:
        """Analytic load check, independent of the simulation.

        Sum of (family mix x that family's work at the station) x arrival rate,
        over machine count. If this exceeds 1 the station is a hard bottleneck
        and queues grow without bound -- worth knowing before blaming the
        dispatch rule.
        """
        mix_total = sum(family.mix for family in self.families)
        work: dict[str, float] = {name: 0.0 for name in self.station_names}
        for family in self.families:
            share = family.mix / mix_total
            for op in family.route:
                work[op.station] += share * op.mean_time
        return {
            station.name: work[station.name] * self.arrival_rate / station.capacity
            for station in self.stations
        }

    def with_overrides(self, **changes: Any) -> "FactoryConfig":
        """A copy with fields replaced -- for sweeps and RL episode resets."""
        return replace(self, **changes).validate()


def _require(mapping: dict, key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"{context}: missing required key {key!r}")
    return mapping[key]


def parse_config(raw: dict) -> FactoryConfig:
    """Turn a parsed YAML mapping into a validated :class:`FactoryConfig`."""
    simulation = raw.get("simulation", {})
    dispatch = raw.get("dispatch", {})

    stations = tuple(
        StationSpec(
            name=_require(entry, "name", "station"),
            capacity=int(_require(entry, "capacity", "station")),
        )
        for entry in _require(raw, "stations", "config")
    )

    families = []
    for entry in _require(raw, "families", "config"):
        name = _require(entry, "name", "family")
        route = tuple(
            Operation(
                station=_require(op, "station", f"family {name!r} route"),
                mean_time=float(_require(op, "mean_time", f"family {name!r} route")),
            )
            for op in _require(entry, "route", f"family {name!r}")
        )
        families.append(
            FamilySpec(
                name=name,
                route=route,
                weight=float(_require(entry, "weight", f"family {name!r}")),
                mix=float(_require(entry, "mix", f"family {name!r}")),
                due_factor=float(_require(entry, "due_factor", f"family {name!r}")),
            )
        )

    return FactoryConfig(
        stations=stations,
        families=tuple(families),
        n_jobs=int(simulation.get("n_jobs", 100)),
        arrival_rate=float(simulation.get("arrival_rate", 0.45)),
        processing_cv=float(simulation.get("processing_cv", 0.0)),
        seed=int(simulation.get("seed", 0)),
        dispatch_weights=tuple(
            float(w) for w in dispatch.get("weights", (1.0, 0.0, 0.0, 0.0))
        ),
    ).validate()


def load_config(path: str | Path | None = None) -> FactoryConfig:
    """Read and validate a factory YAML file."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return parse_config(raw)
