"""Day 1 end-to-end run: build the twin, simulate, print KPIs.

    python scripts/run_simulation.py
    python scripts/run_simulation.py --rule min_slack
    python scripts/run_simulation.py --weights 0.24 0.99 0.90 -0.08
    python scripts/run_simulation.py --compare
"""

from __future__ import annotations

import argparse

from dtmo.digital_twin import CLASSICAL_RULES, FactoryModel
from dtmo.utils.config import DEFAULT_CONFIG_PATH, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="factory YAML to load"
    )
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument("--jobs", type=int, default=None, help="override n_jobs")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--rule",
        choices=sorted(CLASSICAL_RULES),
        help="use a named classical rule instead of the config weights",
    )
    group.add_argument(
        "--weights",
        type=float,
        nargs=4,
        metavar=("PROC", "SLACK", "REMAIN", "WAIT"),
        help="explicit dispatch weights",
    )
    group.add_argument(
        "--compare",
        action="store_true",
        help="run every classical rule on the same seed and rank them",
    )
    return parser


def resolve_weights(args) -> tuple[float, ...] | None:
    if args.rule:
        return CLASSICAL_RULES[args.rule]
    if args.weights:
        return tuple(args.weights)
    return None


def print_expected_load(config) -> None:
    """Analytic load, before any simulating.

    Worth reading first: above 100% the station is a hard bottleneck and queues
    grow without bound, which is a capacity problem no dispatch rule can fix.
    """
    print("expected station load (analytic, before simulating):")
    for name, load in config.expected_utilisation().items():
        flag = "  <-- OVER CAPACITY" if load >= 1.0 else ""
        print(f"    {name:<20} {load:>6.1%}{flag}")


def print_dispatch_leverage(model) -> None:
    """How often the rule actually chose between two or more waiting jobs.

    A station at 0% never exercises the dispatch rule, so no amount of weight
    tuning changes anything there. This is the diagnostic that shows whether
    the RL agent has leverage at all.
    """
    decisions = sum(s.dispatch_decisions for s in model.stations.values())
    contested = sum(s.contested_decisions for s in model.stations.values())
    width = max(len(name) for name in model.stations)
    print()
    print(f"  {'DISPATCH LEVERAGE'.ljust(width)}   {'CHOICES':>8}  {'CONTESTED':>10}")
    for name, station in model.stations.items():
        print(
            f"  {name.ljust(width)}   {station.dispatch_decisions:>8}  "
            f"{station.contested_fraction:>10.1%}"
        )
    overall = contested / decisions if decisions else 0.0
    print(f"  {'overall'.ljust(width)}   {decisions:>8}  {overall:>10.1%}")


def run_comparison(config) -> None:
    """Same jobs, same seed, one row per rule -- the Day 4 baseline table."""
    rows = []
    for name, weights in sorted(CLASSICAL_RULES.items()):
        rows.append((name, FactoryModel(config, weights=weights).run()))

    header = (
        f"{'RULE':<10} {'MAKESPAN':>9} {'FLOW':>8} {'ON-TIME':>8} "
        f"{'WGT TARD':>10} {'UTIL':>7}"
    )
    print()
    print(header)
    print("-" * len(header))
    for name, kpis in sorted(rows, key=lambda r: r[1].total_weighted_tardiness):
        print(
            f"{name:<10} {kpis.makespan:>9.1f} {kpis.mean_flow_time:>8.1f} "
            f"{kpis.on_time_rate:>7.1%} {kpis.total_weighted_tardiness:>10.1f} "
            f"{kpis.mean_utilisation:>6.1%}"
        )
    print("-" * len(header))
    best = min(rows, key=lambda r: r[1].total_weighted_tardiness)
    print(f"best by weighted tardiness: {best[0]} ({best[1].total_weighted_tardiness:.1f})")


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.jobs is not None:
        overrides["n_jobs"] = args.jobs
    if overrides:
        config = config.with_overrides(**overrides)

    print(f"config: {args.config}")
    print_expected_load(config)

    if args.compare:
        run_comparison(config)
        return

    model = FactoryModel(config, weights=resolve_weights(args))
    print()
    print(f"running {model!r}")
    print()
    kpis = model.run()
    print(kpis.summary())
    print_dispatch_leverage(model)


if __name__ == "__main__":
    main()
