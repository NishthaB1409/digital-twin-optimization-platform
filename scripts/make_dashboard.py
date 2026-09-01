"""Day 5: render the factory dashboard to a self-contained HTML file.

    python scripts/make_dashboard.py
    python scripts/make_dashboard.py --seed 1000 --out runs/dashboard.html
    python scripts/make_dashboard.py --policy blend

Opens in any browser, no server needed.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from dtmo.agents.policies import BEST_KNOWN_BLEND, ConstantPolicy, classical_policies
from dtmo.agents.train import baseline_by_name, make_eval_env
from dtmo.digital_twin.dispatch import CLASSICAL_RULES
from dtmo.digital_twin.factory import FactoryModel
from dtmo.evaluation.paired import benchmark
from dtmo.optimization.lp import best_weighted_tardiness_bound
from dtmo.utils.config import load_config
from dtmo.visualization.charts import (
    gantt,
    learning_curve,
    policy_comparison,
    station_load,
)
from dtmo.visualization.dashboard import StatTile, build_dashboard
from dtmo.visualization.theme import DARK, LIGHT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=1000, help="instance to draw")
    parser.add_argument("--seeds", type=int, default=12, help="seeds for benchmarking")
    parser.add_argument("--policy", default="spt", help="policy for the Gantt")
    parser.add_argument("--max-jobs", type=int, default=45, help="jobs on the Gantt")
    parser.add_argument("--out", default="runs/dashboard.html")
    parser.add_argument(
        "--history",
        default="runs/ppo_shaped/history.json",
        help="training history for the learning curve (skipped if absent)",
    )
    return parser


def main() -> None:
    warnings.simplefilter("ignore")
    args = build_parser().parse_args()
    config = load_config(args.config)

    weights = (
        BEST_KNOWN_BLEND
        if args.policy == "blend"
        else CLASSICAL_RULES.get(args.policy)
    )
    if weights is None:
        raise SystemExit(
            f"unknown policy {args.policy!r}; choose 'blend' or one of "
            f"{sorted(CLASSICAL_RULES)}"
        )

    print(f"simulating seed {args.seed} under {args.policy} ...", flush=True)
    model = FactoryModel(config, weights=weights, seed=args.seed)
    kpis = model.run()

    print(f"benchmarking policies over {args.seeds} paired seeds ...", flush=True)
    env = make_eval_env(config)
    seeds = list(range(1000, 1000 + args.seeds))
    results = benchmark(classical_policies(), env, seeds)
    tardiness = {
        name: result.mean("total_weighted_tardiness")
        for name, result in results.items()
    }

    print("solving the lower bound ...", flush=True)
    bounds = []
    for seed in seeds:
        instance = FactoryModel(config, seed=seed)
        instance.reset()
        value, _ = best_weighted_tardiness_bound(instance.jobs, config.stations, 2.0)
        bounds.append(value)
    lower_bound = float(np.mean(bounds))

    # The contenders: everything not significantly worse than the best.
    ranked = sorted(tardiness, key=tardiness.get)
    contenders = [name for name in ranked[:3]]

    station_order = list(config.station_names)
    figures = {
        "gantt": (
            gantt(model.completed, station_order, LIGHT, args.max_jobs),
            gantt(model.completed, station_order, DARK, args.max_jobs),
        ),
        "policies": (
            policy_comparison(
                tardiness, LIGHT, highlight=contenders, lower_bound=lower_bound
            ),
            policy_comparison(
                tardiness, DARK, highlight=contenders, lower_bound=lower_bound
            ),
        ),
        "load": (
            station_load(
                kpis.station_utilisation, LIGHT, config.expected_utilisation()
            ),
            station_load(
                kpis.station_utilisation, DARK, config.expected_utilisation()
            ),
        ),
    }

    history_path = Path(args.history)
    if history_path.exists():
        history = json.loads(history_path.read_text())
        figures["learning"] = (
            learning_curve(history, LIGHT),
            learning_curve(history, DARK),
        )
    else:
        print(f"note: {history_path} not found; skipping the learning curve")

    shown = model.completed[: args.max_jobs]
    family_counts = Counter()
    for job in shown:
        family_counts[job.family.name] += len(job.op_log)

    best_name = ranked[0]
    tiles = [
        StatTile("On-time rate", f"{kpis.on_time_rate:.0%}", f"seed {args.seed}"),
        StatTile("Mean flow time", f"{kpis.mean_flow_time:.0f} h", "release to delivery"),
        StatTile("Makespan", f"{kpis.makespan:.0f} h", "last job out"),
        StatTile(
            "Weighted tardiness",
            f"{tardiness[best_name]:,.0f}",
            f"best policy ({best_name})",
        ),
        StatTile(
            "Lower bound",
            f"{lower_bound:,.0f}",
            "no schedule beats this",
        ),
    ]

    notes = {
        "gantt": (
            "Each row is a work centre and each bar one operation, so gaps are idle "
            f"capacity. Showing the first {len(shown)} jobs of seed {args.seed} under "
            f"{args.policy}; the full run is {config.n_jobs} jobs."
        ),
        "policies": (
            "Every policy scored on the same seeds. The three contenders are "
            "highlighted; they are statistically tied with each other, and the gap "
            "to the dotted bound is an upper limit on what better scheduling could "
            "recover, not a promise that it can be recovered."
        ),
        "load": (
            "Simulated utilisation against the load predicted analytically before "
            "running anything. Heat Treatment, Surface Treatment and Inspection are "
            "the constraint; the others have slack."
        ),
        "learning": (
            "Paired against the baseline on held-out seeds, so this is not moved by "
            "which job sets came up. The agent climbs to parity and stays there."
        ),
    }

    html = build_dashboard(
        figures=figures,
        tiles=tiles,
        family_counts=family_counts,
        title="Factory Twin Dashboard",
        subtitle=(
            f"{config.n_jobs} jobs across {len(config.stations)} stations. "
            f"Policies compared on {len(seeds)} paired seeds against a provable "
            "lower bound."
        ),
        notes=notes,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\nwrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"open it with:  start {out}")


if __name__ == "__main__":
    main()
