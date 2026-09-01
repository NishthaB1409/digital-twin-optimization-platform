"""Day 4: benchmark every policy against provable lower bounds.

Day 3 ended with PPO tied against SPT, which on its own does not say whether
the agent underperformed or whether SPT is already near the ceiling. This
script answers that by computing what *no* schedule could beat, and measuring
how much of each policy's cost is forced by the instance rather than chosen by
the scheduler.

    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --seeds 24 --slot-hours 1
    python scripts/run_benchmark.py --model runs/ppo_shaped/ppo_best
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from dtmo.agents.policies import SB3Policy, classical_policies
from dtmo.agents.train import make_eval_env
from dtmo.evaluation.paired import benchmark, compare
from dtmo.optimization.bounds import bounds_for_seed
from dtmo.digital_twin.factory import FactoryModel
from dtmo.optimization.lp import best_weighted_tardiness_bound, lp_bound_for_seed
from dtmo.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", type=int, default=16, help="paired episodes")
    parser.add_argument(
        "--slot-hours",
        type=float,
        default=2.0,
        help="LP time discretisation; smaller is tighter and slower",
    )
    parser.add_argument(
        "--model",
        default="runs/ppo_shaped/ppo_best",
        help="trained agent to include (omit with --no-model)",
    )
    parser.add_argument("--no-model", action="store_true")
    return parser


def main() -> None:
    warnings.simplefilter("ignore")
    args = build_parser().parse_args()
    config = load_config(args.config)
    env = make_eval_env(config)
    seeds = list(range(1000, 1000 + args.seeds))

    policies = classical_policies()
    if not args.no_model:
        try:
            policies = policies + [SB3Policy.load(args.model, name="ppo")]
        except Exception as exc:  # noqa: BLE001 -- report and carry on
            print(f"note: could not load {args.model} ({exc}); continuing without it\n")

    print(f"line   : {config.n_jobs} jobs, {len(config.stations)} stations")
    print(f"seeds  : {seeds[0]}..{seeds[-1]} ({len(seeds)} paired episodes)")
    print(f"LP     : {args.slot_hours:g}h slots\n")

    print("solving lower bounds ...", flush=True)
    simple = [bounds_for_seed(config, s) for s in seeds]
    lp = [lp_bound_for_seed(config, s, args.slot_hours) for s in seeds]
    # Neither relaxation dominates the other, so report the stronger per seed.
    combined = []
    for seed in seeds:
        model = FactoryModel(config, seed=seed)
        model.reset()
        value, _ = best_weighted_tardiness_bound(
            model.jobs, config.stations, args.slot_hours
        )
        combined.append(value)
    lb_tard = np.array(combined)
    lb_simple = np.array([b.weighted_tardiness for b in simple])
    lb_make = np.array([b.makespan for b in simple])

    binding = {}
    for b in lp:
        binding[b.binding_station] = binding.get(b.binding_station, 0) + 1
    binding_station = max(binding, key=binding.get)

    print(f"  unavoidable tardiness (no contention) : {lb_simple.mean():8.1f}")
    print(f"  LP relaxation (keeps station capacity): "
          f"{np.mean([b.weighted_tardiness for b in lp]):8.1f}")
    print(f"  strongest valid bound (max of the two): {lb_tard.mean():8.1f}")
    print(f"  binding station                       : {binding_station} "
          f"({binding[binding_station]}/{len(seeds)} seeds)")
    print(f"  makespan lower bound                  : {lb_make.mean():8.1f}\n")

    results = benchmark(policies, env, seeds)

    header = (
        f"{'POLICY':<10} {'WGT TARD':>10} {'/ LP LB':>9} {'FORCED':>8} "
        f"{'MAKESPAN':>10} {'/ LB':>7}"
    )
    print(header)
    print("-" * len(header))
    rows = sorted(results.values(), key=lambda r: r.mean("total_weighted_tardiness"))
    for result in rows:
        tard = result.metric("total_weighted_tardiness")
        make = result.metric("makespan")
        # Share of the achieved cost that the instance forces on any scheduler.
        forced = float(np.mean(lb_tard / tard))
        print(
            f"{result.name:<10} {tard.mean():>10.1f} {np.mean(tard / lb_tard):>8.2f}x "
            f"{forced:>7.0%} {make.mean():>10.1f} {np.mean(make / lb_make):>6.2f}x"
        )
    print("-" * len(header))
    print(f"{'LOWER BND':<10} {lb_tard.mean():>10.1f} {'1.00x':>9} {'100%':>8} "
          f"{lb_make.mean():>10.1f} {'1.00x':>7}")

    best = rows[0]
    print(f"\npaired against the best policy ({best.name}):")
    for result in rows[1:]:
        c = compare(result, best, metric="total_weighted_tardiness")
        verdict = "tie" if not c.is_significant else "worse"
        print(
            f"  {result.name:<10} {c.improvement_pct:>+7.1f}%  "
            f"wins {c.win_rate:>4.0%}  p={c.p_value:>6.3f}  {verdict}"
        )

    ratio = float(np.mean(best.metric("total_weighted_tardiness") / lb_tard))
    make_ratio = float(np.mean(best.metric("makespan") / lb_make))
    print(f"\nREADING THIS")
    print(
        f"  Makespan is nearly settled: the best policy sits {make_ratio:.2f}x a bound\n"
        f"  no schedule can beat, so at most {1 - 1 / make_ratio:.0%} of it is schedulable away."
    )
    print(
        f"  Tardiness is not: the best policy is {ratio:.2f}x the LP bound. That gap is\n"
        f"  an upper limit on the remaining room, not a promise -- the LP relaxes\n"
        f"  preemption and every station but {binding_station}, so the true optimum sits\n"
        f"  somewhere between the two, and an exact solve would be needed to place it."
    )


if __name__ == "__main__":
    main()
