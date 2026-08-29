"""Score fixed policies through the Gymnasium environment.

Each classical rule is held constant for a whole episode, which turns it into a
fixed policy the env can score. That gives the baseline a learned policy has to
beat, in the same reward units the agent actually sees.

Every policy runs over the *same* seed list and results are reported as paired
differences. On this line the across-seed spread of episode return is ~38.7
while the paired difference between two policies is ~11.2 -- instance
difficulty outweighs the policy effect roughly 3.5 to 1, so two independently
sampled means would mostly measure which job sets each policy happened to draw.

    python scripts/run_env.py
    python scripts/run_env.py --episodes 20
    python scripts/run_env.py --baseline blend --metric total_weighted_tardiness
"""

from __future__ import annotations

import argparse
import warnings

from dtmo.agents.policies import RandomPolicy, classical_policies
from dtmo.agents.train import make_eval_env
from dtmo.evaluation.paired import LOWER_IS_BETTER, benchmark, compare, leaderboard
from dtmo.utils.config import load_config

METRICS = ("return", "total_weighted_tardiness", "on_time_rate", "makespan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="factory YAML to load")
    parser.add_argument(
        "--episodes", type=int, default=12, help="seeds per policy (paired)"
    )
    parser.add_argument(
        "--interval", type=float, default=8.0, help="hours between decisions"
    )
    parser.add_argument(
        "--baseline", default="spt", help="policy every other is compared against"
    )
    parser.add_argument(
        "--metric", default="return", choices=METRICS, help="metric to compare on"
    )
    return parser


def main() -> None:
    warnings.simplefilter("ignore")
    args = build_parser().parse_args()
    config = load_config(args.config)
    env = make_eval_env(config, decision_interval=args.interval)

    # Held-out seeds, disjoint from the 3000/4000 ranges used to search for and
    # validate the tuned weight vector.
    seeds = list(range(2000, 2000 + args.episodes))

    policies = classical_policies() + [RandomPolicy(seed=0)]
    names = {policy.name for policy in policies}
    if args.baseline not in names:
        raise SystemExit(
            f"unknown baseline {args.baseline!r}; choose from {sorted(names)}"
        )

    print(f"line     : {config.n_jobs} jobs, {len(config.stations)} stations")
    print(f"episodes : {args.episodes} paired seeds ({seeds[0]}..{seeds[-1]})")
    print(f"decisions: every {args.interval:g}h, up to {env.max_steps} per episode")
    print()

    results = benchmark(policies, env, seeds)
    print(leaderboard(results, metric=args.metric))

    baseline = results[args.baseline]
    direction = "lower is better" if args.metric in LOWER_IS_BETTER else "higher is better"
    print(f"\npaired against {args.baseline} on {args.metric} ({direction}):")
    header = f"  {'POLICY':<10} {'DIFF':>9} {'WIN':>5} {'p':>7}  VERDICT"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for name, result in results.items():
        if name == args.baseline:
            continue
        rows.append((name, compare(result, baseline, metric=args.metric)))

    for name, c in sorted(rows, key=lambda r: -r[1].improvement):
        if not c.is_significant:
            verdict = "tie (not significant)"
        elif c.is_better:
            verdict = "better"
        else:
            verdict = "worse"
        print(
            f"  {name:<10} {c.improvement:>+9.2f} {c.win_rate:>4.0%} "
            f"{c.p_value:>7.3f}  {verdict}"
        )

    print(
        "\nEvery row above is a *constant* weight vector. A learned policy that "
        "varies\nits weights with the state has to beat the best of them to earn "
        "its keep."
    )


if __name__ == "__main__":
    main()
