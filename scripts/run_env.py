"""Day 2 check: roll policies through the Gymnasium env and score them.

Each classical rule is held constant for a whole episode, which turns it into a
fixed policy the env can score. That gives the baseline return PPO has to beat
on Day 3 -- expressed in the same reward units the agent will actually see, not
in raw KPIs.

    python scripts/run_env.py
    python scripts/run_env.py --episodes 10
    python scripts/run_env.py --interval 24
"""

from __future__ import annotations

import argparse

import numpy as np

from dtmo.digital_twin import CLASSICAL_RULES
from dtmo.env import FactorySchedulingEnv
from dtmo.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5, help="seeds per policy")
    parser.add_argument(
        "--interval", type=float, default=8.0, help="hours between decisions"
    )
    parser.add_argument("--config", default=None, help="factory YAML to load")
    return parser


def roll(env, policy, seed):
    """Run one episode. ``policy`` maps an observation to an action."""
    obs, info = env.reset(seed=seed)
    total = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        steps += 1
    return total, steps, info


def constant(weights):
    action = np.asarray(weights, dtype=np.float32)
    return lambda obs: action


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    env = FactorySchedulingEnv(
        config=config, decision_interval=args.interval, randomise_seed=False
    )
    seeds = list(range(1, args.episodes + 1))

    policies = {name: constant(w) for name, w in sorted(CLASSICAL_RULES.items())}
    # The blend that beat SPT by 12.4% in the Day 1 random search.
    policies["blend*"] = constant([0.24, 0.99, 0.90, -0.08])
    policies["random"] = lambda obs: env.action_space.sample()

    print(f"env    : {args.episodes} episodes/policy, {args.interval:g}h decisions")
    print(f"line   : {config.n_jobs} jobs, {len(config.stations)} stations")
    print(f"steps  : up to {env.max_steps} decisions per episode")

    header = (
        f"{'POLICY':<10} {'RETURN':>9} {'+/-':>7} {'ON-TIME':>8} "
        f"{'WGT TARD':>10} {'MAKESPAN':>9}"
    )
    print()
    print(header)
    print("-" * len(header))

    rows = []
    for name, policy in policies.items():
        returns, ontime, tard, mkspan = [], [], [], []
        for seed in seeds:
            total, _, info = roll(env, policy, seed)
            returns.append(total)
            kpis = info.get("kpis")
            if kpis is not None:
                ontime.append(kpis.on_time_rate)
                tard.append(kpis.total_weighted_tardiness)
                mkspan.append(kpis.makespan)
        rows.append((name, float(np.mean(returns)), float(np.std(returns)),
                     float(np.mean(ontime)), float(np.mean(tard)),
                     float(np.mean(mkspan))))

    for name, ret, sd, ot, wt, ms in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:<10} {ret:>9.2f} {sd:>7.2f} {ot:>7.1%} {wt:>10.1f} {ms:>9.1f}")
    print("-" * len(header))

    best = max(rows, key=lambda r: r[1])
    worst = min(rows, key=lambda r: r[1])
    print(f"best fixed policy : {best[0]} (return {best[1]:.2f})")
    print(f"worst             : {worst[0]} (return {worst[1]:.2f})")
    print(f"spread            : {best[1] - worst[1]:.2f} reward")
    print()
    print("Day 3 target: a PPO policy that varies its weights with the observed")
    print(f"state should beat {best[1]:.2f} -- any fixed vector, however good, cannot")
    print("react to a queue building at the bottleneck.")


if __name__ == "__main__":
    main()
