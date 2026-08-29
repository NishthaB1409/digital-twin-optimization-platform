"""Day 3: train a PPO agent to tune the dispatch weights.

    python scripts/train_agent.py
    python scripts/train_agent.py --timesteps 500000 --baseline blend
    python scripts/train_agent.py --evaluate runs/ppo_day3/ppo_best

Training draws a fresh job set every episode so the agent learns scheduling
rather than one instance. Evaluation runs a fixed, held-out seed list and
reports the seed-paired difference against a constant-weight baseline -- on
this line, instance difficulty outweighs the policy effect ~3.5x, so unpaired
means are not interpretable.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from dtmo.agents.policies import SB3Policy, classical_policies
from dtmo.agents.train import (
    DEFAULT_EVAL_SEEDS,
    TrainingConfig,
    baseline_by_name,
    make_eval_env,
    train_ppo,
)
from dtmo.evaluation.paired import benchmark, compare, evaluate, leaderboard
from dtmo.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="factory YAML to load")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--envs", type=int, default=4, help="parallel rollout workers")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--baseline",
        default="spt",
        help="constant-weight policy to beat (a classical rule, or 'blend')",
    )
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--save-dir", default="runs/ppo_day3")
    parser.add_argument(
        "--evaluate",
        metavar="MODEL",
        default=None,
        help="skip training; benchmark a saved model against every baseline",
    )
    return parser


def print_weight_behaviour(policy, env, seed: int) -> None:
    """Show whether the policy actually varies its weights with the state.

    This is the whole thesis of the project. A learned policy that emits the
    same vector at every step has, in effect, rediscovered a classical rule --
    useful to know, and invisible in the KPIs alone.
    """
    observation, _ = env.reset(seed=seed)
    actions = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(observation)
        actions.append(np.asarray(action, dtype=float).copy())
        observation, _, terminated, truncated, _ = env.step(action)

    actions = np.array(actions)
    names = ("proc", "slack", "remain", "wait")
    print(f"\nweight behaviour over one episode (seed {seed}, {len(actions)} steps)")
    print(f"  {'FEATURE':<8} {'MEAN':>7} {'MIN':>7} {'MAX':>7} {'STD':>7}")
    for i, name in enumerate(names):
        column = actions[:, i]
        print(
            f"  {name:<8} {column.mean():>7.3f} {column.min():>7.3f} "
            f"{column.max():>7.3f} {column.std():>7.3f}"
        )
    spread = float(actions.std(axis=0).mean())
    if spread < 0.02:
        print("  -> essentially constant; the agent settled on a fixed rule")
    else:
        print(f"  -> varies with state (mean std {spread:.3f})")


def run_benchmark(model_path: str, config, seeds) -> None:
    env = make_eval_env(config)
    policy = SB3Policy.load(model_path, name="ppo")

    policies = classical_policies() + [policy]
    results = benchmark(policies, env, seeds)

    print(f"\nheld-out seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} episodes each)")
    print()
    print(leaderboard(results))

    print("\npaired comparisons against ppo:")
    for name in sorted(results):
        if name == "ppo":
            continue
        c = compare(results["ppo"], results[name], metric="return")
        flag = "significant" if c.is_significant else "n.s."
        print(
            f"  vs {name:<10} {c.improvement:>+8.2f}  "
            f"wins {c.win_rate:>4.0%}  p={c.p_value:>6.3f}  {flag}"
        )

    print_weight_behaviour(policy, env, seed=seeds[0])


def main() -> None:
    warnings.simplefilter("ignore")
    args = build_parser().parse_args()
    config = load_config(args.config)
    seeds = list(DEFAULT_EVAL_SEEDS)

    if args.evaluate:
        run_benchmark(args.evaluate, config, seeds)
        return

    training_config = TrainingConfig(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        seed=args.seed,
        eval_freq=args.eval_freq,
        baseline=args.baseline,
        eval_seeds=tuple(seeds),
    )

    print(f"line     : {config.n_jobs} jobs, {len(config.stations)} stations")
    print(f"training : {args.timesteps:,} timesteps on {args.envs} envs")
    print(f"baseline : {args.baseline} {baseline_by_name(args.baseline).weights}")
    print(f"eval     : seeds {seeds[0]}..{seeds[-1]}, paired\n")

    run = train_ppo(
        factory_config=config,
        training_config=training_config,
        save_dir=args.save_dir,
        verbose=1,
    )

    print()
    print(run.summary())

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "history.json").write_text(json.dumps(run.history, indent=2))
    print(f"history written to {save_dir / 'history.json'}")

    print_weight_behaviour(
        SB3Policy(run.model, name="ppo"), make_eval_env(config), seed=seeds[0]
    )


if __name__ == "__main__":
    main()
