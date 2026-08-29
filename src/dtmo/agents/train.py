"""PPO training against the factory environment.

The agent's job is narrow and worth stating precisely: read the floor every
``decision_interval`` hours and emit four dispatch weights. Every baseline it
competes with is a *constant* vector, so the only way it can win is by varying
its weights with the state -- leaning on slack when the bottleneck backs up,
and on processing time when the floor is clear.

Training uses randomised job sets so the agent learns scheduling rather than
one instance. Evaluation uses a fixed seed list, paired against the baseline,
because instance difficulty on this line outweighs the policy effect ~3.5x.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..digital_twin.dispatch import CLASSICAL_RULES
from ..env.factory_env import FactorySchedulingEnv, RewardConfig
from ..evaluation.paired import PairedComparison, PolicyResult, compare, evaluate
from ..utils.config import FactoryConfig, load_config
from .policies import BEST_KNOWN_BLEND, ConstantPolicy, SB3Policy

#: Seeds held out for evaluation. Fixed so every checkpoint, and every future
#: benchmark, is scored on exactly the same job sets.
DEFAULT_EVAL_SEEDS: tuple[int, ...] = tuple(range(1000, 1016))


@dataclass(frozen=True)
class TrainingConfig:
    """PPO hyperparameters and the training/eval protocol."""

    total_timesteps: int = 200_000
    n_envs: int = 4
    seed: int = 0

    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 10
    # An episode is only ~50 decisions and the makespan penalty lands on the
    # last one, so the discount has to be shallow enough to carry it back.
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    # The action space is bounded and the good region is interior, so an
    # entropy bonus mostly keeps the policy from committing. Near zero.
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (64, 64)
    #: SB3 defaults to std=1.0, which on a [-1, 1] action space makes the
    #: initial policy near-uniform noise over the corners. Start tighter.
    log_std_init: float = -1.0

    #: Reward scale varies with load; normalising stabilises the value loss.
    #: Observations are already bounded to [-1, 1], so they are left alone --
    #: which also lets `model.predict` take raw observations at eval time.
    normalise_reward: bool = True

    eval_freq: int = 20_000
    eval_seeds: tuple[int, ...] = DEFAULT_EVAL_SEEDS
    baseline: str = "spt"

    def __post_init__(self) -> None:
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be >= 1")
        if self.n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        rollout = self.n_steps * self.n_envs
        if self.batch_size > rollout:
            raise ValueError(
                f"batch_size ({self.batch_size}) exceeds the rollout buffer "
                f"({self.n_steps} x {self.n_envs} = {rollout})"
            )


def baseline_by_name(name: str) -> ConstantPolicy:
    """Resolve a baseline name to a constant-weight policy.

    ``"blend"`` is the tuned vector validated on held-out seeds (see
    :data:`~dtmo.agents.policies.BEST_KNOWN_BLEND`); everything else is a
    textbook rule.
    """
    if name == "blend":
        return ConstantPolicy("blend", BEST_KNOWN_BLEND)
    if name not in CLASSICAL_RULES:
        raise KeyError(
            f"unknown baseline {name!r}; choose 'blend' or one of "
            f"{sorted(CLASSICAL_RULES)}"
        )
    return ConstantPolicy(name, CLASSICAL_RULES[name])


def make_training_env(
    factory_config: FactoryConfig,
    training_config: TrainingConfig,
    reward: RewardConfig | None = None,
    decision_interval: float = 8.0,
) -> VecNormalize | DummyVecEnv:
    """Vectorised training envs, each drawing fresh job sets."""

    def factory(rank: int):
        def _init():
            env = FactorySchedulingEnv(
                config=factory_config,
                reward=reward,
                decision_interval=decision_interval,
                randomise_seed=True,
            )
            env = Monitor(env)
            # Seed each worker's episode-seed stream differently, once, so the
            # run is reproducible without every worker seeing the same jobs.
            env.reset(seed=training_config.seed * 1000 + rank)
            return env

        return _init

    venv = DummyVecEnv([factory(i) for i in range(training_config.n_envs)])
    if training_config.normalise_reward:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return venv


def make_eval_env(
    factory_config: FactoryConfig,
    reward: RewardConfig | None = None,
    decision_interval: float = 8.0,
) -> FactorySchedulingEnv:
    """A raw, unnormalised env -- evaluation reports true rewards and KPIs."""
    return FactorySchedulingEnv(
        config=factory_config,
        reward=reward,
        decision_interval=decision_interval,
        randomise_seed=False,
    )


class PairedEvalCallback(BaseCallback):
    """Periodically score the policy against a fixed baseline, seed-paired.

    Mean episode return alone is not a usable training signal here: it moves
    with whichever job sets the workers happened to draw. The paired
    improvement against a constant-weight baseline on held-out seeds is stable
    enough to actually read.
    """

    def __init__(
        self,
        eval_env: FactorySchedulingEnv,
        baseline: PolicyResult,
        seeds: Sequence[int],
        eval_freq: int,
        save_path: Path | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.baseline = baseline
        self.seeds = list(seeds)
        self.eval_freq = eval_freq
        self.save_path = Path(save_path) if save_path else None
        self.history: list[dict[str, Any]] = []
        self.best_improvement = -np.inf
        self._next_eval = eval_freq

    def _run_eval(self) -> PairedComparison:
        policy = SB3Policy(self.model, name="ppo")
        result = evaluate(policy, self.eval_env, self.seeds)
        comparison = compare(result, self.baseline, metric="return")
        self.history.append(
            {
                "timesteps": int(self.num_timesteps),
                "mean_return": result.mean_return,
                "improvement": comparison.improvement,
                "win_rate": comparison.win_rate,
                "p_value": comparison.p_value,
                "weighted_tardiness": result.mean("total_weighted_tardiness"),
                "on_time_rate": result.mean("on_time_rate"),
            }
        )
        if comparison.improvement > self.best_improvement:
            self.best_improvement = comparison.improvement
            if self.save_path is not None:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(self.save_path)
        return comparison

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_freq
            comparison = self._run_eval()
            if self.verbose:
                marker = "*" if comparison.improvement >= self.best_improvement else " "
                print(
                    f"  {self.num_timesteps:>7,} steps  "
                    f"return {comparison.candidate_mean:>8.2f}  "
                    f"vs {self.baseline.name} {comparison.improvement:>+7.2f}  "
                    f"wins {comparison.win_rate:>5.0%}  "
                    f"p={comparison.p_value:.3f} {marker}"
                )
        return True


@dataclass
class TrainingRun:
    """Everything a finished training run produced."""

    model: PPO
    baseline: PolicyResult
    final: PairedComparison
    history: list[dict[str, Any]]
    best_model_path: Path | None
    config: TrainingConfig

    def summary(self) -> str:
        lines = [
            f"trained {self.config.total_timesteps:,} timesteps "
            f"on {self.config.n_envs} envs",
            self.final.summary(),
        ]
        if self.best_model_path:
            lines.append(f"best checkpoint: {self.best_model_path}")
        return "\n".join(lines)


def train_ppo(
    factory_config: FactoryConfig | None = None,
    training_config: TrainingConfig | None = None,
    reward: RewardConfig | None = None,
    decision_interval: float = 8.0,
    save_dir: str | Path | None = None,
    verbose: int = 1,
) -> TrainingRun:
    """Train a PPO policy and score it against a constant-weight baseline."""
    factory_config = (factory_config or load_config()).validate()
    training_config = training_config or TrainingConfig()

    train_env = make_training_env(
        factory_config, training_config, reward, decision_interval
    )
    eval_env = make_eval_env(factory_config, reward, decision_interval)

    baseline_policy = baseline_by_name(training_config.baseline)
    baseline = evaluate(baseline_policy, eval_env, training_config.eval_seeds)
    if verbose:
        print(
            f"baseline {baseline.name}: return {baseline.mean_return:.2f} "
            f"+/- {baseline.std_return:.2f} over {len(baseline.seeds)} seeds"
        )

    save_dir = Path(save_dir) if save_dir else None
    best_path = (save_dir / "ppo_best") if save_dir else None

    callback = PairedEvalCallback(
        eval_env=eval_env,
        baseline=baseline,
        seeds=training_config.eval_seeds,
        eval_freq=training_config.eval_freq,
        save_path=best_path,
        verbose=verbose,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=training_config.learning_rate,
        n_steps=training_config.n_steps,
        batch_size=training_config.batch_size,
        n_epochs=training_config.n_epochs,
        gamma=training_config.gamma,
        gae_lambda=training_config.gae_lambda,
        clip_range=training_config.clip_range,
        ent_coef=training_config.ent_coef,
        vf_coef=training_config.vf_coef,
        max_grad_norm=training_config.max_grad_norm,
        policy_kwargs={
            "net_arch": list(training_config.net_arch),
            "log_std_init": training_config.log_std_init,
        },
        seed=training_config.seed,
        verbose=0,
    )
    model.learn(total_timesteps=training_config.total_timesteps, callback=callback)

    final_result = evaluate(
        SB3Policy(model, name="ppo"), eval_env, training_config.eval_seeds
    )
    final = compare(final_result, baseline, metric="return")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(save_dir / "ppo_final")

    train_env.close()
    return TrainingRun(
        model=model,
        baseline=baseline,
        final=final,
        history=callback.history,
        best_model_path=best_path,
        config=training_config,
    )
