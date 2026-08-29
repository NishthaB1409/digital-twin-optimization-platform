"""Seed-paired policy evaluation.

Comparing two policies on *different* random seeds is invalid here, and not by
a small margin. Measured on this line: the across-seed standard deviation of
episode return is ~38.7, while the standard deviation of the seed-paired
difference between two policies is ~11.2. Instance difficulty outweighs the
policy effect roughly 3.5 to 1, so an unpaired comparison mostly measures which
job sets each policy happened to draw.

Everything here therefore runs every policy over the *same* seed list, and
:func:`compare` refuses to work on mismatched seeds rather than quietly
returning a meaningless number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

from ..digital_twin.kpis import KPIs

if TYPE_CHECKING:  # avoids a package-level cycle: agents.train imports this
    from ..agents.policies import Policy
    from ..env.factory_env import FactorySchedulingEnv

#: KPIs where a smaller number is a better factory.
LOWER_IS_BETTER = frozenset(
    {
        "makespan",
        "mean_flow_time",
        "max_flow_time",
        "mean_tardiness",
        "max_tardiness",
        "total_weighted_tardiness",
    }
)


@dataclass(frozen=True)
class EpisodeResult:
    """One policy, one seed."""

    seed: int
    total_reward: float
    steps: int
    terminated: bool
    kpis: KPIs | None

    def metric(self, name: str) -> float:
        if name == "return":
            return self.total_reward
        if self.kpis is None:
            raise ValueError(
                f"seed {self.seed}: episode did not finish, so {name!r} is undefined"
            )
        return float(getattr(self.kpis, name))


@dataclass(frozen=True)
class PolicyResult:
    """One policy across the whole seed list."""

    name: str
    episodes: tuple[EpisodeResult, ...]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(episode.seed for episode in self.episodes)

    @property
    def returns(self) -> np.ndarray:
        return self.metric("return")

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns))

    @property
    def std_return(self) -> float:
        return float(np.std(self.returns))

    @property
    def all_finished(self) -> bool:
        return all(episode.terminated for episode in self.episodes)

    def metric(self, name: str) -> np.ndarray:
        """Per-seed values of one metric, in seed order."""
        return np.array([episode.metric(name) for episode in self.episodes], dtype=float)

    def mean(self, name: str) -> float:
        return float(np.mean(self.metric(name)))


@dataclass(frozen=True)
class PairedComparison:
    """Candidate against baseline, on identical seeds."""

    candidate: str
    baseline: str
    metric: str
    n_seeds: int
    candidate_mean: float
    baseline_mean: float
    mean_difference: float      # candidate - baseline, in raw metric units
    std_difference: float
    improvement: float          # signed so positive always means "better"
    improvement_pct: float
    win_rate: float
    p_value: float

    @property
    def is_better(self) -> bool:
        return self.improvement > 0

    @property
    def is_significant(self) -> bool:
        """Two-sided paired t-test at the conventional 5% level."""
        return self.p_value < 0.05

    def summary(self) -> str:
        verdict = "better" if self.is_better else "worse"
        stars = "significant" if self.is_significant else "not significant"
        return (
            f"{self.candidate} vs {self.baseline} on {self.metric}: "
            f"{self.candidate_mean:.2f} vs {self.baseline_mean:.2f} "
            f"({self.improvement_pct:+.1f}% {verdict}), "
            f"wins {self.win_rate:.0%} of {self.n_seeds} seeds, "
            f"p={self.p_value:.4f} ({stars})"
        )


def evaluate(
    policy: "Policy",
    env: "FactorySchedulingEnv",
    seeds: Sequence[int],
    max_steps: int = 10_000,
) -> PolicyResult:
    """Roll ``policy`` through one episode per seed."""
    if not seeds:
        raise ValueError("need at least one seed to evaluate")

    episodes: list[EpisodeResult] = []
    for seed in seeds:
        observation, info = env.reset(seed=int(seed))
        if hasattr(policy, "reset"):
            policy.reset()

        total = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated) and steps < max_steps:
            action = policy.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1

        episodes.append(
            EpisodeResult(
                seed=int(seed),
                total_reward=total,
                steps=steps,
                terminated=bool(terminated),
                kpis=info.get("kpis"),
            )
        )
    return PolicyResult(name=policy.name, episodes=tuple(episodes))


def benchmark(
    policies: Iterable["Policy"],
    env: "FactorySchedulingEnv",
    seeds: Sequence[int],
) -> dict[str, PolicyResult]:
    """Evaluate several policies over one shared seed list."""
    return {policy.name: evaluate(policy, env, seeds) for policy in policies}


def compare(
    candidate: PolicyResult,
    baseline: PolicyResult,
    metric: str = "return",
) -> PairedComparison:
    """Paired comparison of two policies on the same seeds.

    Raises if the seed lists differ -- an unpaired comparison on this problem
    is dominated by instance difficulty and would be misleading, so it is made
    impossible rather than merely discouraged.
    """
    if candidate.seeds != baseline.seeds:
        raise ValueError(
            "paired comparison requires identical seeds, but "
            f"{candidate.name!r} ran {candidate.seeds} and "
            f"{baseline.name!r} ran {baseline.seeds}"
        )

    a = candidate.metric(metric)
    b = baseline.metric(metric)
    difference = a - b

    # Flip the sign for metrics where smaller is better, so `improvement`
    # always reads "positive means the candidate won".
    direction = -1.0 if metric in LOWER_IS_BETTER else 1.0
    improvement = direction * difference

    baseline_mean = float(np.mean(b))
    scale = abs(baseline_mean)
    improvement_pct = (
        100.0 * float(np.mean(improvement)) / scale if scale > 1e-12 else 0.0
    )

    if len(difference) < 2 or np.allclose(difference, difference[0]):
        # ttest_rel needs variation; a constant difference has no p-value.
        p_value = 0.0 if np.any(difference != 0) else 1.0
    else:
        p_value = float(stats.ttest_rel(a, b).pvalue)

    return PairedComparison(
        candidate=candidate.name,
        baseline=baseline.name,
        metric=metric,
        n_seeds=len(difference),
        candidate_mean=float(np.mean(a)),
        baseline_mean=baseline_mean,
        mean_difference=float(np.mean(difference)),
        std_difference=float(np.std(difference)),
        improvement=float(np.mean(improvement)),
        improvement_pct=improvement_pct,
        win_rate=float(np.mean(improvement > 0)),
        p_value=p_value,
    )


def leaderboard(
    results: Mapping[str, PolicyResult],
    metric: str = "return",
) -> str:
    """Rank policies best-first on one metric."""
    if not results:
        return "(no results)"

    lower_better = metric in LOWER_IS_BETTER
    rows = sorted(
        results.values(),
        key=lambda r: r.mean(metric),
        reverse=not lower_better,
    )
    width = max(len(name) for name in results)
    header = (
        f"{'POLICY'.ljust(width)}  {'RETURN':>9} {'+/-':>7} "
        f"{'ON-TIME':>8} {'WGT TARD':>10} {'MAKESPAN':>9}"
    )
    lines = [header, "-" * len(header)]
    for result in rows:
        finished = result.all_finished
        on_time = f"{result.mean('on_time_rate'):>7.1%}" if finished else f"{'--':>8}"
        tardiness = (
            f"{result.mean('total_weighted_tardiness'):>10.1f}" if finished else f"{'--':>10}"
        )
        makespan = f"{result.mean('makespan'):>9.1f}" if finished else f"{'--':>9}"
        lines.append(
            f"{result.name.ljust(width)}  {result.mean_return:>9.2f} "
            f"{result.std_return:>7.2f} {on_time} {tardiness} {makespan}"
        )
    return "\n".join(lines)
