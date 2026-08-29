"""Seed-paired evaluation.

The load-bearing test here is ``test_mismatched_seeds_are_refused``. Instance
difficulty on this line outweighs the policy effect roughly 3.5x, so an
unpaired comparison is not merely imprecise -- it mostly measures which job
sets each policy happened to draw. Making it raise is the point.
"""

import numpy as np
import pytest

from dtmo.agents.policies import ConstantPolicy, classical_policies
from dtmo.digital_twin.dispatch import CLASSICAL_RULES
from dtmo.env import FactorySchedulingEnv
from dtmo.evaluation.paired import (
    LOWER_IS_BETTER,
    benchmark,
    compare,
    evaluate,
    leaderboard,
)

SEEDS = [101, 102, 103, 104, 105, 106]


@pytest.fixture(scope="module")
def env():
    from dtmo.utils.config import load_config

    return FactorySchedulingEnv(
        config=load_config().with_overrides(n_jobs=30), randomise_seed=False
    )


@pytest.fixture(scope="module")
def spt(env):
    return evaluate(ConstantPolicy("spt", CLASSICAL_RULES["spt"]), env, SEEDS)


@pytest.fixture(scope="module")
def mwkr(env):
    return evaluate(ConstantPolicy("mwkr", CLASSICAL_RULES["mwkr"]), env, SEEDS)


class TestEvaluate:
    def test_runs_one_episode_per_seed(self, spt):
        assert len(spt.episodes) == len(SEEDS)
        assert spt.seeds == tuple(SEEDS)

    def test_every_episode_finishes(self, spt):
        assert spt.all_finished
        assert all(episode.kpis is not None for episode in spt.episodes)

    def test_returns_are_reproducible(self, env):
        policy = ConstantPolicy("spt", CLASSICAL_RULES["spt"])
        first = evaluate(policy, env, SEEDS)
        second = evaluate(policy, env, SEEDS)
        assert first.returns == pytest.approx(second.returns)

    def test_needs_at_least_one_seed(self, env):
        with pytest.raises(ValueError, match="at least one seed"):
            evaluate(ConstantPolicy("spt", CLASSICAL_RULES["spt"]), env, [])

    def test_metric_reads_kpi_fields(self, spt):
        tardiness = spt.metric("total_weighted_tardiness")
        assert tardiness.shape == (len(SEEDS),)
        assert np.all(tardiness >= 0)

    def test_metric_return_matches_episode_rewards(self, spt):
        assert spt.metric("return") == pytest.approx(
            [e.total_reward for e in spt.episodes]
        )


class TestPairing:
    def test_mismatched_seeds_are_refused(self, env, spt):
        other = evaluate(
            ConstantPolicy("spt", CLASSICAL_RULES["spt"]), env, [201, 202, 203]
        )
        with pytest.raises(ValueError, match="identical seeds"):
            compare(other, spt)

    def test_a_policy_against_itself_shows_no_difference(self, spt):
        result = compare(spt, spt)
        assert result.mean_difference == pytest.approx(0.0)
        assert result.improvement == pytest.approx(0.0)
        assert result.win_rate == 0.0

    def test_records_the_seed_count(self, spt, mwkr):
        assert compare(mwkr, spt).n_seeds == len(SEEDS)


class TestDirection:
    def test_higher_return_counts_as_better(self, spt, mwkr):
        # SPT dominates MWKR on this line by a wide margin.
        result = compare(spt, mwkr, metric="return")
        assert result.improvement > 0
        assert result.is_better
        assert result.win_rate > 0.5

    def test_lower_tardiness_counts_as_better(self, spt, mwkr):
        """The sign must flip for cost-like metrics, or 'better' inverts."""
        result = compare(spt, mwkr, metric="total_weighted_tardiness")
        assert result.candidate_mean < result.baseline_mean
        assert result.improvement > 0
        assert result.is_better

    def test_the_reverse_comparison_is_worse(self, spt, mwkr):
        result = compare(mwkr, spt, metric="total_weighted_tardiness")
        assert result.improvement < 0
        assert not result.is_better

    def test_cost_metrics_are_registered(self):
        assert "total_weighted_tardiness" in LOWER_IS_BETTER
        assert "makespan" in LOWER_IS_BETTER
        assert "return" not in LOWER_IS_BETTER


class TestStatistics:
    def test_a_large_gap_is_significant(self, spt, mwkr):
        assert compare(spt, mwkr).is_significant

    def test_p_value_is_a_probability(self, spt, mwkr):
        assert 0.0 <= compare(spt, mwkr).p_value <= 1.0

    def test_identical_results_are_not_significant(self, spt):
        assert compare(spt, spt).p_value == pytest.approx(1.0)

    def test_win_rate_is_a_fraction(self, spt, mwkr):
        assert 0.0 <= compare(spt, mwkr).win_rate <= 1.0

    def test_summary_states_the_verdict(self, spt, mwkr):
        text = compare(spt, mwkr).summary()
        assert "spt" in text and "mwkr" in text
        assert "wins" in text and "p=" in text


class TestBenchmark:
    def test_scores_every_policy_on_the_same_seeds(self, env):
        results = benchmark(classical_policies(), env, SEEDS)
        assert len(results) == len(classical_policies())
        assert all(r.seeds == tuple(SEEDS) for r in results.values())

    def test_results_are_pairable(self, env):
        results = benchmark(classical_policies(), env, SEEDS)
        comparison = compare(results["spt"], results["lpt"])
        assert comparison.n_seeds == len(SEEDS)

    def test_leaderboard_ranks_best_first(self, env):
        results = benchmark(classical_policies(), env, SEEDS)
        lines = leaderboard(results).splitlines()
        ranked = [line.split()[0] for line in lines[2:]]
        returns = [results[name].mean_return for name in ranked]
        assert returns == sorted(returns, reverse=True)

    def test_leaderboard_handles_no_results(self):
        assert "no results" in leaderboard({})
