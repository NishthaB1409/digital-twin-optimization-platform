"""The composite rule must reproduce the classical heuristics exactly.

These are the tests that matter most: if the rule does not collapse to SPT at
weights (1,0,0,0), then the Day 4 benchmark comparison is meaningless and the
RL agent is searching a space whose landmarks are wrong.
"""

import numpy as np
import pytest

from dtmo.digital_twin.dispatch import (
    CLASSICAL_RULES,
    N_FEATURES,
    CompositeDispatchRule,
    min_max_normalise,
)
from dtmo.digital_twin.entities import FamilySpec, Job, Operation


def make_job(job_id, proc, due, queue_entry=0.0, weight=1.0):
    family = FamilySpec(
        name=f"F{job_id}",
        route=(Operation("A", proc),),
        weight=weight,
        mix=1.0,
        due_factor=2.0,
    )
    job = Job(
        job_id=job_id,
        family=family,
        release_time=0.0,
        due_date=due,
        proc_times=(proc,),
    )
    job.queue_entry_time = queue_entry
    return job


@pytest.fixture
def queue():
    # id: (processing, due, queued_at)
    return [
        make_job(0, proc=5.0, due=100.0, queue_entry=0.0),   # long, loose, waited longest
        make_job(1, proc=1.0, due=50.0, queue_entry=5.0),    # shortest
        make_job(2, proc=3.0, due=10.0, queue_entry=9.0),    # tightest due, waited least
    ]


class TestNormalisation:
    def test_scales_each_column_into_unit_range(self):
        out = min_max_normalise(np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]))
        assert out.min(axis=0) == pytest.approx([0.0, 0.0])
        assert out.max(axis=0) == pytest.approx([1.0, 1.0])

    def test_constant_column_collapses_to_zero_not_nan(self):
        out = min_max_normalise(np.array([[7.0, 1.0], [7.0, 2.0]]))
        assert np.all(np.isfinite(out))
        assert out[:, 0] == pytest.approx([0.0, 0.0])

    def test_normalisation_is_invariant_to_time_units(self):
        raw = np.array([[2.0, 4.0], [6.0, 8.0]])
        assert min_max_normalise(raw) == pytest.approx(min_max_normalise(raw * 60.0))


class TestClassicalEquivalence:
    def test_spt_picks_the_shortest_job(self, queue):
        rule = CompositeDispatchRule.from_name("spt")
        assert rule.select(queue, now=10.0).job_id == 1

    def test_lpt_picks_the_longest_job(self, queue):
        rule = CompositeDispatchRule.from_name("lpt")
        assert rule.select(queue, now=10.0).job_id == 0

    def test_min_slack_picks_the_most_urgent_job(self, queue):
        # at t=10: slacks are 85, 39, -3 -> job 2 is already past saving
        rule = CompositeDispatchRule.from_name("min_slack")
        assert rule.select(queue, now=10.0).job_id == 2

    def test_fifo_picks_the_longest_waiting_job(self, queue):
        rule = CompositeDispatchRule.from_name("fifo")
        assert rule.select(queue, now=10.0).job_id == 0

    def test_lifo_picks_the_most_recent_arrival(self, queue):
        rule = CompositeDispatchRule.from_name("lifo")
        assert rule.select(queue, now=10.0).job_id == 2

    def test_lwkr_picks_least_work_remaining(self, queue):
        rule = CompositeDispatchRule.from_name("lwkr")
        assert rule.select(queue, now=10.0).job_id == 1

    def test_every_classical_rule_is_a_valid_weight_vector(self):
        for name, weights in CLASSICAL_RULES.items():
            assert len(weights) == N_FEATURES, name
            assert all(-1.0 <= w <= 1.0 for w in weights), name


class TestWeights:
    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="expected 4 weights"):
            CompositeDispatchRule([1.0, 0.0])

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_rejects_non_finite(self, bad):
        with pytest.raises(ValueError, match="finite"):
            CompositeDispatchRule([bad, 0.0, 0.0, 0.0])

    def test_weights_property_returns_a_defensive_copy(self):
        rule = CompositeDispatchRule.from_name("spt")
        rule.weights[0] = 999.0
        assert rule.weights[0] == pytest.approx(1.0)

    def test_weights_are_mutable_between_runs(self):
        rule = CompositeDispatchRule.from_name("spt")
        rule.weights = CLASSICAL_RULES["lpt"]
        assert rule.weights == pytest.approx([-1.0, 0.0, 0.0, 0.0])

    def test_unknown_rule_name_lists_the_valid_ones(self):
        with pytest.raises(KeyError, match="spt"):
            CompositeDispatchRule.from_name("nope")


class TestSelection:
    def test_empty_queue_is_an_error(self):
        with pytest.raises(ValueError, match="empty queue"):
            CompositeDispatchRule.from_name("spt").select([], now=0.0)

    def test_single_job_queue_returns_that_job(self, queue):
        only = queue[:1]
        assert CompositeDispatchRule.from_name("spt").select(only, 0.0) is only[0]

    def test_identical_jobs_break_ties_toward_the_earlier_arrival(self):
        twins = [make_job(0, 4.0, 40.0), make_job(1, 4.0, 40.0)]
        rule = CompositeDispatchRule.from_name("spt")
        assert rule.select(twins, now=1.0).job_id == 0

    def test_zero_weights_degenerate_to_queue_order(self, queue):
        rule = CompositeDispatchRule([0.0, 0.0, 0.0, 0.0])
        assert rule.select(queue, now=10.0).job_id == 0

    def test_blended_weights_can_differ_from_every_pure_rule(self, queue):
        # A weighted blend is allowed to disagree with its components -- that
        # headroom is the entire reason an RL agent is worth training.
        blend = CompositeDispatchRule([0.4, 0.9, 0.0, -0.3])
        scores = blend.scores(queue, now=10.0)
        assert len(scores) == len(queue)
        assert np.all(np.isfinite(scores))
