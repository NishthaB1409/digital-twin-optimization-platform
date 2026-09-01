"""Lower bounds on what any schedule could achieve.

The property that matters is validity: a lower bound that a real policy can
beat is not a bound, it is a bug. Several tests here therefore run actual
policies and assert the bound never exceeds what they achieved.

That check has already earned its keep. An earlier version charged work at the
end of its time slot rather than the start, overstating lateness by up to one
slot per job, and produced "bounds" that rose as the relaxation was coarsened.
"""

import numpy as np
import pytest

from dtmo.agents.policies import classical_policies
from dtmo.digital_twin.entities import FamilySpec, Job, Operation, StationSpec
from dtmo.digital_twin.factory import FactoryModel
from dtmo.env import FactorySchedulingEnv
from dtmo.evaluation.paired import benchmark
from dtmo.optimization.bounds import (
    compute_bounds,
    bounds_for_seed,
    earliest_completion,
    route_makespan_bound,
    station_makespan_bound,
    station_work,
    unavoidable_tardiness,
    unavoidable_weighted_tardiness,
)
from dtmo.optimization.lp import (
    best_weighted_tardiness_bound,
    lp_bound,
    lp_bound_for_seed,
    relax_station,
)

SEEDS = [1000, 1001, 1002, 1003]


def make_job(job_id=0, release=0.0, due=100.0, times=(2.0, 3.0), weight=1.0):
    family = FamilySpec(
        name="F",
        route=tuple(Operation(s, t) for s, t in zip(("A", "B"), times)),
        weight=weight,
        mix=1.0,
        due_factor=2.0,
    )
    return Job(
        job_id=job_id,
        family=family,
        release_time=release,
        due_date=due,
        proc_times=tuple(times),
    )


@pytest.fixture(scope="module")
def small_config():
    from dtmo.utils.config import load_config

    return load_config().with_overrides(n_jobs=30)


@pytest.fixture(scope="module")
def instance(small_config):
    model = FactoryModel(small_config, seed=1000)
    model.reset()
    return model.jobs, small_config


class TestJobArithmetic:
    def test_earliest_completion_is_release_plus_own_work(self):
        assert earliest_completion(make_job(release=10.0)) == pytest.approx(15.0)

    def test_a_comfortable_job_has_no_unavoidable_tardiness(self):
        assert unavoidable_tardiness(make_job(release=0.0, due=100.0)) == 0.0

    def test_an_impossible_job_is_late_before_it_starts(self):
        # released at 10, needs 5h of work, due at 12 -> 3h late whatever happens
        assert unavoidable_tardiness(
            make_job(release=10.0, due=12.0)
        ) == pytest.approx(3.0)

    def test_weighted_sum_applies_the_family_weight(self):
        jobs = [make_job(release=10.0, due=12.0, weight=4.0)]
        assert unavoidable_weighted_tardiness(jobs) == pytest.approx(12.0)

    def test_station_work_totals_every_visit(self):
        totals = station_work([make_job(), make_job(job_id=1)])
        assert totals == pytest.approx({"A": 4.0, "B": 6.0})


class TestMakespanBound:
    def test_route_bound_is_the_longest_single_job(self):
        jobs = [make_job(release=0.0), make_job(job_id=1, release=50.0)]
        assert route_makespan_bound(jobs) == pytest.approx(55.0)

    def test_capacity_halves_the_station_bound(self, instance):
        jobs, config = instance
        station = config.stations[0]
        one = station_makespan_bound(jobs, StationSpec(station.name, 1))
        two = station_makespan_bound(jobs, StationSpec(station.name, 2))
        assert two < one

    def test_station_with_no_visitors_contributes_nothing(self, instance):
        jobs, _ = instance
        assert station_makespan_bound(jobs, StationSpec("Nowhere", 1)) == 0.0

    def test_bounds_name_their_binding_source(self, instance):
        jobs, config = instance
        assert compute_bounds(jobs, config.stations).makespan_source

    def test_empty_instance_is_an_error(self, small_config):
        with pytest.raises(ValueError, match="empty instance"):
            compute_bounds([], small_config.stations)


class TestCombinatorialValidity:
    """No real schedule may beat the bound."""

    @pytest.fixture(scope="class")
    def achieved(self, small_config):
        env = FactorySchedulingEnv(config=small_config, randomise_seed=False)
        return benchmark(classical_policies(), env, SEEDS)

    def test_no_policy_beats_the_makespan_bound(self, small_config, achieved):
        bounds = np.array(
            [bounds_for_seed(small_config, s).makespan for s in SEEDS]
        )
        for name, result in achieved.items():
            assert np.all(result.metric("makespan") >= bounds - 1e-6), name

    def test_no_policy_beats_the_tardiness_bound(self, small_config, achieved):
        bounds = np.array(
            [bounds_for_seed(small_config, s).weighted_tardiness for s in SEEDS]
        )
        for name, result in achieved.items():
            achieved_cost = result.metric("total_weighted_tardiness")
            assert np.all(achieved_cost >= bounds - 1e-6), name

    def test_gap_is_zero_for_a_provably_optimal_cost(self, small_config):
        bounds = bounds_for_seed(small_config, 1000)
        assert bounds.gap(bounds.weighted_tardiness) == pytest.approx(0.0)

    def test_gap_is_a_fraction(self, small_config):
        bounds = bounds_for_seed(small_config, 1000)
        assert 0.0 <= bounds.gap(5000.0) <= 1.0

    def test_gap_of_zero_cost_is_zero(self, small_config):
        assert bounds_for_seed(small_config, 1000).gap(0.0) == 0.0


class TestLPRelaxation:
    def test_it_solves(self, instance):
        jobs, config = instance
        result = lp_bound(jobs, config.stations, slot_hours=4.0)
        assert result.weighted_tardiness >= 0
        assert all(r.solved for r in result.stations)

    def test_it_names_the_binding_station(self, instance):
        jobs, config = instance
        result = lp_bound(jobs, config.stations, slot_hours=4.0)
        assert result.binding_station in {s.name for s in config.stations}

    def test_it_takes_the_strongest_station_not_the_sum(self, instance):
        """Summing would count the same jobs several times and overshoot."""
        jobs, config = instance
        result = lp_bound(jobs, config.stations, slot_hours=4.0)
        best = max(r.bound for r in result.stations)
        assert result.weighted_tardiness == pytest.approx(best)

    def test_refining_the_slots_tightens_the_bound(self, instance):
        """A coarser relaxation must be weaker, never stronger.

        This is what caught the slot-end charging bug: it made coarse slots
        report a *higher* bound, which is impossible for a valid relaxation.
        """
        jobs, config = instance
        coarse = lp_bound(jobs, config.stations, slot_hours=8.0)
        fine = lp_bound(jobs, config.stations, slot_hours=2.0)
        assert fine.weighted_tardiness >= coarse.weighted_tardiness - 1e-6

    def test_it_dominates_on_a_congested_line(self):
        """Keeping station capacity is the whole point of paying for an LP.

        Only on a congested instance, though -- see
        :class:`TestCombinedBound` for why neither relaxation always wins.
        """
        from dtmo.utils.config import load_config

        config = load_config()  # full 120-job load
        model = FactoryModel(config, seed=1000)
        model.reset()
        lp = lp_bound(model.jobs, config.stations, slot_hours=2.0)
        simple = compute_bounds(model.jobs, config.stations)
        assert lp.weighted_tardiness > simple.weighted_tardiness


class TestCombinedBound:
    """Neither relaxation dominates, so the reported bound takes the max.

    The LP keeps one station's capacity but allows preemption and charges
    conservatively at slot boundaries; the combinatorial bound keeps each job's
    route but drops contention entirely. On a lightly loaded line the LP can
    schedule everything on time and returns zero while the combinatorial bound
    still catches individually impossible jobs -- so taking the larger of two
    valid bounds is both valid and strictly better than either alone.
    """

    def test_it_is_at_least_as_strong_as_either(self, instance):
        jobs, config = instance
        value, _ = best_weighted_tardiness_bound(jobs, config.stations, 2.0)
        assert value >= compute_bounds(jobs, config.stations).weighted_tardiness
        assert value >= lp_bound(jobs, config.stations, 2.0).weighted_tardiness

    def test_it_reports_which_relaxation_won(self, instance):
        jobs, config = instance
        _, source = best_weighted_tardiness_bound(jobs, config.stations, 2.0)
        assert source in {"LP relaxation", "unavoidable tardiness"}

    def test_lightly_loaded_lines_fall_back_to_the_combinatorial_bound(self, instance):
        """With no contention the LP has nothing to say."""
        jobs, config = instance  # 30 jobs: the line is not congested
        _, source = best_weighted_tardiness_bound(jobs, config.stations, 2.0)
        assert source == "unavoidable tardiness"

    def test_congested_lines_are_bounded_by_the_lp(self):
        from dtmo.utils.config import load_config

        config = load_config()
        model = FactoryModel(config, seed=1000)
        model.reset()
        _, source = best_weighted_tardiness_bound(model.jobs, config.stations, 2.0)
        assert source == "LP relaxation"

    def test_it_is_never_beaten_by_a_real_policy(self, small_config):
        env = FactorySchedulingEnv(config=small_config, randomise_seed=False)
        achieved = benchmark(classical_policies(), env, SEEDS)
        bounds = []
        for seed in SEEDS:
            model = FactoryModel(small_config, seed=seed)
            model.reset()
            value, _ = best_weighted_tardiness_bound(
                model.jobs, small_config.stations, 2.0
            )
            bounds.append(value)
        bounds = np.array(bounds)
        for name, result in achieved.items():
            cost = result.metric("total_weighted_tardiness")
            assert np.all(cost >= bounds - 1e-6), name

    def test_a_station_nobody_visits_bounds_at_zero(self, instance):
        jobs, _ = instance
        result = relax_station(jobs, StationSpec("Nowhere", 1), slot_hours=4.0)
        assert result.solved and result.bound == 0.0

    def test_empty_instance_is_an_error(self, small_config):
        with pytest.raises(ValueError, match="empty instance"):
            lp_bound([], small_config.stations)


class TestLPValidity:
    """The LP bound must also be unbeatable by any real policy."""

    def test_no_policy_beats_the_lp_bound(self, small_config):
        env = FactorySchedulingEnv(config=small_config, randomise_seed=False)
        achieved = benchmark(classical_policies(), env, SEEDS)
        bounds = np.array(
            [
                lp_bound_for_seed(small_config, s, slot_hours=2.0).weighted_tardiness
                for s in SEEDS
            ]
        )
        for name, result in achieved.items():
            cost = result.metric("total_weighted_tardiness")
            assert np.all(cost >= bounds - 1e-6), (
                f"{name} achieved {cost} below the bound {bounds}"
            )
