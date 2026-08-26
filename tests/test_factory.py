"""End-to-end simulation invariants.

The physics checks here (capacity, route order, no job in two places at once)
are reconstructed from each job's operation log rather than read off the
counters the simulation maintains -- a counter that is wrong would otherwise
happily agree with itself.
"""

import pytest

from dtmo.digital_twin import CLASSICAL_RULES, FactoryModel
from dtmo.utils.config import load_config


@pytest.fixture(scope="module")
def run():
    model = FactoryModel(load_config().with_overrides(n_jobs=40, seed=11))
    kpis = model.run()
    return model, kpis


def max_concurrent(intervals):
    """Peak overlap among (start, finish) intervals, by sweep line."""
    events = []
    for start, finish in intervals:
        if finish > start:            # zero-length ops cannot occupy a machine
            events.append((start, 1))
            events.append((finish, -1))
    # Releases sort before starts at equal timestamps, so a machine handing off
    # at exactly time t is not double-counted.
    events.sort(key=lambda e: (e[0], e[1]))
    peak = current = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


class TestCompletion:
    def test_every_released_job_finishes(self, run):
        model, kpis = run
        assert kpis.jobs_completed == kpis.jobs_released == len(model.jobs)

    def test_every_job_has_a_completion_time(self, run):
        model, _ = run
        assert all(job.completion_time is not None for job in model.completed)

    def test_every_job_ran_all_of_its_operations(self, run):
        model, _ = run
        for job in model.completed:
            assert len(job.op_log) == job.family.n_operations


class TestPhysics:
    def test_station_capacity_is_never_exceeded(self, run):
        model, _ = run
        per_station = {name: [] for name in model.stations}
        for job in model.completed:
            for station, start, finish in job.op_log:
                per_station[station].append((start, finish))

        for name, intervals in per_station.items():
            capacity = model.stations[name].capacity
            peak = max_concurrent(intervals)
            assert peak <= capacity, (
                f"{name}: {peak} concurrent operations exceeds capacity {capacity}"
            )

    def test_jobs_visit_their_route_in_order(self, run):
        model, _ = run
        for job in model.completed:
            visited = [station for station, _, _ in job.op_log]
            expected = [op.station for op in job.family.route]
            assert visited == expected, f"job {job.job_id} took a wrong route"

    def test_a_job_is_never_in_two_places_at_once(self, run):
        model, _ = run
        for job in model.completed:
            for (_, _, prev_finish), (_, start, _) in zip(job.op_log, job.op_log[1:]):
                assert start >= prev_finish - 1e-9, (
                    f"job {job.job_id} started its next operation before "
                    f"finishing the previous one"
                )

    def test_no_job_starts_before_it_is_released(self, run):
        model, _ = run
        for job in model.completed:
            first_start = job.op_log[0][1]
            assert first_start >= job.release_time - 1e-9

    def test_operation_durations_match_the_sampled_times(self, run):
        model, _ = run
        for job in model.completed:
            for (_, start, finish), planned in zip(job.op_log, job.proc_times):
                assert finish - start == pytest.approx(planned, abs=1e-9)


class TestKpiConsistency:
    def test_utilisation_stays_within_bounds(self, run):
        _, kpis = run
        for name, util in kpis.station_utilisation.items():
            assert 0.0 <= util <= 1.0 + 1e-9, f"{name} utilisation {util}"

    def test_on_time_rate_is_a_fraction(self, run):
        _, kpis = run
        assert 0.0 <= kpis.on_time_rate <= 1.0

    def test_throughput_matches_completions_over_makespan(self, run):
        _, kpis = run
        assert kpis.throughput == pytest.approx(kpis.jobs_completed / kpis.makespan)

    def test_makespan_is_the_last_completion(self, run):
        model, kpis = run
        assert kpis.makespan == pytest.approx(
            max(job.completion_time for job in model.completed)
        )

    def test_mean_flow_time_does_not_exceed_the_max(self, run):
        _, kpis = run
        assert kpis.mean_flow_time <= kpis.max_flow_time + 1e-9

    def test_zero_tardiness_implies_everyone_on_time(self, run):
        _, kpis = run
        if kpis.total_weighted_tardiness == 0.0:
            assert kpis.on_time_rate == pytest.approx(1.0)

    def test_summary_renders(self, run):
        _, kpis = run
        text = kpis.summary()
        assert "FACTORY KPIs" in text
        assert "Machining" in text


class TestReproducibility:
    def test_same_seed_gives_identical_kpis(self, small_config):
        first = FactoryModel(small_config).run()
        second = FactoryModel(small_config).run()
        assert first.as_dict() == second.as_dict()

    def test_rerunning_one_model_is_idempotent(self, small_config):
        model = FactoryModel(small_config)
        assert model.run().as_dict() == model.run().as_dict()

    def test_different_seeds_give_different_runs(self, small_config):
        a = FactoryModel(small_config.with_overrides(seed=1)).run()
        b = FactoryModel(small_config.with_overrides(seed=2)).run()
        assert a.makespan != pytest.approx(b.makespan)

    def test_zero_cv_makes_processing_times_deterministic(self, small_config):
        model = FactoryModel(small_config.with_overrides(processing_cv=0.0))
        model.run()
        for job in model.completed:
            planned = [op.mean_time for op in job.family.route]
            assert list(job.proc_times) == pytest.approx(planned)


class TestDispatchMatters:
    """If the weights do not move the KPIs, there is nothing for RL to learn."""

    def test_weights_change_the_outcome(self, config):
        cfg = config.with_overrides(n_jobs=60, seed=3)
        spt = FactoryModel(cfg, weights=CLASSICAL_RULES["spt"]).run()
        lpt = FactoryModel(cfg, weights=CLASSICAL_RULES["lpt"]).run()
        assert spt.total_weighted_tardiness != pytest.approx(
            lpt.total_weighted_tardiness
        )

    def test_spt_beats_lpt_on_flow_time(self, config):
        # The one ordering result that is theory rather than tuning: SPT
        # minimises mean flow time. If this fails the rule is wired backwards.
        cfg = config.with_overrides(n_jobs=60, seed=3)
        spt = FactoryModel(cfg, weights=CLASSICAL_RULES["spt"]).run()
        lpt = FactoryModel(cfg, weights=CLASSICAL_RULES["lpt"]).run()
        assert spt.mean_flow_time < lpt.mean_flow_time

    def test_queues_are_deep_enough_for_dispatch_to_matter(self, config):
        model = FactoryModel(config)
        model.run()
        decisions = sum(s.dispatch_decisions for s in model.stations.values())
        contested = sum(s.contested_decisions for s in model.stations.values())
        assert contested / decisions > 0.20, (
            "fewer than 20% of dispatches had a choice to make -- the line is "
            "too lightly loaded for the dispatch rule to have any leverage"
        )

    def test_set_weights_takes_effect_on_the_next_run(self, small_config):
        model = FactoryModel(small_config, weights=CLASSICAL_RULES["spt"])
        before = model.run().mean_flow_time
        model.set_weights(CLASSICAL_RULES["lpt"])
        assert model.run().mean_flow_time != pytest.approx(before)
