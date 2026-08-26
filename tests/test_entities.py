"""Job/family arithmetic -- the arithmetic every KPI is built on."""

import pytest

from dtmo.digital_twin.entities import FamilySpec, Job, Operation, StationSpec


def make_family(**overrides):
    defaults = dict(
        name="Test",
        route=(Operation("A", 2.0), Operation("B", 3.0), Operation("C", 5.0)),
        weight=2.0,
        mix=1.0,
        due_factor=2.0,
    )
    defaults.update(overrides)
    return FamilySpec(**defaults)


def make_job(**overrides):
    family = overrides.pop("family", make_family())
    defaults = dict(
        job_id=0,
        family=family,
        release_time=0.0,
        due_date=20.0,
        proc_times=(2.0, 3.0, 5.0),
    )
    defaults.update(overrides)
    return Job(**defaults)


class TestValidation:
    @pytest.mark.parametrize("capacity", [0, -1])
    def test_station_capacity_must_be_positive(self, capacity):
        with pytest.raises(ValueError, match="capacity"):
            StationSpec("A", capacity)

    def test_operation_time_must_be_positive(self):
        with pytest.raises(ValueError, match="mean_time"):
            Operation("A", 0.0)

    def test_family_route_must_be_non_empty(self):
        with pytest.raises(ValueError, match="route"):
            make_family(route=())

    def test_family_weight_must_be_positive(self):
        with pytest.raises(ValueError, match="weight"):
            make_family(weight=0.0)


class TestRouteProgress:
    def test_planned_work_sums_the_route(self):
        assert make_family().planned_work == pytest.approx(10.0)

    def test_remaining_work_shrinks_as_the_job_advances(self):
        job = make_job()
        assert job.remaining_work == pytest.approx(10.0)
        job.op_index = 1
        assert job.remaining_work == pytest.approx(8.0)
        job.op_index = 2
        assert job.remaining_work == pytest.approx(5.0)

    def test_is_done_only_after_the_last_operation(self):
        job = make_job()
        assert not job.is_done
        job.op_index = 2
        assert not job.is_done
        job.op_index = 3
        assert job.is_done

    def test_current_station_follows_the_route(self):
        job = make_job()
        assert [
            (job.__setattr__("op_index", i), job.current_station)[1]
            for i in range(3)
        ] == ["A", "B", "C"]

    def test_completion_fraction(self):
        job = make_job()
        assert job.completion_fraction == pytest.approx(0.0)
        job.op_index = 3
        assert job.completion_fraction == pytest.approx(1.0)


class TestDispatchFeatures:
    def test_slack_discounts_remaining_work_and_elapsed_time(self):
        # due 20, 10h of work left, clock at 4 -> 6h of genuine slack
        assert make_job().slack(4.0) == pytest.approx(6.0)

    def test_slack_goes_negative_when_the_date_is_unreachable(self):
        assert make_job().slack(15.0) == pytest.approx(-5.0)

    def test_waiting_time_measures_from_queue_entry(self):
        job = make_job()
        job.queue_entry_time = 3.0
        assert job.waiting_time(8.0) == pytest.approx(5.0)


class TestCompletionKpis:
    def test_kpis_raise_before_completion(self):
        job = make_job()
        for prop in ("flow_time", "tardiness", "is_on_time"):
            with pytest.raises(ValueError, match="not completed"):
                getattr(job, prop)

    def test_late_job_is_tardy_and_weighted(self):
        job = make_job(release_time=2.0)
        job.completion_time = 26.0
        assert job.flow_time == pytest.approx(24.0)
        assert job.tardiness == pytest.approx(6.0)          # 26 - 20
        assert job.weighted_tardiness == pytest.approx(12.0)  # weight 2.0
        assert not job.is_on_time

    def test_early_job_has_zero_tardiness(self):
        job = make_job()
        job.completion_time = 12.0
        assert job.tardiness == 0.0
        assert job.weighted_tardiness == 0.0
        assert job.is_on_time

    def test_exactly_on_the_due_date_counts_as_on_time(self):
        job = make_job()
        job.completion_time = 20.0
        assert job.tardiness == 0.0
        assert job.is_on_time
