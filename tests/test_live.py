"""The live control loop.

The floor runs on a background thread and HTTP handlers read it, so the tests
that matter are the ones about the boundary: a snapshot is always consistent,
a misbehaving policy cannot stall or corrupt the run, and stopping actually
stops.

Safety is the point of the fallback tests. A plant will not deploy a scheduler
that can emit anything at all; one that occasionally does something merely
adequate is fine, one that occasionally does something wild is not.
"""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dtmo.agents.policies import BEST_KNOWN_BLEND, ConstantPolicy
from dtmo.serving.app import app
from dtmo.serving.live import FALLBACK_WEIGHTS, LiveConfig, LiveFactory


class ExplodingPolicy:
    """Fails every time it is asked. Stands in for a broken model."""

    name = "exploding"

    def act(self, observation):
        raise RuntimeError("model is broken")


class WildPolicy:
    """Returns values far outside the allowed range."""

    name = "wild"

    def act(self, observation):
        return np.array([50.0, -50.0, 50.0, -50.0])


class NanPolicy:
    name = "nan"

    def act(self, observation):
        return np.array([np.nan, 0.0, 0.0, 0.0])


@pytest.fixture
def small(config):
    return config.with_overrides(n_jobs=25)


def run_until_done(factory, timeout=25.0):
    factory.start()
    deadline = time.time() + timeout
    while factory.running and time.time() < deadline:
        time.sleep(0.05)
    factory.stop()
    return factory.snapshot()


class TestLifecycle:
    def test_it_runs_the_floor_to_completion(self, small):
        factory = LiveFactory(
            small,
            ConstantPolicy("blend", BEST_KNOWN_BLEND),
            LiveConfig(time_scale=400.0, seed=1000),
        )
        snapshot = run_until_done(factory)
        assert snapshot["finished"]
        assert snapshot["jobs_completed"] == small.n_jobs

    def test_it_is_not_running_before_start(self, small):
        factory = LiveFactory(small, ConstantPolicy("spt", [1, 0, 0, 0]))
        assert not factory.running

    def test_stop_actually_stops_it(self, small):
        factory = LiveFactory(
            small,
            ConstantPolicy("spt", [1, 0, 0, 0]),
            LiveConfig(time_scale=6.0, seed=1000),
        )
        factory.start()
        time.sleep(0.4)
        factory.stop()
        assert not factory.running
        assert not factory.snapshot()["finished"]

    def test_starting_twice_is_refused(self, small):
        factory = LiveFactory(
            small,
            ConstantPolicy("spt", [1, 0, 0, 0]),
            LiveConfig(time_scale=6.0, seed=1000),
        )
        factory.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                factory.start()
        finally:
            factory.stop()

    def test_the_clock_advances(self, small):
        factory = LiveFactory(
            small,
            ConstantPolicy("spt", [1, 0, 0, 0]),
            LiveConfig(time_scale=60.0, seed=1000),
        )
        factory.start()
        time.sleep(0.3)
        first = factory.snapshot()["clock_hours"]
        time.sleep(0.4)
        second = factory.snapshot()["clock_hours"]
        factory.stop()
        assert second > first


class TestSafety:
    def test_a_broken_policy_falls_back_instead_of_stalling(self, small):
        factory = LiveFactory(
            small, ExplodingPolicy(), LiveConfig(time_scale=400.0, seed=1000)
        )
        snapshot = run_until_done(factory)
        assert snapshot["fallbacks"] > 0
        assert snapshot["jobs_completed"] == small.n_jobs, "the floor must keep running"

    def test_the_fallback_is_the_known_good_rule(self, small):
        factory = LiveFactory(
            small, ExplodingPolicy(), LiveConfig(time_scale=400.0, seed=1000)
        )
        snapshot = run_until_done(factory)
        applied = np.array(list(snapshot["weights"].values()))
        expected = np.array(FALLBACK_WEIGHTS, dtype=float)
        expected = expected / np.linalg.norm(expected)
        assert applied == pytest.approx(expected, abs=1e-6)

    def test_a_wild_policy_is_clamped(self, small):
        factory = LiveFactory(
            small, WildPolicy(), LiveConfig(time_scale=400.0, seed=1000)
        )
        snapshot = run_until_done(factory)
        applied = np.array(list(snapshot["weights"].values()))
        assert np.all(np.abs(applied) <= 1.0 + 1e-9)
        assert np.linalg.norm(applied) == pytest.approx(1.0, abs=1e-6)

    def test_a_non_finite_action_falls_back(self, small):
        factory = LiveFactory(
            small, NanPolicy(), LiveConfig(time_scale=400.0, seed=1000)
        )
        snapshot = run_until_done(factory)
        assert snapshot["fallbacks"] > 0
        assert np.all(np.isfinite(list(snapshot["weights"].values())))

    def test_fallbacks_are_recorded_as_events(self, small):
        factory = LiveFactory(
            small, ExplodingPolicy(), LiveConfig(time_scale=400.0, seed=1000)
        )
        run_until_done(factory)
        kinds = {event.kind for event in factory._events}
        assert "fallback" in kinds


class TestSnapshot:
    @pytest.fixture
    def snapshot(self, small):
        factory = LiveFactory(
            small,
            ConstantPolicy("blend", BEST_KNOWN_BLEND),
            LiveConfig(time_scale=400.0, seed=1000),
        )
        return run_until_done(factory)

    def test_it_reports_every_station(self, snapshot, config):
        names = {station["name"] for station in snapshot["stations"]}
        assert names == set(config.station_names)

    def test_queues_never_go_negative(self, snapshot):
        for station in snapshot["stations"]:
            assert station["queue_length"] >= 0
            assert 0 <= station["busy_machines"] <= station["capacity"]

    def test_it_names_the_four_weights(self, snapshot):
        assert set(snapshot["weights"]) == {
            "processing_time",
            "slack",
            "remaining_work",
            "waiting_time",
        }

    def test_the_policy_was_consulted(self, snapshot):
        assert snapshot["decisions"] > 0

    def test_on_time_rate_is_a_fraction(self, snapshot):
        assert 0.0 <= snapshot["on_time_rate"] <= 1.0

    def test_the_event_log_is_bounded(self, small):
        """A long run must not grow the log without limit."""
        factory = LiveFactory(
            small,
            ConstantPolicy("spt", [1, 0, 0, 0]),
            LiveConfig(time_scale=400.0, seed=1000),
        )
        run_until_done(factory)
        assert len(factory._events) <= 40


class TestConfig:
    @pytest.mark.parametrize("field", ["time_scale", "decision_interval"])
    def test_non_positive_values_are_rejected(self, field):
        with pytest.raises(ValueError, match=field):
            LiveConfig(**{field: 0.0})


class TestRoutes:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_state_before_start_is_a_clear_error(self, client):
        import dtmo.serving.live_routes as routes

        routes._factory = None
        response = client.get("/live/state")
        assert response.status_code == 409
        assert "start" in response.json()["detail"]

    def test_start_then_state_then_stop(self, client):
        started = client.post(
            "/live/start", json={"policy": "spt", "seed": 1000, "time_scale": 400}
        )
        assert started.status_code == 200
        assert started.json()["started"]

        snapshot = client.get("/live/state").json()
        assert snapshot["jobs_total"] > 0
        assert "stations" in snapshot

        assert client.post("/live/stop").json()["stopped"]

    def test_an_unknown_policy_is_rejected(self, client):
        assert (
            client.post("/live/start", json={"policy": "nope"}).status_code == 404
        )

    def test_an_impossible_speed_is_rejected(self, client):
        assert (
            client.post(
                "/live/start", json={"policy": "spt", "time_scale": 0}
            ).status_code
            == 422
        )

    def test_the_dashboard_is_served(self, client):
        response = client.get("/live")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "EventSource" in response.text

    def test_the_dashboard_offers_a_deliberately_bad_policy(self, client):
        """Being able to watch a bad rule fail is the point of the demo."""
        assert "mwkr" in client.get("/live").text
