"""The scheduling API.

The test that matters most is
``test_the_api_encodes_exactly_like_the_environment``. A policy served an
observation scaled differently from the one it trained on is not degraded, it
is reading noise -- and every response would still look perfectly well-formed.
Training/serving skew is the classic way a model silently stops working, so the
encoder is shared and this test pins that it stays shared.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dtmo.digital_twin.dispatch import CLASSICAL_RULES
from dtmo.env.factory_env import OBS_DIM, FactorySchedulingEnv, encode_observation
from dtmo.serving.app import app, describe_rule


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def floor(config):
    capacities = {s.name: s.capacity for s in config.stations}
    return {
        "stations": [
            {
                "name": name,
                "queue_length": 3,
                "busy_machines": capacities[name],
            }
            for name in config.station_names
        ],
        "clock_hours": 180.0,
        "jobs_completed": 40,
        "jobs_in_progress": 22,
        "total_jobs": 120,
        "mean_slack_hours": -6.5,
    }


class TestOps:
    def test_health_reports_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["policy_loaded"]

    def test_info_describes_the_configured_line(self, client, config):
        body = client.get("/info").json()
        assert body["stations"] == list(config.station_names)
        assert body["observation_features"] == OBS_DIM
        assert len(body["families"]) == len(config.families)

    def test_policies_lists_every_classical_rule(self, client):
        body = client.get("/policies").json()
        assert set(CLASSICAL_RULES).issubset(body["classical"])
        assert "blend" in body["tuned"]

    def test_openapi_schema_is_served(self, client):
        assert client.get("/openapi.json").status_code == 200


class TestWeights:
    def test_it_returns_four_weights(self, client, floor):
        body = client.post("/weights", json=floor).json()
        assert set(body["weights"]) == {
            "processing_time",
            "slack",
            "remaining_work",
            "waiting_time",
        }

    def test_it_returns_a_unit_direction(self, client, floor):
        """The rule is scale-invariant, so only direction is meaningful."""
        body = client.post("/weights", json=floor).json()
        vector = np.array(list(body["weights"].values()))
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)

    def test_it_echoes_the_observation_it_used(self, client, floor):
        body = client.post("/weights", json=floor).json()
        assert len(body["observation"]) == OBS_DIM
        assert all(-1.0 <= v <= 1.0 for v in body["observation"])

    def test_a_named_rule_comes_back_unchanged(self, client, floor):
        body = client.post("/weights?policy=spt", json=floor).json()
        assert body["policy"] == "spt"
        assert body["weights"]["processing_time"] == pytest.approx(1.0)

    def test_it_explains_the_rule_in_words(self, client, floor):
        body = client.post("/weights?policy=spt", json=floor).json()
        assert "short jobs" in body["rule"]

    def test_unknown_policy_is_a_404_that_lists_the_options(self, client, floor):
        response = client.post("/weights?policy=nope", json=floor)
        assert response.status_code == 404
        assert "spt" in response.json()["detail"]


class TestValidation:
    def test_wrong_stations_are_rejected(self, client, floor):
        floor["stations"][0]["name"] = "Not A Station"
        response = client.post("/weights", json=floor)
        assert response.status_code == 422
        assert "expected" in response.json()["detail"]

    def test_missing_stations_are_rejected(self, client, floor):
        floor["stations"] = floor["stations"][:3]
        assert client.post("/weights", json=floor).status_code == 422

    def test_more_busy_machines_than_capacity_is_rejected(self, client, floor):
        floor["stations"][0]["busy_machines"] = 99
        response = client.post("/weights", json=floor)
        assert response.status_code == 422
        assert "capacity" in response.json()["detail"]

    def test_negative_queue_is_rejected(self, client, floor):
        floor["stations"][0]["queue_length"] = -1
        assert client.post("/weights", json=floor).status_code == 422

    def test_duplicate_station_names_are_rejected(self, client, floor):
        floor["stations"][1]["name"] = floor["stations"][0]["name"]
        assert client.post("/weights", json=floor).status_code == 422

    def test_unknown_fields_are_rejected(self, client, floor):
        floor["surprise"] = 1
        assert client.post("/weights", json=floor).status_code == 422

    def test_zero_total_jobs_is_rejected(self, client, floor):
        floor["total_jobs"] = 0
        assert client.post("/weights", json=floor).status_code == 422


class TestNoTrainServeSkew:
    def test_the_api_encodes_exactly_like_the_environment(self, config):
        """Serving must build the same observation the env builds.

        Both call `encode_observation`; this asserts the shared path actually
        produces the environment's own vector for the same floor state.
        """
        env = FactorySchedulingEnv(config=config, randomise_seed=False)
        observation, _ = env.reset(seed=1000)
        for _ in range(12):
            observation, *_ = env.step(
                np.array(CLASSICAL_RULES["spt"], dtype=np.float32)
            )

        stations = [env.model.stations[n] for n in config.station_names]
        wip = env.model.wip
        mean_slack = (
            float(np.mean([j.slack(env.model.now) for j in wip])) if wip else 0.0
        )
        rebuilt = encode_observation(
            queue_lengths=[s.queue_length for s in stations],
            busy_fractions=[s.busy_machines / s.capacity for s in stations],
            clock_hours=env.model.now,
            horizon_hours=env.horizon,
            jobs_completed=len(env.model.completed),
            jobs_in_progress=len(wip),
            total_jobs=config.n_jobs,
            mean_slack_hours=mean_slack,
            slack_scale=float(
                np.mean([f.planned_work for f in config.families])
            ),
        )
        assert rebuilt == pytest.approx(observation, abs=1e-6)

    def test_encoder_rejects_a_wrong_station_count(self):
        with pytest.raises(ValueError, match="expected 16 features"):
            encode_observation(
                queue_lengths=[1, 2],
                busy_fractions=[0.5, 0.5],
                clock_hours=10.0,
                horizon_hours=100.0,
                jobs_completed=1,
                jobs_in_progress=1,
                total_jobs=10,
                mean_slack_hours=0.0,
                slack_scale=25.0,
            )

    def test_encoder_survives_a_zero_horizon(self):
        observation = encode_observation(
            queue_lengths=[0] * 6,
            busy_fractions=[0.0] * 6,
            clock_hours=0.0,
            horizon_hours=0.0,
            jobs_completed=0,
            jobs_in_progress=0,
            total_jobs=1,
            mean_slack_hours=0.0,
            slack_scale=0.0,
        )
        assert np.all(np.isfinite(observation))


class TestSimulate:
    def test_it_runs_a_named_policy(self, client):
        body = client.post(
            "/simulate", json={"policy": "spt", "seed": 1000, "n_jobs": 30}
        ).json()
        assert body["jobs_completed"] == 30
        assert body["makespan_hours"] > 0
        assert 0.0 <= body["on_time_rate"] <= 1.0

    def test_it_runs_explicit_weights(self, client):
        body = client.post(
            "/simulate",
            json={
                "weights": {
                    "processing_time": 1.0,
                    "slack": 0.0,
                    "remaining_work": 0.0,
                    "waiting_time": 0.0,
                },
                "seed": 1000,
                "n_jobs": 30,
            },
        ).json()
        assert body["policy"] == "custom"

    def test_it_reports_a_lower_bound_below_what_was_achieved(self, client):
        body = client.post(
            "/simulate", json={"policy": "spt", "seed": 1000, "n_jobs": 30}
        ).json()
        bound = body["weighted_tardiness_lower_bound"]
        if bound is not None:
            assert bound <= body["total_weighted_tardiness"] + 1e-6

    def test_it_reports_every_station(self, client, config):
        body = client.post(
            "/simulate", json={"policy": "spt", "seed": 1000, "n_jobs": 30}
        ).json()
        assert {s["name"] for s in body["stations"]} == set(config.station_names)

    def test_neither_weights_nor_policy_is_rejected(self, client):
        response = client.post("/simulate", json={"seed": 1000})
        assert response.status_code == 422

    def test_the_same_seed_gives_the_same_answer(self, client):
        payload = {"policy": "spt", "seed": 1000, "n_jobs": 30}
        first = client.post("/simulate", json=payload).json()
        second = client.post("/simulate", json=payload).json()
        assert first == second


class TestRuleDescription:
    def test_it_reads_spt_as_favouring_short_jobs(self):
        assert "short jobs" in describe_rule(np.array([1.0, 0.0, 0.0, 0.0]))

    def test_it_reads_lpt_as_favouring_long_jobs(self):
        assert "long jobs" in describe_rule(np.array([-1.0, 0.0, 0.0, 0.0]))

    def test_it_reads_min_slack_as_urgency(self):
        assert "urgent" in describe_rule(np.array([0.0, 1.0, 0.0, 0.0]))

    def test_a_flat_vector_has_no_preference(self):
        assert describe_rule(np.zeros(4)) == "no strong preference"
