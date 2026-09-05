"""The numpy policy export.

Two things must hold or the slim serving image is a liability rather than an
optimisation: the export has to behave identically to the model it came from,
and the serving path has to actually avoid importing torch. An export that
quietly diverges is worse than none, because every response would still look
perfectly well-formed.
"""

import subprocess
import sys

import numpy as np
import pytest

from dtmo.serving.export import ACTION_LOW, ACTION_HIGH, NumpyPolicy, extract

MODEL = "runs/ppo_shaped/ppo_best"


def make_params(n_obs=16, hidden=64, n_act=4, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "w0": rng.normal(0, 0.4, (hidden, n_obs)).astype(np.float32),
        "b0": rng.normal(0, 0.1, hidden).astype(np.float32),
        "w1": rng.normal(0, 0.4, (hidden, hidden)).astype(np.float32),
        "b1": rng.normal(0, 0.1, hidden).astype(np.float32),
        "wa": rng.normal(0, 0.4, (n_act, hidden)).astype(np.float32),
        "ba": rng.normal(0, 0.1, n_act).astype(np.float32),
    }


@pytest.fixture
def policy():
    return NumpyPolicy(make_params())


class TestForwardPass:
    def test_it_returns_one_action_per_output(self, policy):
        assert policy.act(np.zeros(16, dtype=np.float32)).shape == (4,)

    def test_actions_stay_inside_the_action_space(self, policy):
        rng = np.random.default_rng(1)
        for _ in range(50):
            action = policy.act(rng.uniform(-1, 1, 16).astype(np.float32))
            assert np.all(action >= ACTION_LOW) and np.all(action <= ACTION_HIGH)

    def test_it_is_deterministic(self, policy):
        observation = np.linspace(-1, 1, 16, dtype=np.float32)
        assert policy.act(observation) == pytest.approx(policy.act(observation))

    def test_different_observations_give_different_actions(self, policy):
        a = policy.act(np.full(16, -0.8, dtype=np.float32))
        b = policy.act(np.full(16, 0.8, dtype=np.float32))
        assert a != pytest.approx(b)

    def test_it_reports_its_parameter_count(self, policy):
        # 64x16 + 64 + 64x64 + 64 + 4x64 + 4
        assert policy.n_parameters == 5508

    def test_reset_is_a_no_op(self, policy):
        observation = np.zeros(16, dtype=np.float32)
        before = policy.act(observation).copy()
        policy.reset()
        assert policy.act(observation) == pytest.approx(before)


class TestRoundTrip:
    def test_saving_and_loading_preserves_behaviour(self, policy, tmp_path):
        path = policy.save(tmp_path / "p.npz")
        reloaded = NumpyPolicy.load(path)
        observation = np.linspace(-1, 1, 16, dtype=np.float32)
        assert reloaded.act(observation) == pytest.approx(policy.act(observation))

    def test_the_file_is_small(self, policy, tmp_path):
        """The whole point: a policy you can ship without a framework."""
        path = policy.save(tmp_path / "p.npz")
        assert path.stat().st_size < 100_000


class TestExtractionGuards:
    def test_an_unexpected_architecture_is_refused(self):
        class Wrong:
            class policy:
                squash_output = False

                @staticmethod
                def state_dict():
                    return {}

        with pytest.raises(ValueError, match="architecture"):
            extract(Wrong())

    def test_a_squashing_policy_is_refused(self):
        import torch

        keys = [
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.policy_net.0.bias",
            "mlp_extractor.policy_net.2.weight",
            "mlp_extractor.policy_net.2.bias",
            "action_net.weight",
            "action_net.bias",
        ]

        class Squashing:
            class policy:
                squash_output = True

                @staticmethod
                def state_dict():
                    return {k: torch.zeros(2, 2) for k in keys}

        with pytest.raises(ValueError, match="squash"):
            extract(Squashing())


@pytest.mark.skipif(
    not __import__("pathlib").Path(f"{MODEL}.zip").exists(),
    reason="no trained model checked in",
)
class TestMatchesTheTrainedModel:
    """The export must agree with the model it replaces."""

    def test_it_matches_torch_on_random_observations(self, tmp_path):
        from dtmo.serving.export import export_policy

        _, worst = export_policy(MODEL, tmp_path / "p.npz", n_checks=256)
        assert worst < 1e-5

    def test_it_produces_identical_episodes(self, tmp_path, config):
        from dtmo.agents.policies import SB3Policy
        from dtmo.agents.train import make_eval_env
        from dtmo.evaluation.paired import evaluate
        from dtmo.serving.export import export_policy

        path, _ = export_policy(MODEL, tmp_path / "p.npz", n_checks=8)
        env = make_eval_env(config)
        seeds = [1000, 1001, 1002]

        torch_result = evaluate(SB3Policy.load(MODEL, name="t"), env, seeds)
        numpy_result = evaluate(NumpyPolicy.load(path, name="n"), env, seeds)
        assert numpy_result.returns == pytest.approx(torch_result.returns)


class TestSlimServing:
    def test_the_serving_path_never_imports_torch(self):
        """Run it in a fresh interpreter -- this test's own process has torch.

        If serving pulls torch in, the slim image fails at startup rather than
        degrading, so this is worth catching here instead of in a build.
        """
        script = (
            "import os, sys; "
            "os.environ['DTMO_POLICY'] = 'runs/ppo_shaped/policy.npz'; "
            "from fastapi.testclient import TestClient; "
            "from dtmo.serving.app import app; "
            "c = TestClient(app); "
            "assert c.get('/health').status_code == 200; "
            "heavy = [m for m in ('torch','stable_baselines3','pandas','matplotlib') "
            "if m in sys.modules]; "
            "print('HEAVY:' + ','.join(heavy))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "HEAVY:\n" in result.stdout or result.stdout.strip().endswith("HEAVY:")
