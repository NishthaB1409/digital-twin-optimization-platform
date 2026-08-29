"""Policy wrappers.

Classical rules and trained agents have to be interchangeable, or the
benchmark cannot score them the same way.
"""

import numpy as np
import pytest

from dtmo.agents.policies import (
    BEST_KNOWN_BLEND,
    ConstantPolicy,
    Policy,
    RandomPolicy,
    SB3Policy,
    classical_policies,
)
from dtmo.digital_twin.dispatch import CLASSICAL_RULES


class StubModel:
    """Stands in for a trained SB3 model, without the training cost."""

    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = []

    def predict(self, observation, deterministic=True):
        self.calls.append(deterministic)
        return self.action, None


class TestConstantPolicy:
    def test_returns_its_weights_whatever_the_observation(self):
        policy = ConstantPolicy("spt", CLASSICAL_RULES["spt"])
        a = policy.act(np.zeros(16, dtype=np.float32))
        b = policy.act(np.ones(16, dtype=np.float32))
        assert a == pytest.approx(b)
        assert a == pytest.approx([1.0, 0.0, 0.0, 0.0])

    def test_rejects_a_wrong_length_vector(self):
        with pytest.raises(ValueError, match="expected 4 weights"):
            ConstantPolicy("bad", [1.0, 0.0])

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_rejects_non_finite_weights(self, bad):
        with pytest.raises(ValueError, match="finite"):
            ConstantPolicy("bad", [bad, 0.0, 0.0, 0.0])

    def test_clips_into_the_action_space(self):
        policy = ConstantPolicy("hot", [5.0, -5.0, 0.5, -0.5])
        assert policy.weights == pytest.approx([1.0, -1.0, 0.5, -0.5])

    def test_reset_is_a_no_op(self):
        policy = ConstantPolicy("spt", CLASSICAL_RULES["spt"])
        before = policy.act(np.zeros(16, dtype=np.float32)).copy()
        policy.reset()
        assert policy.act(np.zeros(16, dtype=np.float32)) == pytest.approx(before)

    def test_satisfies_the_policy_protocol(self):
        assert isinstance(ConstantPolicy("spt", CLASSICAL_RULES["spt"]), Policy)


class TestRandomPolicy:
    def test_actions_stay_in_range(self):
        policy = RandomPolicy(seed=0)
        for _ in range(20):
            action = policy.act(np.zeros(16, dtype=np.float32))
            assert action.shape == (4,)
            assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_actions_actually_vary(self):
        policy = RandomPolicy(seed=0)
        first = policy.act(np.zeros(16, dtype=np.float32)).copy()
        second = policy.act(np.zeros(16, dtype=np.float32))
        assert first != pytest.approx(second)

    def test_reset_replays_the_same_stream(self):
        policy = RandomPolicy(seed=3)
        first = [policy.act(np.zeros(16, dtype=np.float32)).copy() for _ in range(3)]
        policy.reset()
        second = [policy.act(np.zeros(16, dtype=np.float32)).copy() for _ in range(3)]
        for a, b in zip(first, second):
            assert a == pytest.approx(b)


class TestSB3Policy:
    def test_forwards_the_model_action(self):
        model = StubModel([0.1, 0.2, 0.3, 0.4])
        policy = SB3Policy(model, name="ppo")
        assert policy.act(np.zeros(16, dtype=np.float32)) == pytest.approx(
            [0.1, 0.2, 0.3, 0.4]
        )

    def test_evaluates_deterministically_by_default(self):
        model = StubModel([0.0, 0.0, 0.0, 0.0])
        SB3Policy(model).act(np.zeros(16, dtype=np.float32))
        assert model.calls == [True]

    def test_stochastic_mode_is_opt_in(self):
        model = StubModel([0.0, 0.0, 0.0, 0.0])
        SB3Policy(model, deterministic=False).act(np.zeros(16, dtype=np.float32))
        assert model.calls == [False]

    def test_flattens_a_batched_action(self):
        model = StubModel([[0.1, 0.2, 0.3, 0.4]])
        assert SB3Policy(model).act(np.zeros(16, dtype=np.float32)).shape == (4,)


class TestClassicalPolicies:
    def test_covers_every_classical_rule(self):
        names = {policy.name for policy in classical_policies()}
        assert set(CLASSICAL_RULES).issubset(names)

    def test_includes_the_tuned_blend_by_default(self):
        assert "blend" in {policy.name for policy in classical_policies()}

    def test_blend_can_be_excluded(self):
        names = {p.name for p in classical_policies(include_blend=False)}
        assert "blend" not in names

    def test_blend_is_a_valid_action(self):
        assert len(BEST_KNOWN_BLEND) == 4
        assert all(-1.0 <= w <= 1.0 for w in BEST_KNOWN_BLEND)

    def test_blend_leans_on_processing_time_and_slack(self):
        # The validated vector has the ATC shape: mostly SPT, real slack term.
        # If a future search replaces it with something structurally different,
        # that is worth noticing rather than silently accepting.
        proc, slack, remaining, wait = BEST_KNOWN_BLEND
        assert proc > 0.5
        assert slack > 0.0
