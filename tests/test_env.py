"""Gymnasium environment contract and physics.

The single most important test here is
``test_holding_weights_reproduces_the_batch_run``: if stepping the clock in
slices does not give the same factory as running it straight through, then
every KPI the agent optimises is an artefact of the stepping, not the schedule.
"""

import warnings

import numpy as np
import pytest

from dtmo.digital_twin import CLASSICAL_RULES, FactoryModel
from dtmo.env import OBS_DIM, FactorySchedulingEnv, RewardConfig

SPT = np.array(CLASSICAL_RULES["spt"], dtype=np.float32)


@pytest.fixture
def env(config):
    return FactorySchedulingEnv(config=config, randomise_seed=False)


def rollout(env, action, seed=0, max_steps=10_000):
    obs, info = env.reset(seed=seed)
    steps = 0
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated) and steps < max_steps:
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
    return obs, total, steps, terminated, truncated, info


class TestSpaces:
    def test_action_space_is_the_four_weights(self, env):
        assert env.action_space.shape == (4,)
        assert env.action_space.low.min() == -1.0
        assert env.action_space.high.max() == 1.0

    def test_observation_space_is_16_dimensional(self, env):
        assert env.observation_space.shape == (OBS_DIM,)

    def test_reset_returns_an_observation_inside_the_space(self, env):
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

    def test_observations_stay_inside_the_space_all_episode(self, env):
        obs, info = env.reset(seed=1)
        for _ in range(40):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs), obs
            if terminated or truncated:
                break

    def test_passes_the_gymnasium_env_checker(self, config):
        from gymnasium.utils.env_checker import check_env

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_env(
                FactorySchedulingEnv(config=config, randomise_seed=False),
                skip_render_check=True,
            )


class TestPhysicsMatchesBatch:
    def test_holding_weights_reproduces_the_batch_run(self, env, config):
        """Stepping in slices must not change the factory."""
        *_, info = rollout(env, SPT, seed=5)
        stepped = info["kpis"]
        batch = FactoryModel(config, weights=CLASSICAL_RULES["spt"], seed=5).run()

        assert stepped.jobs_completed == batch.jobs_completed
        assert stepped.makespan == pytest.approx(batch.makespan)
        assert stepped.mean_flow_time == pytest.approx(batch.mean_flow_time)
        assert stepped.on_time_rate == pytest.approx(batch.on_time_rate)
        assert stepped.total_weighted_tardiness == pytest.approx(
            batch.total_weighted_tardiness
        )

    def test_a_different_held_rule_also_matches_its_batch_run(self, env, config):
        lpt = np.array(CLASSICAL_RULES["lpt"], dtype=np.float32)
        *_, info = rollout(env, lpt, seed=5)
        batch = FactoryModel(config, weights=CLASSICAL_RULES["lpt"], seed=5).run()
        assert info["kpis"].makespan == pytest.approx(batch.makespan)

    def test_decision_interval_does_not_change_held_weight_physics(self, config):
        """The grid is a sampling choice, not part of the model."""
        results = []
        for interval in (4.0, 8.0, 24.0):
            env = FactorySchedulingEnv(
                config=config, decision_interval=interval, randomise_seed=False
            )
            *_, info = rollout(env, SPT, seed=5)
            results.append(info["kpis"].makespan)
        assert results[0] == pytest.approx(results[1])
        assert results[1] == pytest.approx(results[2])


class TestEpisode:
    def test_episode_terminates_when_every_job_is_done(self, env):
        _, _, _, terminated, truncated, info = rollout(env, SPT, seed=3)
        assert terminated and not truncated
        assert info["completed"] == env.config.n_jobs

    def test_terminated_episode_reports_kpis(self, env):
        *_, info = rollout(env, SPT, seed=3)
        assert "kpis" in info
        assert info["kpis"].jobs_completed == env.config.n_jobs

    def test_truncation_fires_when_the_step_budget_runs_out(self, config):
        env = FactorySchedulingEnv(config=config, randomise_seed=False, max_steps=3)
        _, _, steps, terminated, truncated, _ = rollout(env, SPT, seed=3)
        assert truncated and not terminated
        assert steps == 3

    def test_unfinished_episode_reports_no_kpis(self, config):
        env = FactorySchedulingEnv(config=config, randomise_seed=False, max_steps=3)
        *_, info = rollout(env, SPT, seed=3)
        assert "kpis" not in info

    def test_clock_advances_by_the_decision_interval(self, config):
        env = FactorySchedulingEnv(
            config=config, decision_interval=6.0, randomise_seed=False
        )
        env.reset(seed=0)
        _, _, _, _, info = env.step(SPT)
        assert info["time"] == pytest.approx(6.0)
        _, _, _, _, info = env.step(SPT)
        assert info["time"] == pytest.approx(12.0)


class TestSeeding:
    def test_same_seed_gives_the_same_episode(self, env):
        first = rollout(env, SPT, seed=11)[5]["kpis"]
        second = rollout(env, SPT, seed=11)[5]["kpis"]
        assert first.as_dict() == second.as_dict()

    def test_different_seeds_give_different_episodes(self, env):
        a = rollout(env, SPT, seed=11)[5]["kpis"]
        b = rollout(env, SPT, seed=12)[5]["kpis"]
        assert a.makespan != pytest.approx(b.makespan)

    def test_randomising_varies_the_job_set_between_episodes(self, config):
        """Training on one fixed instance teaches that instance, not scheduling."""
        env = FactorySchedulingEnv(config=config, randomise_seed=True)
        env.reset(seed=0)
        seeds = set()
        for _ in range(5):
            env.reset()
            seeds.add(env.model.seed)
        assert len(seeds) > 1

    def test_fixed_mode_repeats_the_config_seed(self, config):
        env = FactorySchedulingEnv(config=config, randomise_seed=False)
        env.reset()
        first = env.model.seed
        env.reset()
        assert env.model.seed == first == config.seed


class TestActions:
    def test_out_of_range_actions_are_clipped_not_rejected(self, env):
        env.reset(seed=0)
        env.step(np.array([5.0, -5.0, 5.0, -5.0], dtype=np.float32))
        assert env.model.dispatcher.weights == pytest.approx([1.0, -1.0, 1.0, -1.0])

    def test_wrong_shape_action_is_an_error(self, env):
        env.reset(seed=0)
        with pytest.raises(ValueError, match="shape"):
            env.step(np.array([1.0, 0.0], dtype=np.float32))

    def test_action_reaches_the_dispatch_rule(self, env):
        env.reset(seed=0)
        action = np.array([0.2, -0.4, 0.6, -0.8], dtype=np.float32)
        env.step(action)
        assert env.model.dispatcher.weights == pytest.approx(action, abs=1e-6)

    def test_weights_can_change_between_steps(self, env):
        env.reset(seed=0)
        env.step(SPT)
        assert env.model.dispatcher.weights == pytest.approx(SPT)
        lpt = np.array(CLASSICAL_RULES["lpt"], dtype=np.float32)
        env.step(lpt)
        assert env.model.dispatcher.weights == pytest.approx(lpt)


class TestReward:
    def test_reward_is_finite_throughout(self, env):
        env.reset(seed=2)
        for _ in range(30):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward)
            if terminated or truncated:
                break

    def test_policy_choice_changes_the_return(self, env):
        """If the return does not move with the weights, there is no signal."""
        spt_return = rollout(env, SPT, seed=7)[1]
        mwkr = np.array(CLASSICAL_RULES["mwkr"], dtype=np.float32)
        mwkr_return = rollout(env, mwkr, seed=7)[1]
        assert spt_return != pytest.approx(mwkr_return)

    def test_the_better_schedule_earns_the_higher_return(self, env):
        # SPT crushes MWKR on this line (1111 vs 5388 weighted tardiness in the
        # Day 1 benchmark). Reward that disagreed would be pointing uphill.
        spt_return = rollout(env, SPT, seed=7)[1]
        mwkr = np.array(CLASSICAL_RULES["mwkr"], dtype=np.float32)
        mwkr_return = rollout(env, mwkr, seed=7)[1]
        assert spt_return > mwkr_return

    def test_tardiness_penalty_can_be_switched_off(self, config):
        env = FactorySchedulingEnv(
            config=config,
            randomise_seed=False,
            reward=RewardConfig(tardiness=0.0, makespan=0.0),
        )
        assert rollout(env, SPT, seed=7)[1] > 0

    def test_negative_penalties_are_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            RewardConfig(tardiness=-1.0)


class TestConstruction:
    def test_decision_interval_must_be_positive(self, config):
        with pytest.raises(ValueError, match="decision_interval"):
            FactorySchedulingEnv(config=config, decision_interval=0.0)

    def test_loads_the_default_config_when_given_none(self):
        assert FactorySchedulingEnv().config.n_jobs > 0

    def test_registered_under_a_gymnasium_id(self):
        import gymnasium as gym

        import dtmo.env  # noqa: F401  -- registers the id

        env = gym.make("dtmo/FactoryScheduling-v0")
        assert env.action_space.shape == (4,)
        env.close()
