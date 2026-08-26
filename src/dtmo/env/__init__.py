"""Gymnasium environment wrapping the digital twin."""

from gymnasium.envs.registration import register

from .factory_env import OBS_DIM, FactorySchedulingEnv, RewardConfig

register(
    id="dtmo/FactoryScheduling-v0",
    entry_point="dtmo.env.factory_env:FactorySchedulingEnv",
)

__all__ = ["OBS_DIM", "FactorySchedulingEnv", "RewardConfig"]
