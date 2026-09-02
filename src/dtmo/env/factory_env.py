"""Gymnasium environment: an agent retunes the dispatch weights mid-run.

The agent does not schedule jobs directly. It steers the *rule* that schedules
them -- every ``decision_interval`` hours it reads the state of the floor and
writes a new 4-weight vector, which takes effect at the very next dispatch.

That framing is deliberate. Picking a job per station per decision would give a
combinatorial action space that changes shape as queues grow; picking four
continuous weights gives a fixed ``Box(-1, 1, (4,))`` that PPO/SAC/TD3 handle
natively, and the resulting policy stays interpretable -- you can read off
whether it learned to favour short jobs or urgent ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..digital_twin.dispatch import N_FEATURES
from ..digital_twin.factory import FactoryModel
from ..utils.config import FactoryConfig, load_config

#: 6 queue lengths + 6 busy fractions + clock, completion, WIP, mean slack.
OBS_DIM = 16

#: Queue lengths are squashed by tanh(q / QUEUE_SCALE) so a station with 40
#: waiting jobs does not saturate the input the way a raw count would.
QUEUE_SCALE = 5.0


def encode_observation(
    queue_lengths: Sequence[float],
    busy_fractions: Sequence[float],
    clock_hours: float,
    horizon_hours: float,
    jobs_completed: int,
    jobs_in_progress: int,
    total_jobs: int,
    mean_slack_hours: float,
    slack_scale: float,
) -> np.ndarray:
    """Turn raw floor state into the vector the policy expects.

    The single source of truth for the observation layout. The training
    environment and the serving API both call this, because a policy served a
    differently-scaled observation than it trained on is not merely degraded --
    it is reading noise, and nothing about the response would look wrong.
    """
    total = max(1, total_jobs)
    scale = slack_scale if slack_scale > 0 else 1.0
    observation = np.array(
        [
            *[np.tanh(q / QUEUE_SCALE) for q in queue_lengths],
            *busy_fractions,
            min(1.0, clock_hours / horizon_hours) if horizon_hours > 0 else 0.0,
            jobs_completed / total,
            jobs_in_progress / total,
            np.tanh(mean_slack_hours / scale),
        ],
        dtype=np.float32,
    )
    if observation.shape != (OBS_DIM,):
        raise ValueError(
            f"expected {OBS_DIM} features, built {observation.shape[0]} -- "
            f"got {len(queue_lengths)} queues and {len(busy_fractions)} "
            "busy fractions"
        )
    return np.clip(observation, -1.0, 1.0)


@dataclass(frozen=True)
class RewardConfig:
    """Coefficients for the reward terms.

    Four terms score the schedule -- throughput, utilisation, tardiness, and a
    terminal makespan penalty -- and a fifth, ``shaping``, redistributes the
    tardiness cost through the episode without changing the total.

    Every term is normalised to roughly unit scale before weighting, so these
    numbers are comparable to each other and a change of factory size does not
    silently rescale the reward. The defaults put throughput and lateness in
    tension, which is where scheduling lives.
    """

    throughput: float = 1.0
    utilisation: float = 0.3
    tardiness: float = 1.0
    #: Applied once, at termination, against the normalised makespan.
    makespan: float = 0.5

    #: Potential-based shaping on the projected lateness of work in progress.
    #:
    #: Weighted tardiness is only *realised* when a job finishes, tens of
    #: decisions after the dispatch that caused it, while the throughput term
    #: pays out immediately. That gap is a myopia trap: measured on this line,
    #: the reward ranking of four dispatch rules at step 15 is exactly the
    #: reverse of their ranking over the full episode. An agent following the
    #: early gradient walks into LWKR, which is the worst rule of the four.
    #:
    #: Shaping charges lateness as it accrues instead. Because the potential is
    #: zero with an empty floor -- true at both the start and the end of an
    #: episode -- the shaping telescopes to zero and total episode return is
    #: unchanged. The optimal policy is provably preserved (Ng et al., 1999);
    #: only the per-step credit assignment improves.
    shaping: float = 1.0

    def __post_init__(self) -> None:
        if self.tardiness < 0 or self.makespan < 0:
            raise ValueError("penalty coefficients must be non-negative")
        if self.shaping < 0:
            raise ValueError("shaping coefficient must be non-negative")


class FactorySchedulingEnv(gym.Env):
    """Tune the composite dispatch rule while the line runs.

    Observation (16-dim, all bounded to [-1, 1]):

    ==========  ====================================================
    index       meaning
    ==========  ====================================================
    0-5         queue length per station, squashed by ``tanh(q / 5)``
    6-11        busy machines / capacity per station
    12          clock / expected horizon
    13          fraction of jobs completed
    14          WIP / total jobs
    15          mean due-date slack of WIP, squashed
    ==========  ====================================================

    Action: the four dispatch weights, in [-1, 1]. See
    :data:`~dtmo.digital_twin.dispatch.CLASSICAL_RULES` for the corners of that
    space that correspond to textbook rules.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: FactoryConfig | None = None,
        reward: RewardConfig | None = None,
        decision_interval: float = 8.0,
        randomise_seed: bool = True,
        max_steps: int | None = None,
        normalise_action: bool = True,
    ) -> None:
        super().__init__()
        self.config = (config or load_config()).validate()
        self.reward_config = reward or RewardConfig()
        if decision_interval <= 0:
            raise ValueError("decision_interval must be > 0")
        self.decision_interval = float(decision_interval)
        self.randomise_seed = randomise_seed
        self.normalise_action = normalise_action

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_FEATURES,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Expected time to release every job, plus slack for the line to drain.
        # Used to normalise the clock and the terminal makespan term.
        self.horizon = 2.0 * self.config.n_jobs / self.config.arrival_rate
        self.max_steps = (
            max_steps
            if max_steps is not None
            else int(np.ceil(self.horizon / self.decision_interval))
        )
        #: Expected completions per decision interval, for reward scaling.
        self._completions_per_step = self.config.arrival_rate * self.decision_interval
        self._slack_scale = float(
            np.mean([family.planned_work for family in self.config.families])
        )
        self._mean_weight = float(
            np.mean([family.weight for family in self.config.families])
        )

        self.model = FactoryModel(self.config)
        self._steps = 0
        self._prev_completed = 0
        self._prev_tardiness = 0.0
        self._prev_potential = 0.0
        self._prev_busy: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            episode_seed = int(seed)
        elif self.randomise_seed:
            # A fresh job set each episode. Training on one fixed instance
            # teaches the agent that instance, not scheduling.
            episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        else:
            episode_seed = self.config.seed

        self.model = FactoryModel(self.config, seed=episode_seed)
        self.model.reset()
        self._steps = 0
        self._prev_completed = 0
        self._prev_tardiness = 0.0
        # Empty floor at t=0, so the potential starts at zero.
        self._prev_potential = 0.0
        self._prev_busy = {name: 0.0 for name in self.model.stations}
        return self._observation(), self._info()

    def step(
        self, action: Sequence[float]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(
            np.asarray(action, dtype=np.float64).reshape(-1),
            self.action_space.low,
            self.action_space.high,
        )
        if action.shape != (N_FEATURES,):
            raise ValueError(
                f"expected an action of shape ({N_FEATURES},), got {action.shape}"
            )
        self.model.set_weights(self._to_weights(action))

        self.model.advance(self.model.now + self.decision_interval)
        self._steps += 1

        terminated = self.model.is_complete
        truncated = (not terminated) and self._steps >= self.max_steps
        reward = self._reward(terminated)
        return self._observation(), reward, terminated, truncated, self._info()

    def _to_weights(self, action: np.ndarray) -> np.ndarray:
        """Project an action onto the unit sphere.

        The dispatch rule scores jobs as ``w . x`` and takes the argmin, so
        scaling every weight by a positive constant cannot change which job is
        chosen -- verified empirically: reward is identical to four decimals
        across an 8x range of magnitudes. Magnitude is therefore a dead
        dimension of a 4-D action space, and worse, near the origin a tiny
        change in the action swings the *direction* wildly, which is exactly
        where an untrained policy starts.

        Normalising costs no expressiveness (every reachable rule is still
        reachable, and the classical rules are unit-norm already) and hands the
        agent a well-conditioned search over directions instead.

        A near-zero action has no direction to speak of; it is passed through,
        which the rule reads as "no preference" and serves in queue order.
        """
        if not self.normalise_action:
            return action
        norm = float(np.linalg.norm(action))
        if norm < 1e-6:
            return action
        return action / norm

    def render(self) -> None:
        print(
            f"t={self.model.now:7.1f}h  "
            f"done={len(self.model.completed):3d}/{self.config.n_jobs}  "
            f"wip={len(self.model.wip):3d}  "
            f"w={np.round(self.model.dispatcher.weights, 2)}"
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        stations = [self.model.stations[name] for name in self.config.station_names]
        wip = self.model.wip
        now = self.model.now
        mean_slack = (
            float(np.mean([job.slack(now) for job in wip])) if wip else 0.0
        )
        return encode_observation(
            queue_lengths=[station.queue_length for station in stations],
            busy_fractions=[
                station.busy_machines / station.capacity for station in stations
            ],
            clock_hours=now,
            horizon_hours=self.horizon,
            jobs_completed=len(self.model.completed),
            jobs_in_progress=len(wip),
            total_jobs=self.config.n_jobs,
            mean_slack_hours=mean_slack,
            slack_scale=self._slack_scale,
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _potential(self) -> float:
        """Negative projected weighted lateness of everything on the floor.

        For each job in progress, ``now + remaining_work - due`` is the
        earliest lateness it can still achieve, assuming it never queues again.
        It is a lower bound on the tardiness that job will eventually book.

        Zero when the floor is empty, which is exactly the state at both the
        start and the end of an episode -- that is what makes the shaping
        telescope away and leave total return untouched.
        """
        now = self.model.now
        total = 0.0
        for job in self.model.wip:
            lateness = now + job.remaining_work - job.due_date
            if lateness > 0.0:
                total += job.family.weight * lateness
        return -total / max(self._mean_weight * self.decision_interval, 1e-9)

    def _reward(self, terminated: bool) -> float:
        cfg = self.reward_config

        completed = len(self.model.completed)
        progress = completed - self._prev_completed
        self._prev_completed = completed

        tardiness = self.model.total_weighted_tardiness
        new_tardiness = tardiness - self._prev_tardiness
        self._prev_tardiness = tardiness

        # Time-averaged utilisation over the interval just simulated, rather
        # than an instantaneous sample -- far less noisy to learn from.
        busy_delta = 0.0
        capacity_hours = 0.0
        for name, station in self.model.stations.items():
            busy_delta += station.busy_time - self._prev_busy[name]
            self._prev_busy[name] = station.busy_time
            capacity_hours += station.capacity * self.decision_interval
        utilisation = busy_delta / capacity_hours if capacity_hours else 0.0

        reward = (
            cfg.throughput * (progress / max(self._completions_per_step, 1e-9))
            + cfg.utilisation * utilisation
            - cfg.tardiness
            * (new_tardiness / max(self._mean_weight * self.decision_interval, 1e-9))
        )

        if cfg.shaping:
            potential = self._potential()
            reward += cfg.shaping * (potential - self._prev_potential)
            self._prev_potential = potential

        if terminated:
            reward -= cfg.makespan * (self.model.makespan / self.horizon)
        return float(reward)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _info(self) -> dict[str, Any]:
        decisions = sum(s.dispatch_decisions for s in self.model.stations.values())
        contested = sum(s.contested_decisions for s in self.model.stations.values())
        info: dict[str, Any] = {
            "time": self.model.now,
            "completed": len(self.model.completed),
            "wip": len(self.model.wip),
            "weighted_tardiness": self.model.total_weighted_tardiness,
            "weights": self.model.dispatcher.weights,
            "contested_fraction": contested / decisions if decisions else 0.0,
        }
        if self.model.is_complete:
            info["kpis"] = self.model.kpis()
        return info
