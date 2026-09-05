"""Export the trained policy to plain numpy, so serving needs no PyTorch.

The policy is a 16-64-64-4 network -- about 5,500 numbers, roughly 22 KB. Torch
and Stable-Baselines3 exist to *train* that; they contribute nothing at
inference beyond three matrix multiplies. Carrying them into the serving image
costs about 1.8 GB, which is most of the reason the container is slow to pull
and awkward to host on a free tier.

This module lifts the weights out and reimplements the forward pass. The export
is verified against the torch model at export time and again in the tests: if
they ever disagree by more than a rounding error, the export is wrong and
should not ship.

Only the *actor* is exported. The value network is a training artefact and is
never consulted when serving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

#: Deterministic prediction uses the distribution's mean, which for the
#: continuous policy is the action network's raw output. There is no squashing
#: (``squash_output=False``), so the only post-processing is the clip to the
#: action space bounds.
ACTION_LOW, ACTION_HIGH = -1.0, 1.0


class NumpyPolicy:
    """The trained policy, as three matrix multiplies.

    Same interface as :class:`~dtmo.agents.policies.SB3Policy`, so anything
    that scores or serves a policy accepts this without changes.
    """

    def __init__(self, params: dict[str, np.ndarray], name: str = "ppo") -> None:
        self.name = name
        self._w0 = params["w0"]
        self._b0 = params["b0"]
        self._w1 = params["w1"]
        self._b1 = params["b1"]
        self._wa = params["wa"]
        self._ba = params["ba"]

    def __repr__(self) -> str:
        return (
            f"NumpyPolicy({self.name!r}, "
            f"{self._w0.shape[1]}-{self._w0.shape[0]}-"
            f"{self._w1.shape[0]}-{self._wa.shape[0]})"
        )

    @property
    def n_parameters(self) -> int:
        return sum(
            arr.size
            for arr in (self._w0, self._b0, self._w1, self._b1, self._wa, self._ba)
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float32).reshape(-1)
        x = np.tanh(self._w0 @ x + self._b0)
        x = np.tanh(self._w1 @ x + self._b1)
        action = self._wa @ x + self._ba
        return np.clip(action, ACTION_LOW, ACTION_HIGH).astype(np.float32)

    def reset(self) -> None:
        return None

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, name: str = "ppo") -> "NumpyPolicy":
        with np.load(str(path)) as data:
            return cls({key: data[key] for key in data.files}, name=name)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w0=self._w0,
            b0=self._b0,
            w1=self._w1,
            b1=self._b1,
            wa=self._wa,
            ba=self._ba,
        )
        return path


def extract(model: Any) -> NumpyPolicy:
    """Pull the actor's weights out of a loaded Stable-Baselines3 model."""
    state = {k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()}

    expected = [
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight",
        "mlp_extractor.policy_net.2.bias",
        "action_net.weight",
        "action_net.bias",
    ]
    missing = [key for key in expected if key not in state]
    if missing:
        raise ValueError(
            "this policy does not have the architecture the export assumes "
            f"(missing {missing}). Re-check net_arch before exporting."
        )
    if getattr(model.policy, "squash_output", False):
        raise ValueError(
            "policy squashes its output; the numpy forward pass does not "
            "reproduce that and would serve wrong actions"
        )

    return NumpyPolicy(
        {
            "w0": state["mlp_extractor.policy_net.0.weight"],
            "b0": state["mlp_extractor.policy_net.0.bias"],
            "w1": state["mlp_extractor.policy_net.2.weight"],
            "b1": state["mlp_extractor.policy_net.2.bias"],
            "wa": state["action_net.weight"],
            "ba": state["action_net.bias"],
        }
    )


def export_policy(
    model_path: str | Path,
    out_path: str | Path,
    n_checks: int = 512,
    tolerance: float = 1e-5,
) -> tuple[Path, float]:
    """Convert a saved SB3 model to a numpy policy, verifying they agree.

    Returns the written path and the worst disagreement observed. Raises if
    that exceeds ``tolerance`` -- an export that quietly diverges from the
    trained model is worse than no export, because nothing downstream would
    look wrong.
    """
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path))
    policy = extract(model)

    n_features = policy._w0.shape[1]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n_checks):
        observation = rng.uniform(-1.0, 1.0, size=n_features).astype(np.float32)
        reference, _ = model.predict(observation, deterministic=True)
        mine = policy.act(observation)
        worst = max(worst, float(np.max(np.abs(reference - mine))))

    if worst > tolerance:
        raise ValueError(
            f"numpy export disagrees with the torch model by {worst:.2e} "
            f"(tolerance {tolerance:.0e}) -- not safe to ship"
        )

    written = policy.save(out_path)
    return written, worst
