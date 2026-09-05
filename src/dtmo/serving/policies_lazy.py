"""Import Stable-Baselines3 only if a torch model is actually requested.

The slim serving image has no torch. Importing SB3 at module load would make
the whole app fail to start there, even though the numpy policy needs none of
it. Deferring the import to the moment a `.zip` model is asked for keeps both
images running the same code.
"""

from __future__ import annotations

from typing import Any


class load_sb3_policy:  # noqa: N801 -- used as a namespace, not instantiated
    @staticmethod
    def load(path: str, name: str = "ppo") -> Any:
        from ..agents.policies import SB3Policy

        return SB3Policy.load(path, name=name)
