"""Random policy agent for baseline simulations."""

from __future__ import annotations

import random

from env.soccer_env import ACTION_SPACE_SIZE


class RandomAgent:
    """Selects a uniformly random discrete action each step."""

    def act(self) -> int:
        return random.randrange(ACTION_SPACE_SIZE)

