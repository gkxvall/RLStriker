"""Experience replay buffer for DQN training."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    """Fixed-size buffer that returns uniformly random transition batches."""

    def __init__(self, capacity: int) -> None:
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        self._buffer.append(
            Transition(
                state=list(state),
                action=action,
                reward=reward,
                next_state=list(next_state),
                done=done,
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)
