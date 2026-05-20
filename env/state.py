"""State helpers for the RLStriker environment."""

from __future__ import annotations

from dataclasses import dataclass

from env.entities import Ball, Player


@dataclass
class EnvState:
    """Compact environment state for RL loops."""

    agent_1_x: float
    agent_1_y: float
    agent_2_x: float
    agent_2_y: float
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    steps: int

    def to_list(self) -> list[float]:
        return [
            self.agent_1_x,
            self.agent_1_y,
            self.agent_2_x,
            self.agent_2_y,
            self.ball_x,
            self.ball_y,
            self.ball_vx,
            self.ball_vy,
            float(self.steps),
        ]


def build_state(player_1: Player, player_2: Player, ball: Ball, steps: int) -> EnvState:
    return EnvState(
        agent_1_x=player_1.x,
        agent_1_y=player_1.y,
        agent_2_x=player_2.x,
        agent_2_y=player_2.y,
        ball_x=ball.x,
        ball_y=ball.y,
        ball_vx=ball.vx,
        ball_vy=ball.vy,
        steps=steps,
    )
