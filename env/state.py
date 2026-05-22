"""State helpers for the RLStriker environment."""

from __future__ import annotations

import math
from dataclasses import dataclass

from env import constants as C
from env.entities import Ball, Player


@dataclass
class EnvState:
    """V10 state vector with position, ball, geometry, and possession signals."""

    agent_1_x: float
    agent_1_y: float
    agent_2_x: float
    agent_2_y: float
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    agent_1_distance_to_ball: float
    agent_1_angle_to_ball: float
    agent_2_distance_to_ball: float
    agent_2_angle_to_ball: float
    ball_distance_to_agent_1_goal: float
    ball_distance_to_agent_2_goal: float
    ball_moving_toward_agent_1_goal: float
    ball_moving_toward_agent_2_goal: float
    last_touch_owner: float
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
            self.agent_1_distance_to_ball,
            self.agent_1_angle_to_ball,
            self.agent_2_distance_to_ball,
            self.agent_2_angle_to_ball,
            self.ball_distance_to_agent_1_goal,
            self.ball_distance_to_agent_2_goal,
            self.ball_moving_toward_agent_1_goal,
            self.ball_moving_toward_agent_2_goal,
            self.last_touch_owner,
            float(self.steps),
        ]


def build_state(
    player_1: Player,
    player_2: Player,
    ball: Ball,
    steps: int,
    last_touch_owner: int | None,
) -> EnvState:
    agent_1_goal = (C.FIELD_LEFT, C.FIELD_CENTER_Y)
    agent_2_goal = (C.FIELD_RIGHT, C.FIELD_CENTER_Y)

    return EnvState(
        agent_1_x=player_1.x,
        agent_1_y=player_1.y,
        agent_2_x=player_2.x,
        agent_2_y=player_2.y,
        ball_x=ball.x,
        ball_y=ball.y,
        ball_vx=ball.vx,
        ball_vy=ball.vy,
        agent_1_distance_to_ball=player_1.distance_to(ball.x, ball.y),
        agent_1_angle_to_ball=_angle_to_ball(player_1, ball),
        agent_2_distance_to_ball=player_2.distance_to(ball.x, ball.y),
        agent_2_angle_to_ball=_angle_to_ball(player_2, ball),
        ball_distance_to_agent_1_goal=_distance(ball.x, ball.y, *agent_1_goal),
        ball_distance_to_agent_2_goal=_distance(ball.x, ball.y, *agent_2_goal),
        ball_moving_toward_agent_1_goal=1.0 if ball.vx < 0 else 0.0,
        ball_moving_toward_agent_2_goal=1.0 if ball.vx > 0 else 0.0,
        last_touch_owner=float(last_touch_owner or 0),
        steps=steps,
    )


def state_size() -> int:
    """Return the current V10 state vector length."""
    return len(
        build_state(
            Player(0.0, 0.0, (0, 0, 0), "tmp1", team=1),
            Player(0.0, 0.0, (0, 0, 0), "tmp2", team=2),
            Ball(0.0, 0.0),
            steps=0,
            last_touch_owner=None,
        ).to_list()
    )


def _angle_to_ball(player: Player, ball: Ball) -> float:
    return math.atan2(ball.y - player.y, ball.x - player.x)


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)
