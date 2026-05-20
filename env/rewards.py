"""Reward shaping V1 for RLStriker."""

from __future__ import annotations

from typing import Any

from env import constants as C
from env.entities import Ball, Player

# V6 reward constants
REWARD_GOAL = 100.0
REWARD_CONCEDE = -100.0
REWARD_TOUCH = 2.0
REWARD_PROGRESS = 0.2
REWARD_STEP_PENALTY = -0.01

# Minimum ball x movement (pixels) toward opponent goal to grant progress reward
PROGRESS_MIN_DELTA = 0.5


def _touching(player: Player, ball: Ball) -> bool:
    dist = player.distance_to(ball.x, ball.y)
    return dist <= C.PLAYER_RADIUS + C.BALL_RADIUS + 0.5


def _ball_progress_toward_opponent_goal(team: int, prev_ball_x: float, ball_x: float) -> bool:
    """Team 1 attacks right; team 2 attacks left."""
    delta = ball_x - prev_ball_x
    if team == 1:
        return delta >= PROGRESS_MIN_DELTA
    return delta <= -PROGRESS_MIN_DELTA


def compute_step_rewards(
    player_1: Player,
    player_2: Player,
    ball: Ball,
    info: dict[str, Any],
    *,
    prev_ball_x: float,
) -> tuple[float, float]:
    """
    Per-step rewards for both agents.

    - +100 / -100 on goal scored / conceded
    - +2 per step while touching the ball
    - +0.2 if ball moved toward the opponent's goal this step
    - -0.01 time penalty every step
    """
    reward_1 = REWARD_STEP_PENALTY
    reward_2 = REWARD_STEP_PENALTY

    if _touching(player_1, ball):
        reward_1 += REWARD_TOUCH
    if _touching(player_2, ball):
        reward_2 += REWARD_TOUCH

    if _ball_progress_toward_opponent_goal(1, prev_ball_x, ball.x):
        reward_1 += REWARD_PROGRESS
    if _ball_progress_toward_opponent_goal(2, prev_ball_x, ball.x):
        reward_2 += REWARD_PROGRESS

    scorer = info.get("scorer")
    if scorer == 1:
        reward_1 += REWARD_GOAL
        reward_2 += REWARD_CONCEDE
    elif scorer == 2:
        reward_1 += REWARD_CONCEDE
        reward_2 += REWARD_GOAL

    return reward_1, reward_2
