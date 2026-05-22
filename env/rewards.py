"""Reward shaping V2 for RLStriker."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from env import constants as C
from env.entities import Ball, Player

# V12 reward constants
REWARD_GOAL = 100.0
REWARD_CONCEDE = -100.0
REWARD_TOUCH = 2.0
REWARD_PROGRESS = 0.2
REWARD_STEAL = 8.0
REWARD_USEFUL_KICK = 1.0
REWARD_ATTACKING_POSITION = 0.5
PENALTY_OWN_GOAL = -1.0
PENALTY_UNNECESSARY_KICK = -0.05
PENALTY_PUSH_OWN_GOAL = -0.2
REWARD_STEP_PENALTY = -0.01
ENERGY_PENALTY_SCALE = -0.001

PROGRESS_MIN_DELTA = 0.5
ATTACKING_POSITION_MIN_DELTA = 0.25
USEFUL_KICK_MIN_DELTA = 0.5

COMPONENT_KEYS = [
    "goal_reward",
    "touch_reward",
    "progress_reward",
    "steal_reward",
    "useful_kick_reward",
    "attacking_position_reward",
    "own_goal_penalty",
    "unnecessary_kick_penalty",
    "energy_penalty",
    "own_goal_push_penalty",
    "time_penalty",
]


@dataclass
class RewardBreakdown:
    """Component-level reward values for one agent on one step."""

    goal_reward: float = 0.0
    touch_reward: float = 0.0
    progress_reward: float = 0.0
    steal_reward: float = 0.0
    useful_kick_reward: float = 0.0
    attacking_position_reward: float = 0.0
    own_goal_penalty: float = 0.0
    unnecessary_kick_penalty: float = 0.0
    energy_penalty: float = 0.0
    own_goal_push_penalty: float = 0.0
    time_penalty: float = REWARD_STEP_PENALTY

    @property
    def total(self) -> float:
        return sum(getattr(self, key) for key in COMPONENT_KEYS)

    def as_dict(self) -> dict[str, float]:
        return {key: getattr(self, key) for key in COMPONENT_KEYS} | {"total_reward": self.total}


def compute_step_rewards(
    player_1: Player,
    player_2: Player,
    ball: Ball,
    info: dict[str, Any],
    *,
    prev_ball_x: float,
    prev_ball_y: float,
    prev_distance_to_ball_1: float,
    prev_distance_to_ball_2: float,
    previous_touch_owner: int | None,
    action_1: int,
    action_2: int,
    move_1: tuple[float, float],
    move_2: tuple[float, float],
) -> tuple[float, float, dict[str, dict[str, float]]]:
    """
    V12 soccer-specific reward shaping with explicit component tracking.

    Team 1 attacks right. Team 2 attacks left.
    """
    breakdown_1 = RewardBreakdown()
    breakdown_2 = RewardBreakdown()

    _apply_goal_rewards(breakdown_1, breakdown_2, info, previous_touch_owner)
    _apply_touch_and_steal_rewards(player_1, player_2, ball, breakdown_1, breakdown_2, previous_touch_owner)
    _apply_ball_progress_rewards(ball, prev_ball_x, breakdown_1, breakdown_2)
    _apply_kick_rewards(action_1, action_2, info, prev_ball_x, ball.x, breakdown_1, breakdown_2)
    _apply_position_rewards(
        player_1,
        player_2,
        ball,
        prev_distance_to_ball_1,
        prev_distance_to_ball_2,
        breakdown_1,
        breakdown_2,
    )
    _apply_energy_penalties(move_1, move_2, breakdown_1, breakdown_2)

    components = {
        "agent_1": breakdown_1.as_dict(),
        "agent_2": breakdown_2.as_dict(),
    }
    return breakdown_1.total, breakdown_2.total, components


def _apply_goal_rewards(
    breakdown_1: RewardBreakdown,
    breakdown_2: RewardBreakdown,
    info: dict[str, Any],
    previous_touch_owner: int | None,
) -> None:
    scorer = info.get("scorer")
    scoring_touch_owner = info.get("last_touch_owner") or previous_touch_owner
    if scorer == 1:
        breakdown_1.goal_reward += REWARD_GOAL
        breakdown_2.goal_reward += REWARD_CONCEDE
        if scoring_touch_owner == 2:
            breakdown_2.own_goal_penalty += PENALTY_OWN_GOAL
    elif scorer == 2:
        breakdown_1.goal_reward += REWARD_CONCEDE
        breakdown_2.goal_reward += REWARD_GOAL
        if scoring_touch_owner == 1:
            breakdown_1.own_goal_penalty += PENALTY_OWN_GOAL


def _apply_touch_and_steal_rewards(
    player_1: Player,
    player_2: Player,
    ball: Ball,
    breakdown_1: RewardBreakdown,
    breakdown_2: RewardBreakdown,
    previous_touch_owner: int | None,
) -> None:
    touching_1 = _touching(player_1, ball)
    touching_2 = _touching(player_2, ball)

    if touching_1:
        breakdown_1.touch_reward += REWARD_TOUCH
        if previous_touch_owner == 2 and not touching_2:
            breakdown_1.steal_reward += REWARD_STEAL

    if touching_2:
        breakdown_2.touch_reward += REWARD_TOUCH
        if previous_touch_owner == 1 and not touching_1:
            breakdown_2.steal_reward += REWARD_STEAL


def _apply_ball_progress_rewards(
    ball: Ball,
    prev_ball_x: float,
    breakdown_1: RewardBreakdown,
    breakdown_2: RewardBreakdown,
) -> None:
    if _ball_progress_toward_opponent_goal(1, prev_ball_x, ball.x):
        breakdown_1.progress_reward += REWARD_PROGRESS
        breakdown_2.own_goal_push_penalty += PENALTY_PUSH_OWN_GOAL

    if _ball_progress_toward_opponent_goal(2, prev_ball_x, ball.x):
        breakdown_2.progress_reward += REWARD_PROGRESS
        breakdown_1.own_goal_push_penalty += PENALTY_PUSH_OWN_GOAL


def _apply_kick_rewards(
    action_1: int,
    action_2: int,
    info: dict[str, Any],
    prev_ball_x: float,
    ball_x: float,
    breakdown_1: RewardBreakdown,
    breakdown_2: RewardBreakdown,
) -> None:
    kick_connected_1 = bool(info.get("kick_connected_1"))
    kick_connected_2 = bool(info.get("kick_connected_2"))

    if kick_connected_1 and _ball_progress_toward_opponent_goal(1, prev_ball_x, ball_x, USEFUL_KICK_MIN_DELTA):
        breakdown_1.useful_kick_reward += REWARD_USEFUL_KICK
    if kick_connected_2 and _ball_progress_toward_opponent_goal(2, prev_ball_x, ball_x, USEFUL_KICK_MIN_DELTA):
        breakdown_2.useful_kick_reward += REWARD_USEFUL_KICK

    # ACTION_KICK is 5 in env.soccer_env; keep local to avoid an import cycle.
    if action_1 == 5 and not kick_connected_1:
        breakdown_1.unnecessary_kick_penalty += PENALTY_UNNECESSARY_KICK
    if action_2 == 5 and not kick_connected_2:
        breakdown_2.unnecessary_kick_penalty += PENALTY_UNNECESSARY_KICK


def _apply_position_rewards(
    player_1: Player,
    player_2: Player,
    ball: Ball,
    prev_distance_to_ball_1: float,
    prev_distance_to_ball_2: float,
    breakdown_1: RewardBreakdown,
    breakdown_2: RewardBreakdown,
) -> None:
    if _attacking_position_improved(1, player_1, ball, prev_distance_to_ball_1):
        breakdown_1.attacking_position_reward += REWARD_ATTACKING_POSITION
    if _attacking_position_improved(2, player_2, ball, prev_distance_to_ball_2):
        breakdown_2.attacking_position_reward += REWARD_ATTACKING_POSITION


def _apply_energy_penalties(
    move_1: tuple[float, float],
    move_2: tuple[float, float],
    breakdown_1: RewardBreakdown,
    breakdown_2: RewardBreakdown,
) -> None:
    breakdown_1.energy_penalty += ENERGY_PENALTY_SCALE * math.hypot(*move_1)
    breakdown_2.energy_penalty += ENERGY_PENALTY_SCALE * math.hypot(*move_2)


def _touching(player: Player, ball: Ball) -> bool:
    dist = player.distance_to(ball.x, ball.y)
    return dist <= C.PLAYER_RADIUS + C.BALL_RADIUS + 0.5


def _ball_progress_toward_opponent_goal(
    team: int,
    prev_ball_x: float,
    ball_x: float,
    min_delta: float = PROGRESS_MIN_DELTA,
) -> bool:
    delta = ball_x - prev_ball_x
    if team == 1:
        return delta >= min_delta
    return delta <= -min_delta


def _attacking_position_improved(team: int, player: Player, ball: Ball, prev_distance_to_ball: float) -> bool:
    current_distance = player.distance_to(ball.x, ball.y)
    closed_distance = current_distance <= prev_distance_to_ball - ATTACKING_POSITION_MIN_DELTA
    if not closed_distance:
        return False

    if team == 1:
        return player.x <= ball.x
    return player.x >= ball.x
