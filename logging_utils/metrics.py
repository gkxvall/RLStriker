"""Per-episode metrics accumulated during simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from env import constants as C
from env.rewards import COMPONENT_KEYS

if TYPE_CHECKING:
    from env.soccer_env import SoccerEnv


def _touching(player_x: float, player_y: float, ball_x: float, ball_y: float) -> bool:
    dist = math.hypot(player_x - ball_x, player_y - ball_y)
    return dist <= C.PLAYER_RADIUS + C.BALL_RADIUS + 0.5


@dataclass
class EpisodeMetricsRow:
    """One row for episodes.csv (values filled at episode end)."""

    run_name: str
    episode: int
    total_steps: int
    winner: str
    score_agent_1: int
    score_agent_2: int
    total_reward_agent_1: float
    total_reward_agent_2: float
    goals_agent_1: int
    goals_agent_2: int
    touches_agent_1: int
    touches_agent_2: int
    kicks_agent_1: int
    kicks_agent_2: int
    steals_agent_1: int
    steals_agent_2: int
    avg_distance_to_ball_agent_1: float
    avg_distance_to_ball_agent_2: float
    reward_components_agent_1: dict[str, float]
    reward_components_agent_2: dict[str, float]
    epsilon: float
    timestamp: str

    def as_csv_dict(self) -> dict[str, Any]:
        row = {
            "run_name": self.run_name,
            "episode": self.episode,
            "total_steps": self.total_steps,
            "winner": self.winner,
            "score_agent_1": self.score_agent_1,
            "score_agent_2": self.score_agent_2,
            "total_reward_agent_1": self.total_reward_agent_1,
            "total_reward_agent_2": self.total_reward_agent_2,
            "goals_agent_1": self.goals_agent_1,
            "goals_agent_2": self.goals_agent_2,
            "touches_agent_1": self.touches_agent_1,
            "touches_agent_2": self.touches_agent_2,
            "kicks_agent_1": self.kicks_agent_1,
            "kicks_agent_2": self.kicks_agent_2,
            "steals_agent_1": self.steals_agent_1,
            "steals_agent_2": self.steals_agent_2,
            "avg_distance_to_ball_agent_1": round(self.avg_distance_to_ball_agent_1, 4),
            "avg_distance_to_ball_agent_2": round(self.avg_distance_to_ball_agent_2, 4),
            "epsilon": self.epsilon,
            "timestamp": self.timestamp,
        }
        for key in COMPONENT_KEYS:
            row[f"{key}_agent_1"] = round(self.reward_components_agent_1.get(key, 0.0), 4)
            row[f"{key}_agent_2"] = round(self.reward_components_agent_2.get(key, 0.0), 4)
        return row


class EpisodeMetricsTracker:
    """Accumulates step-level signals into episode-level aggregates."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.touches_agent_1 = 0
        self.touches_agent_2 = 0
        self.kicks_agent_1 = 0
        self.kicks_agent_2 = 0
        self.steals_agent_1 = 0
        self.steals_agent_2 = 0
        self._sum_dist_1 = 0.0
        self._sum_dist_2 = 0.0
        self._last_touch_owner: int | None = None
        self.reward_components_agent_1 = {key: 0.0 for key in COMPONENT_KEYS}
        self.reward_components_agent_2 = {key: 0.0 for key in COMPONENT_KEYS}

    def on_step(self, env: SoccerEnv, info: dict[str, Any]) -> None:
        p1, p2, ball = env.player_1, env.player_2, env.ball
        t1 = _touching(p1.x, p1.y, ball.x, ball.y)
        t2 = _touching(p2.x, p2.y, ball.x, ball.y)

        if t1:
            self.touches_agent_1 += 1
        if t2:
            self.touches_agent_2 += 1

        if t1 and not t2 and self._last_touch_owner == 2:
            self.steals_agent_1 += 1
        if t2 and not t1 and self._last_touch_owner == 1:
            self.steals_agent_2 += 1

        if t1 and not t2:
            self._last_touch_owner = 1
        elif t2 and not t1:
            self._last_touch_owner = 2
        elif not t1 and not t2:
            self._last_touch_owner = None

        if info.get("kick_connected_1"):
            self.kicks_agent_1 += 1
        if info.get("kick_connected_2"):
            self.kicks_agent_2 += 1

        components = info.get("reward_components", {})
        self._add_reward_components(self.reward_components_agent_1, components.get("agent_1", {}))
        self._add_reward_components(self.reward_components_agent_2, components.get("agent_2", {}))

        self._sum_dist_1 += p1.distance_to(ball.x, ball.y)
        self._sum_dist_2 += p2.distance_to(ball.x, ball.y)

    def _add_reward_components(self, target: dict[str, float], components: dict[str, Any]) -> None:
        for key in COMPONENT_KEYS:
            target[key] += float(components.get(key, 0.0))

    def finalize(
        self,
        *,
        run_name: str,
        episode: int,
        total_steps: int,
        winner: str,
        score_agent_1: int,
        score_agent_2: int,
        total_reward_agent_1: float,
        total_reward_agent_2: float,
        goals_agent_1: int,
        goals_agent_2: int,
        epsilon: float,
        timestamp: str,
    ) -> EpisodeMetricsRow:
        n = max(total_steps, 1)
        avg1 = self._sum_dist_1 / n
        avg2 = self._sum_dist_2 / n
        return EpisodeMetricsRow(
            run_name=run_name,
            episode=episode,
            total_steps=total_steps,
            winner=winner,
            score_agent_1=score_agent_1,
            score_agent_2=score_agent_2,
            total_reward_agent_1=total_reward_agent_1,
            total_reward_agent_2=total_reward_agent_2,
            goals_agent_1=goals_agent_1,
            goals_agent_2=goals_agent_2,
            touches_agent_1=self.touches_agent_1,
            touches_agent_2=self.touches_agent_2,
            kicks_agent_1=self.kicks_agent_1,
            kicks_agent_2=self.kicks_agent_2,
            steals_agent_1=self.steals_agent_1,
            steals_agent_2=self.steals_agent_2,
            avg_distance_to_ball_agent_1=avg1,
            avg_distance_to_ball_agent_2=avg2,
            reward_components_agent_1=dict(self.reward_components_agent_1),
            reward_components_agent_2=dict(self.reward_components_agent_2),
            epsilon=epsilon,
            timestamp=timestamp,
        )
