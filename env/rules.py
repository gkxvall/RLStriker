"""Match rules: goals, scoring, and episode lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

from env import constants as C
from env.entities import Ball


def goal_y_bounds() -> tuple[float, float]:
    top = C.FIELD_CENTER_Y - C.GOAL_HEIGHT / 2
    bottom = top + C.GOAL_HEIGHT
    return top, bottom


def check_goal(ball: Ball) -> int | None:
    """
    Return the scoring team (1 or 2), or None if no goal.
    Team 1 attacks the right goal; team 2 attacks the left goal.
    """
    top, bottom = goal_y_bounds()
    in_goal_height = top <= ball.y <= bottom

    if not in_goal_height:
        return None

    if ball.x - C.BALL_RADIUS <= C.FIELD_LEFT:
        return 2
    if ball.x + C.BALL_RADIUS >= C.FIELD_RIGHT:
        return 1
    return None


@dataclass
class MatchState:
    score_team_1: int = 0
    score_team_2: int = 0
    steps: int = 0
    done: bool = False
    last_scorer: int | None = None
    end_reason: str = ""

    def reset_scores(self) -> None:
        self.score_team_1 = 0
        self.score_team_2 = 0

    def new_episode(self) -> None:
        self.steps = 0
        self.done = False
        self.last_scorer = None
        self.end_reason = ""

    def step_after_physics(self, ball: Ball) -> dict:
        """Advance episode clock and check termination. Call once per frame."""
        info: dict = {"goal": False, "scorer": None, "max_steps": False}

        if self.done:
            return info

        self.steps += 1
        scorer = check_goal(ball)
        if scorer is not None:
            self._register_goal(scorer, info)
            return info

        if self.steps >= C.MAX_STEPS:
            self.done = True
            self.end_reason = f"Max steps ({C.MAX_STEPS}) reached"
            info["max_steps"] = True

        return info

    def _register_goal(self, scorer: int, info: dict) -> None:
        if scorer == 1:
            self.score_team_1 += 1
        else:
            self.score_team_2 += 1
        self.last_scorer = scorer
        self.done = True
        self.end_reason = f"Goal for team {scorer}"
        info["goal"] = True
        info["scorer"] = scorer
