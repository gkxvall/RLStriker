"""Curriculum stage scheduling and reward shaping."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from env import constants as C
from env.soccer_env import ACTION_STAY

from curriculum.stages import CurriculumStage, DEFAULT_CURRICULUM_STAGES

if TYPE_CHECKING:
    from agents.dqn_agent import DQNAgent
    from agents.random_agent import RandomAgent
    from env.soccer_env import SoccerEnv


REACH_DISTANCE_SCALE = 0.03
PUSH_DISTANCE_SCALE = 0.02
CURRICULUM_TOUCH_REWARD = 5.0
CURRICULUM_PROGRESS_REWARD = 0.8
CURRICULUM_STEP_PENALTY = -0.005
WEAK_RANDOM_ACTION_CHANCE = 0.35


@dataclass(frozen=True)
class CurriculumStepContext:
    """Signals captured before an env step so stage rewards can compare progress."""

    prev_distance_to_ball: float
    prev_ball_x: float


class CurriculumManager:
    """Controls V9 stage selection, opponent behavior, and curriculum rewards."""

    def __init__(
        self,
        *,
        total_episodes: int,
        stage_episodes: list[int] | None = None,
        stages: list[CurriculumStage] | None = None,
    ) -> None:
        self.stages = stages or DEFAULT_CURRICULUM_STAGES
        self.total_episodes = total_episodes
        self.stage_episodes = stage_episodes or self._split_episodes(total_episodes, len(self.stages))
        if len(self.stage_episodes) != len(self.stages):
            raise ValueError("stage_episodes must have the same length as stages")
        if any(count <= 0 for count in self.stage_episodes):
            raise ValueError("each curriculum stage must receive at least one episode")

        self._stage_ends: list[int] = []
        running_total = 0
        for count in self.stage_episodes:
            running_total += count
            self._stage_ends.append(running_total)

    def stage_for_episode(self, episode: int) -> CurriculumStage:
        for stage, end_episode in zip(self.stages, self._stage_ends):
            if episode <= end_episode:
                return stage
        return self.stages[-1]

    def describe_schedule(self) -> list[dict[str, object]]:
        start = 1
        schedule = []
        for stage, end in zip(self.stages, self._stage_ends):
            schedule.append(
                {
                    **stage.as_dict(),
                    "start_episode": start,
                    "end_episode": end,
                    "episodes": end - start + 1,
                }
            )
            start = end + 1
        return schedule

    def configure_episode(self, env: SoccerEnv, stage: CurriculumStage, train_agent: int) -> None:
        """Apply simple stage setup after env.reset()."""
        if stage.opponent_mode != "none":
            return

        learner = env.player_1 if train_agent == 1 else env.player_2
        opponent = env.player_2 if train_agent == 1 else env.player_1

        learner_start_x = C.FIELD_LEFT + C.FIELD_WIDTH * 0.25 if train_agent == 1 else C.FIELD_LEFT + C.FIELD_WIDTH * 0.75
        opponent_x = C.FIELD_RIGHT - C.PLAYER_RADIUS if train_agent == 1 else C.FIELD_LEFT + C.PLAYER_RADIUS

        learner.reset(learner_start_x, C.FIELD_CENTER_Y)
        opponent.reset(opponent_x, C.FIELD_BOTTOM - C.PLAYER_RADIUS)
        env.ball.reset(C.FIELD_CENTER_X, C.FIELD_CENTER_Y)

    def before_step(self, env: SoccerEnv, train_agent: int) -> CurriculumStepContext:
        learner = env.player_1 if train_agent == 1 else env.player_2
        return CurriculumStepContext(
            prev_distance_to_ball=learner.distance_to(env.ball.x, env.ball.y),
            prev_ball_x=env.ball.x,
        )

    def opponent_action(
        self,
        *,
        stage: CurriculumStage,
        state: list[float],
        opponent: RandomAgent,
        learner: DQNAgent,
    ) -> int:
        if stage.opponent_mode == "none":
            return ACTION_STAY
        if stage.opponent_mode == "weak_random":
            if random.random() < WEAK_RANDOM_ACTION_CHANCE:
                return opponent.act()
            return ACTION_STAY
        if stage.opponent_mode == "self_play":
            return learner.act(state, training=False)
        return opponent.act()

    def training_reward(
        self,
        *,
        env: SoccerEnv,
        stage: CurriculumStage,
        train_agent: int,
        base_reward: float,
        context: CurriculumStepContext,
    ) -> float:
        if stage.reward_mode == "match":
            return base_reward

        learner = env.player_1 if train_agent == 1 else env.player_2
        current_distance = learner.distance_to(env.ball.x, env.ball.y)
        distance_delta = context.prev_distance_to_ball - current_distance
        touch_bonus = CURRICULUM_TOUCH_REWARD if _touching_ball(learner.x, learner.y, env.ball.x, env.ball.y) else 0.0

        if stage.reward_mode == "reach_ball":
            return CURRICULUM_STEP_PENALTY + (distance_delta * REACH_DISTANCE_SCALE) + touch_bonus

        if stage.reward_mode == "push_to_goal":
            progress_bonus = CURRICULUM_PROGRESS_REWARD if _ball_progress(train_agent, context.prev_ball_x, env.ball.x) else 0.0
            return (
                CURRICULUM_STEP_PENALTY
                + (distance_delta * PUSH_DISTANCE_SCALE)
                + touch_bonus
                + progress_bonus
            )

        return base_reward

    def keep_playing_if_scoring_disabled(
        self,
        *,
        env: SoccerEnv,
        stage: CurriculumStage,
        info: dict[str, object],
    ) -> bool:
        """Return True when a goal was ignored and the episode should continue."""
        if stage.scoring_enabled or not info.get("goal"):
            return False

        scorer = info.get("scorer")
        if scorer == 1:
            env.match.score_team_1 = max(0, env.match.score_team_1 - 1)
        elif scorer == 2:
            env.match.score_team_2 = max(0, env.match.score_team_2 - 1)

        env.match.done = False
        env.match.last_scorer = None
        env.match.end_reason = ""
        env.ball.reset(C.FIELD_CENTER_X, C.FIELD_CENTER_Y)
        info["goal"] = False
        info["scorer"] = None
        info["curriculum_goal_ignored"] = True
        return True

    @staticmethod
    def parse_stage_episodes(value: str | None) -> list[int] | None:
        if value is None:
            return None
        try:
            counts = [int(part.strip()) for part in value.split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError("stage episode counts must be comma-separated integers") from exc
        return counts or None

    @staticmethod
    def _split_episodes(total_episodes: int, stage_count: int) -> list[int]:
        if total_episodes < stage_count:
            raise ValueError(f"curriculum mode needs at least {stage_count} episodes")

        base = total_episodes // stage_count
        remainder = total_episodes % stage_count
        return [base + (1 if i < remainder else 0) for i in range(stage_count)]


def _touching_ball(player_x: float, player_y: float, ball_x: float, ball_y: float) -> bool:
    return math.hypot(player_x - ball_x, player_y - ball_y) <= C.PLAYER_RADIUS + C.BALL_RADIUS + 0.5


def _ball_progress(train_agent: int, prev_ball_x: float, ball_x: float) -> bool:
    delta = ball_x - prev_ball_x
    if train_agent == 1:
        return delta >= 0.5
    return delta <= -0.5
