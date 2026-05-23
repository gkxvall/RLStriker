"""2v2 team soccer environment for RLStriker."""

from __future__ import annotations

import math
from typing import Any

import pygame

from env import constants as C
from env.entities import Ball, Player
from env.field import Field
from env.physics import kick_ball, update_physics
from env.rules import MatchState, check_goal
from env.soccer_env import ACTION_DOWN, ACTION_KICK, ACTION_LEFT, ACTION_RIGHT, ACTION_UP

TEAM_ACTION_SIZE = 2

REWARD_TEAM_GOAL = 100.0
REWARD_TEAM_CONCEDE = -100.0
REWARD_TEAM_PROGRESS = 0.3
REWARD_PASS = 4.0
REWARD_SPACING = 0.05
REWARD_TEAM_TOUCH = 0.5
STEP_PENALTY = -0.01
PASS_MIN_BALL_SPEED = 2.0
SPACING_MIN_DISTANCE = 95.0


class TeamSoccerEnv:
    """Small 2v2 soccer environment with team-level rewards."""

    def __init__(self, render_mode: str | None = None) -> None:
        self.render_mode = render_mode
        self.field = Field()
        self.match = MatchState()

        self.team_1: list[Player] = []
        self.team_2: list[Player] = []
        self.players: list[Player] = []
        self.ball = Ball(C.FIELD_CENTER_X, C.FIELD_CENTER_Y)
        self.last_touch_team: int | None = None
        self.last_touch_player: str | None = None

        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.big_font: pygame.font.Font | None = None

        self.reset()
        if self.render_mode == "human":
            self._init_pygame()

    def reset(self) -> list[float]:
        self.match.new_episode()
        self.team_1 = [
            Player(C.FIELD_LEFT + C.FIELD_WIDTH * 0.22, C.FIELD_CENTER_Y - 70, C.COLOR_PLAYER_1, "1A", team=1),
            Player(C.FIELD_LEFT + C.FIELD_WIDTH * 0.18, C.FIELD_CENTER_Y + 85, C.COLOR_PLAYER_1, "1B", team=1),
        ]
        self.team_2 = [
            Player(C.FIELD_LEFT + C.FIELD_WIDTH * 0.78, C.FIELD_CENTER_Y - 70, C.COLOR_PLAYER_2, "2A", team=2),
            Player(C.FIELD_LEFT + C.FIELD_WIDTH * 0.82, C.FIELD_CENTER_Y + 85, C.COLOR_PLAYER_2, "2B", team=2),
        ]
        self.players = [*self.team_1, *self.team_2]
        self.ball = Ball(C.FIELD_CENTER_X, C.FIELD_CENTER_Y)
        self.last_touch_team = None
        self.last_touch_player = None
        return self._get_state()

    def step(
        self,
        team_1_actions: list[int] | tuple[int, int],
        team_2_actions: list[int] | tuple[int, int],
    ) -> tuple[list[float], float, float, bool, dict[str, Any]]:
        if self.match.done:
            return self._get_state(), 0.0, 0.0, True, {"note": "Episode is done. Call reset()."}

        actions = self._normalize_actions(team_1_actions) + self._normalize_actions(team_2_actions)
        previous_ball_x = self.ball.x
        previous_touch_team = self.last_touch_team
        previous_touch_player = self.last_touch_player

        moves = [self._action_to_move(action) for action in actions]
        kick_connected = [False, False, False, False]
        for index, (player, action) in enumerate(zip(self.players, actions)):
            if action == ACTION_KICK:
                kick_connected[index] = kick_ball(player, self.ball)

        update_physics(self.players, self.ball, moves)
        self._update_last_touch(kick_connected)
        info = self._step_rules()
        info["kick_connected"] = kick_connected
        info["last_touch_team"] = self.last_touch_team
        info["last_touch_player"] = self.last_touch_player

        reward_1, reward_2, reward_info = self._team_rewards(
            previous_ball_x=previous_ball_x,
            previous_touch_team=previous_touch_team,
            previous_touch_player=previous_touch_player,
            info=info,
        )
        info.update(reward_info)
        return self._get_state(), reward_1, reward_2, self.match.done, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self.screen is None:
            self._init_pygame()

        assert self.screen is not None
        assert self.font is not None
        assert self.big_font is not None

        self.screen.fill((20, 20, 20))
        self.field.draw(self.screen)
        for player in self.players:
            player.draw(self.screen)
        self.ball.draw(self.screen)
        self._draw_ui()

    def close(self) -> None:
        if self.render_mode == "human":
            pygame.quit()
            self.screen = None

    def _normalize_actions(self, actions: list[int] | tuple[int, int]) -> list[int]:
        values = list(actions)
        if len(values) != TEAM_ACTION_SIZE:
            raise ValueError(f"Expected {TEAM_ACTION_SIZE} actions per team, got {len(values)}")
        return values

    def _step_rules(self) -> dict[str, Any]:
        info: dict[str, Any] = {"goal": False, "scorer": None, "max_steps": False}
        self.match.steps += 1
        scorer = check_goal(self.ball)
        if scorer is not None:
            self.match._register_goal(scorer, info)
            return info
        if self.match.steps >= C.MAX_STEPS:
            self.match.done = True
            self.match.end_reason = f"Max steps ({C.MAX_STEPS}) reached"
            info["max_steps"] = True
        return info

    def _team_rewards(
        self,
        *,
        previous_ball_x: float,
        previous_touch_team: int | None,
        previous_touch_player: str | None,
        info: dict[str, Any],
    ) -> tuple[float, float, dict[str, Any]]:
        reward_1 = STEP_PENALTY
        reward_2 = STEP_PENALTY
        components = {
            "team_1": {"goal": 0.0, "progress": 0.0, "touch": 0.0, "pass": 0.0, "spacing": 0.0},
            "team_2": {"goal": 0.0, "progress": 0.0, "touch": 0.0, "pass": 0.0, "spacing": 0.0},
        }

        scorer = info.get("scorer")
        if scorer == 1:
            reward_1 += REWARD_TEAM_GOAL
            reward_2 += REWARD_TEAM_CONCEDE
            components["team_1"]["goal"] += REWARD_TEAM_GOAL
            components["team_2"]["goal"] += REWARD_TEAM_CONCEDE
        elif scorer == 2:
            reward_1 += REWARD_TEAM_CONCEDE
            reward_2 += REWARD_TEAM_GOAL
            components["team_1"]["goal"] += REWARD_TEAM_CONCEDE
            components["team_2"]["goal"] += REWARD_TEAM_GOAL

        delta_x = self.ball.x - previous_ball_x
        if delta_x > 0.5:
            reward_1 += REWARD_TEAM_PROGRESS
            components["team_1"]["progress"] += REWARD_TEAM_PROGRESS
        elif delta_x < -0.5:
            reward_2 += REWARD_TEAM_PROGRESS
            components["team_2"]["progress"] += REWARD_TEAM_PROGRESS

        if self.last_touch_team == 1:
            reward_1 += REWARD_TEAM_TOUCH
            components["team_1"]["touch"] += REWARD_TEAM_TOUCH
        elif self.last_touch_team == 2:
            reward_2 += REWARD_TEAM_TOUCH
            components["team_2"]["touch"] += REWARD_TEAM_TOUCH

        if (
            self.last_touch_team is not None
            and self.last_touch_team == previous_touch_team
            and self.last_touch_player is not None
            and previous_touch_player is not None
            and self.last_touch_player != previous_touch_player
            and self.ball.speed() >= PASS_MIN_BALL_SPEED
        ):
            if self.last_touch_team == 1:
                reward_1 += REWARD_PASS
                components["team_1"]["pass"] += REWARD_PASS
            else:
                reward_2 += REWARD_PASS
                components["team_2"]["pass"] += REWARD_PASS

        spacing_1 = self._spacing_reward(self.team_1)
        spacing_2 = self._spacing_reward(self.team_2)
        reward_1 += spacing_1
        reward_2 += spacing_2
        components["team_1"]["spacing"] += spacing_1
        components["team_2"]["spacing"] += spacing_2

        return reward_1, reward_2, {"team_reward_components": components}

    def _spacing_reward(self, team: list[Player]) -> float:
        distance = team[0].distance_to(team[1].x, team[1].y)
        return REWARD_SPACING if distance >= SPACING_MIN_DISTANCE else 0.0

    def _update_last_touch(self, kick_connected: list[bool]) -> None:
        for index, connected in enumerate(kick_connected):
            if connected:
                player = self.players[index]
                self.last_touch_team = player.team
                self.last_touch_player = player.label
                return

        touching = [
            player.distance_to(self.ball.x, self.ball.y) <= C.PLAYER_RADIUS + C.BALL_RADIUS + 0.5
            for player in self.players
        ]
        for index, is_touching in enumerate(touching):
            if is_touching:
                player = self.players[index]
                self.last_touch_team = player.team
                self.last_touch_player = player.label
                return

    def _action_to_move(self, action: int) -> tuple[float, float]:
        if action == ACTION_UP:
            return (0.0, -C.PLAYER_SPEED)
        if action == ACTION_DOWN:
            return (0.0, C.PLAYER_SPEED)
        if action == ACTION_LEFT:
            return (-C.PLAYER_SPEED, 0.0)
        if action == ACTION_RIGHT:
            return (C.PLAYER_SPEED, 0.0)
        return (0.0, 0.0)

    def _get_state(self) -> list[float]:
        state = []
        for player in self.players:
            state.extend([player.x, player.y])
        state.extend(
            [
                self.ball.x,
                self.ball.y,
                self.ball.vx,
                self.ball.vy,
                float(self.last_touch_team or 0),
                float(self.match.steps),
            ]
        )
        return state

    def _init_pygame(self) -> None:
        if not pygame.get_init():
            pygame.init()
        self.screen = pygame.display.set_mode((C.SCREEN_WIDTH, C.SCREEN_HEIGHT))
        pygame.display.set_caption("RLStriker - 2v2 Team Soccer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 42)

    def _draw_ui(self) -> None:
        assert self.screen is not None
        assert self.font is not None
        assert self.big_font is not None

        score_text = (
            f"2v2  Team 1: {self.match.score_team_1} - Team 2: {self.match.score_team_2}"
            f"  |  Step: {self.match.steps}/{C.MAX_STEPS}"
            f"  |  Last touch: {self.last_touch_player or 'none'}"
        )
        score_surf = self.font.render(score_text, True, C.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (C.FIELD_LEFT, 10))

        if self.match.done:
            overlay = pygame.Surface((C.SCREEN_WIDTH, C.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(C.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            msg = f"Goal for Team {self.match.last_scorer}" if self.match.last_scorer else self.match.end_reason
            title = self.big_font.render(msg, True, (255, 220, 80))
            hint = self.font.render("Episode done", True, C.COLOR_UI_TEXT)
            self.screen.blit(title, title.get_rect(center=(C.SCREEN_WIDTH // 2, C.SCREEN_HEIGHT // 2 - 20)))
            self.screen.blit(hint, hint.get_rect(center=(C.SCREEN_WIDTH // 2, C.SCREEN_HEIGHT // 2 + 20)))


def team_state_size() -> int:
    env = TeamSoccerEnv(render_mode=None)
    try:
        return len(env.reset())
    finally:
        env.close()
