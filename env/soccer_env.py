"""Gym-style soccer environment for RLStriker V3."""

from __future__ import annotations

from typing import Any

import pygame

from env import constants as C
from env.field import Field
from env.physics import kick_ball, update_physics
from env.rules import MatchState
from env.state import build_state

# Discrete action space used by step(action_1, action_2)
ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_KICK = 5
ACTION_SPACE_SIZE = 6


class SoccerEnv:
    """Small 1v1 soccer environment with optional rendering."""

    def __init__(self, render_mode: str | None = None) -> None:
        self.render_mode = render_mode
        self.field = Field()
        self.match = MatchState()

        self.player_1, self.player_2, self.ball = self.field.reset_entities()
        self.players = [self.player_1, self.player_2]

        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.big_font: pygame.font.Font | None = None

        if self.render_mode == "human":
            self._init_pygame()

    def reset(self) -> list[float]:
        self.match.new_episode()
        self.player_1, self.player_2, self.ball = self.field.reset_entities()
        self.players = [self.player_1, self.player_2]
        return self._get_state()

    def step(self, action_1: int, action_2: int) -> tuple[list[float], float, float, bool, dict[str, Any]]:
        if self.match.done:
            return self._get_state(), 0.0, 0.0, True, {"note": "Episode is done. Call reset()."}

        move_1 = self._action_to_move(action_1)
        move_2 = self._action_to_move(action_2)

        if action_1 == ACTION_KICK:
            kick_ball(self.player_1, self.ball)
        if action_2 == ACTION_KICK:
            kick_ball(self.player_2, self.ball)

        update_physics(self.players, self.ball, [move_1, move_2])
        info = self.match.step_after_physics(self.ball)

        reward_1, reward_2 = self._compute_rewards(info)
        state = self._get_state()
        done = self.match.done

        return state, reward_1, reward_2, done, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self.screen is None:
            self._init_pygame()

        assert self.screen is not None
        assert self.font is not None
        assert self.big_font is not None
        assert self.clock is not None

        self.screen.fill((20, 20, 20))
        self.field.draw(self.screen)
        self.player_1.draw(self.screen)
        self.player_2.draw(self.screen)
        self.ball.draw(self.screen)

        self._draw_ui()

    def close(self) -> None:
        if self.render_mode == "human":
            pygame.quit()
            self.screen = None

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

    def _compute_rewards(self, info: dict[str, Any]) -> tuple[float, float]:
        scorer = info.get("scorer")
        if scorer == 1:
            return 1.0, -1.0
        if scorer == 2:
            return -1.0, 1.0
        return 0.0, 0.0

    def _get_state(self) -> list[float]:
        state = build_state(self.player_1, self.player_2, self.ball, self.match.steps)
        return state.to_list()

    def _init_pygame(self) -> None:
        if not pygame.get_init():
            pygame.init()
        self.screen = pygame.display.set_mode((C.SCREEN_WIDTH, C.SCREEN_HEIGHT))
        pygame.display.set_caption("RLStriker - SoccerEnv")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 42)

    def _draw_ui(self) -> None:
        assert self.screen is not None
        assert self.font is not None
        assert self.big_font is not None

        help_surf = self.font.render(C.HELP_TEXT, True, C.COLOR_UI_TEXT)
        self.screen.blit(help_surf, (C.FIELD_LEFT, 8))

        score_text = (
            f"Score  P1: {self.match.score_team_1}  -  P2: {self.match.score_team_2}"
            f"  |  Step: {self.match.steps}/{C.MAX_STEPS}"
        )
        score_surf = self.font.render(score_text, True, C.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (C.FIELD_LEFT, 28))

        if self.match.done:
            overlay = pygame.Surface((C.SCREEN_WIDTH, C.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(C.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))

            if self.match.last_scorer:
                msg = f"GOAL! Team {self.match.last_scorer} scores"
            else:
                msg = self.match.end_reason
            title = self.big_font.render(msg, True, (255, 220, 80))
            hint = self.font.render("Press R or Reset for a new episode", True, C.COLOR_UI_TEXT)
            self.screen.blit(title, title.get_rect(center=(C.SCREEN_WIDTH // 2, C.SCREEN_HEIGHT // 2 - 20)))
            self.screen.blit(hint, hint.get_rect(center=(C.SCREEN_WIDTH // 2, C.SCREEN_HEIGHT // 2 + 20)))
