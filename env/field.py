"""Pitch rendering and starting positions."""

from __future__ import annotations

import pygame

from env import constants as C
from env.entities import Ball, Player


class Field:
  def __init__(self) -> None:
    self.player_1: Player | None = None
    self.player_2: Player | None = None
    self.ball: Ball | None = None

  def reset_entities(self) -> tuple[Player, Player, Ball]:
    """Place players and ball at default kickoff positions."""
    p1_x = C.FIELD_LEFT + C.FIELD_WIDTH * 0.25
    p2_x = C.FIELD_LEFT + C.FIELD_WIDTH * 0.75
    center_y = C.FIELD_CENTER_Y

    self.player_1 = Player(p1_x, center_y, C.COLOR_PLAYER_1, "1")
    self.player_2 = Player(p2_x, center_y, C.COLOR_PLAYER_2, "2")
    self.ball = Ball(C.FIELD_CENTER_X, C.FIELD_CENTER_Y)
    return self.player_1, self.player_2, self.ball

  def draw(self, surface: pygame.Surface) -> None:
    self._draw_pitch(surface)
    self._draw_goals(surface)

  def _draw_pitch(self, surface: pygame.Surface) -> None:
    field_rect = pygame.Rect(C.FIELD_LEFT, C.FIELD_TOP, C.FIELD_WIDTH, C.FIELD_HEIGHT)
    surface.fill(C.COLOR_PITCH_DARK)

    # Alternating stripes
    stripe_w = C.FIELD_WIDTH // 10
    for i in range(10):
      if i % 2 == 0:
        stripe = pygame.Rect(C.FIELD_LEFT + i * stripe_w, C.FIELD_TOP, stripe_w, C.FIELD_HEIGHT)
        pygame.draw.rect(surface, C.COLOR_PITCH, stripe)

    pygame.draw.rect(surface, C.COLOR_LINE, field_rect, 3)

    # Center line and circle
    pygame.draw.line(
      surface,
      C.COLOR_LINE,
      (C.FIELD_CENTER_X, C.FIELD_TOP),
      (C.FIELD_CENTER_X, C.FIELD_BOTTOM),
      2,
    )
    pygame.draw.circle(surface, C.COLOR_LINE, (C.FIELD_CENTER_X, C.FIELD_CENTER_Y), 60, 2)

    # Center spot
    pygame.draw.circle(surface, C.COLOR_LINE, (C.FIELD_CENTER_X, C.FIELD_CENTER_Y), 5)

    # Penalty areas (simplified)
    box_w = int(C.FIELD_WIDTH * 0.16)
    box_h = int(C.FIELD_HEIGHT * 0.55)
    box_y = C.FIELD_CENTER_Y - box_h // 2

    left_box = pygame.Rect(C.FIELD_LEFT, box_y, box_w, box_h)
    right_box = pygame.Rect(C.FIELD_RIGHT - box_w, box_y, box_w, box_h)
    pygame.draw.rect(surface, C.COLOR_LINE, left_box, 2)
    pygame.draw.rect(surface, C.COLOR_LINE, right_box, 2)

  def _draw_goals(self, surface: pygame.Surface) -> None:
    goal_y = C.FIELD_CENTER_Y - C.GOAL_HEIGHT // 2

    left_goal = pygame.Rect(
      C.FIELD_LEFT - C.GOAL_DEPTH,
      goal_y,
      C.GOAL_DEPTH,
      C.GOAL_HEIGHT,
    )
    right_goal = pygame.Rect(
      C.FIELD_RIGHT,
      goal_y,
      C.GOAL_DEPTH,
      C.GOAL_HEIGHT,
    )

    for goal_rect in (left_goal, right_goal):
      pygame.draw.rect(surface, C.COLOR_GOAL, goal_rect)
      pygame.draw.rect(surface, C.COLOR_GOAL_NET, goal_rect, 2)
      # Simple net lines
      for i in range(1, 4):
        y = goal_rect.top + i * goal_rect.height // 4
        pygame.draw.line(surface, C.COLOR_GOAL_NET, (goal_rect.left, y), (goal_rect.right, y), 1)
