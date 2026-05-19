"""Game entities: players and ball."""

from __future__ import annotations

import math

import pygame

from env import constants as C


class Player:
    def __init__(
        self,
        x: float,
        y: float,
        color: tuple[int, int, int],
        label: str,
        team: int,
    ) -> None:
        self.x = x
        self.y = y
        self.color = color
        self.label = label
        self.team = team
        self.last_dx = 1.0 if team == 1 else -1.0
        self.last_dy = 0.0

    def move(self, dx: float, dy: float) -> None:
        if dx or dy:
            length = math.hypot(dx, dy)
            self.last_dx = dx / length
            self.last_dy = dy / length
        self.x += dx
        self.y += dy
        self._clamp_to_field()

    def reset(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.last_dx = 1.0 if self.team == 1 else -1.0
        self.last_dy = 0.0

    def facing(self) -> tuple[float, float]:
        return self.last_dx, self.last_dy

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)

    def _clamp_to_field(self) -> None:
        min_x = C.FIELD_LEFT + C.PLAYER_RADIUS
        max_x = C.FIELD_RIGHT - C.PLAYER_RADIUS
        min_y = C.FIELD_TOP + C.PLAYER_RADIUS
        max_y = C.FIELD_BOTTOM - C.PLAYER_RADIUS
        self.x = max(min_x, min(max_x, self.x))
        self.y = max(min_y, min(max_y, self.y))

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), C.PLAYER_RADIUS)
        pygame.draw.circle(surface, (20, 20, 20), (int(self.x), int(self.y)), C.PLAYER_RADIUS, 2)
        font = pygame.font.Font(None, 22)
        text = font.render(self.label, True, (255, 255, 255))
        surface.blit(text, text.get_rect(center=(int(self.x), int(self.y))))


class Ball:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0

    def reset(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0

    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, C.COLOR_BALL, (int(self.x), int(self.y)), C.BALL_RADIUS)
        pygame.draw.circle(surface, (30, 30, 30), (int(self.x), int(self.y)), C.BALL_RADIUS, 2)
