"""Game entities: players and ball."""

from __future__ import annotations

import pygame

from env import constants as C


class Player:
    def __init__(self, x: float, y: float, color: tuple[int, int, int], label: str) -> None:
        self.x = x
        self.y = y
        self.color = color
        self.label = label

    def move(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy
        self._clamp_to_field()

    def reset(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

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

    def reset(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, C.COLOR_BALL, (int(self.x), int(self.y)), C.BALL_RADIUS)
        pygame.draw.circle(surface, (30, 30, 30), (int(self.x), int(self.y)), C.BALL_RADIUS, 2)
