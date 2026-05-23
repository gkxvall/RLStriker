"""Pygame overlay for demo and debug views."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from env import constants as C


@dataclass
class DemoOverlayData:
    episode: int
    reward_1: float
    reward_2: float
    last_event: str
    model_name: str
    epsilon: float
    fps: float


class DebugOverlay:
    """Draws compact runtime information over the soccer pitch."""

    def __init__(self) -> None:
        self.font = pygame.font.Font(None, 22)
        self.title_font = pygame.font.Font(None, 26)

    def draw(self, surface: pygame.Surface, env: object, data: DemoOverlayData) -> None:
        panel = pygame.Rect(C.SCREEN_WIDTH - 310, 54, 292, 188)
        overlay = pygame.Surface(panel.size, pygame.SRCALPHA)
        overlay.fill((18, 18, 18, 205))
        surface.blit(overlay, panel.topleft)
        pygame.draw.rect(surface, (245, 245, 245), panel, 1, border_radius=6)

        score = getattr(env, "match").score_team_1, getattr(env, "match").score_team_2
        steps = getattr(env, "match").steps
        lines = [
            ("RLStriker Demo", self.title_font, (255, 220, 90)),
            (f"Model: {data.model_name}", self.font, C.COLOR_UI_TEXT),
            (f"Episode: {data.episode}  Step: {steps}/{C.MAX_STEPS}", self.font, C.COLOR_UI_TEXT),
            (f"Score: P1 {score[0]} - P2 {score[1]}", self.font, C.COLOR_UI_TEXT),
            (f"Reward: P1 {data.reward_1:+.1f}  P2 {data.reward_2:+.1f}", self.font, C.COLOR_UI_TEXT),
            (f"Last event: {data.last_event or 'none'}", self.font, C.COLOR_UI_TEXT),
            (f"Epsilon: {data.epsilon:.3f}  FPS: {data.fps:.0f}", self.font, C.COLOR_UI_TEXT),
            ("R reset  |  Esc quit", self.font, (190, 210, 255)),
        ]

        y = panel.y + 12
        for text, font, color in lines:
            rendered = font.render(text, True, color)
            surface.blit(rendered, (panel.x + 12, y))
            y += 22
