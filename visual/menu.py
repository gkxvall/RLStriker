"""Simple Pygame menu widgets for interactive modes."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from env import constants as C


@dataclass(frozen=True)
class MenuOption:
    key: str
    label: str
    description: str


class DifficultyMenu:
    """Keyboard-driven difficulty selector for human-vs-AI mode."""

    def __init__(self, options: list[MenuOption]) -> None:
        self.options = options
        self.selected_index = 0
        self.title_font = pygame.font.Font(None, 46)
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

    @property
    def selected(self) -> MenuOption:
        return self.options[self.selected_index]

    def handle_event(self, event: pygame.event.Event) -> str | None:
        if event.type != pygame.KEYDOWN:
            return None
        if event.key in (pygame.K_UP, pygame.K_w):
            self.selected_index = (self.selected_index - 1) % len(self.options)
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self.selected_index = (self.selected_index + 1) % len(self.options)
        elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
            return self.selected.key
        elif pygame.K_1 <= event.key <= pygame.K_9:
            index = event.key - pygame.K_1
            if index < len(self.options):
                self.selected_index = index
                return self.selected.key
        return None

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill((22, 32, 42))
        title = self.title_font.render("RLStriker Human vs AI", True, (255, 255, 255))
        hint = self.small_font.render("Use W/S or arrow keys, Enter to select", True, (205, 215, 225))
        surface.blit(title, title.get_rect(center=(C.SCREEN_WIDTH // 2, 105)))
        surface.blit(hint, hint.get_rect(center=(C.SCREEN_WIDTH // 2, 145)))

        start_y = 205
        width = 560
        height = 64
        x = C.SCREEN_WIDTH // 2 - width // 2
        for index, option in enumerate(self.options):
            y = start_y + index * (height + 14)
            selected = index == self.selected_index
            bg = (52, 107, 74) if selected else (38, 48, 58)
            border = (245, 245, 245) if selected else (90, 105, 115)
            rect = pygame.Rect(x, y, width, height)
            pygame.draw.rect(surface, bg, rect, border_radius=8)
            pygame.draw.rect(surface, border, rect, 2, border_radius=8)

            label = self.font.render(f"{index + 1}. {option.label}", True, (255, 255, 255))
            desc = self.small_font.render(option.description, True, (220, 225, 230))
            surface.blit(label, (rect.x + 18, rect.y + 10))
            surface.blit(desc, (rect.x + 18, rect.y + 38))
