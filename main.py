"""Manual RLStriker app powered by SoccerEnv."""

from __future__ import annotations

import sys

import pygame

from env import constants as C
from env.soccer_env import (
    ACTION_DOWN,
    ACTION_KICK,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
    SoccerEnv,
)


def get_player_actions() -> tuple[int, int]:
    keys = pygame.key.get_pressed()

    action_1 = ACTION_STAY
    action_2 = ACTION_STAY

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        action_1 = ACTION_LEFT
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        action_1 = ACTION_RIGHT
    elif keys[pygame.K_UP] or keys[pygame.K_w]:
        action_1 = ACTION_UP
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        action_1 = ACTION_DOWN

    if keys[pygame.K_j]:
        action_2 = ACTION_LEFT
    elif keys[pygame.K_l]:
        action_2 = ACTION_RIGHT
    elif keys[pygame.K_i]:
        action_2 = ACTION_UP
    elif keys[pygame.K_k]:
        action_2 = ACTION_DOWN

    if keys[pygame.K_SPACE]:
        action_1 = ACTION_KICK
    if keys[pygame.K_RETURN]:
        action_2 = ACTION_KICK
    return action_1, action_2


def draw_reset_button(surface: pygame.Surface, font: pygame.font.Font, reset_hover: bool) -> pygame.Rect:
    btn_color = C.COLOR_RESET_BTN_HOVER if reset_hover else C.COLOR_RESET_BTN
    reset_rect = pygame.Rect(C.RESET_BUTTON_RECT)
    pygame.draw.rect(surface, btn_color, reset_rect, border_radius=6)
    pygame.draw.rect(surface, C.COLOR_UI_TEXT, reset_rect, 2, border_radius=6)
    label = font.render("Reset", True, C.COLOR_UI_TEXT)
    surface.blit(label, label.get_rect(center=reset_rect.center))
    return reset_rect


def main() -> None:
    pygame.init()  # keep init here for button events; env owns rendering surface
    font = pygame.font.Font(None, 24)
    env = SoccerEnv(render_mode="human")
    env.reset()
    reset_rect = pygame.Rect(C.RESET_BUTTON_RECT)
    reset_hover = False

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        reset_hover = reset_rect.collidepoint(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if reset_rect.collidepoint(event.pos):
                    env.reset()

        action_1, action_2 = get_player_actions()
        env.step(action_1, action_2)
        env.render()

        screen = pygame.display.get_surface()
        if screen is not None and env.clock is not None:
            reset_rect = draw_reset_button(screen, font, reset_hover)
            pygame.display.flip()
            env.clock.tick(C.FPS)

    env.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
