"""RLStriker V2 — physics, goals, scoring, and episode timer."""

from __future__ import annotations

import sys

import pygame

from env import constants as C
from env.entities import Ball, Player
from env.field import Field
from env.physics import kick_ball, update_physics
from env.rules import MatchState


def get_player_input() -> tuple[tuple[float, float], tuple[float, float], bool, bool]:
    keys = pygame.key.get_pressed()

    p1_dx = p1_dy = 0.0
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        p1_dx -= C.PLAYER_SPEED
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        p1_dx += C.PLAYER_SPEED
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        p1_dy -= C.PLAYER_SPEED
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        p1_dy += C.PLAYER_SPEED

    p2_dx = p2_dy = 0.0
    if keys[pygame.K_j]:
        p2_dx -= C.PLAYER_SPEED
    if keys[pygame.K_l]:
        p2_dx += C.PLAYER_SPEED
    if keys[pygame.K_i]:
        p2_dy -= C.PLAYER_SPEED
    if keys[pygame.K_k]:
        p2_dy += C.PLAYER_SPEED

    kick_p1 = keys[pygame.K_SPACE]
    kick_p2 = keys[pygame.K_RETURN]
    return (p1_dx, p1_dy), (p2_dx, p2_dy), kick_p1, kick_p2


def draw_ui(
    surface: pygame.Surface,
    font: pygame.font.Font,
    big_font: pygame.font.Font,
    match: MatchState,
    reset_hover: bool,
) -> pygame.Rect:
    help_surf = font.render(C.HELP_TEXT, True, C.COLOR_UI_TEXT)
    surface.blit(help_surf, (C.FIELD_LEFT, 8))

    score_text = f"Score  P1: {match.score_team_1}  —  P2: {match.score_team_2}  |  Step: {match.steps}/{C.MAX_STEPS}"
    score_surf = font.render(score_text, True, C.COLOR_UI_TEXT)
    surface.blit(score_surf, (C.FIELD_LEFT, 28))

    btn_color = C.COLOR_RESET_BTN_HOVER if reset_hover else C.COLOR_RESET_BTN
    reset_rect = pygame.Rect(C.RESET_BUTTON_RECT)
    pygame.draw.rect(surface, btn_color, reset_rect, border_radius=6)
    pygame.draw.rect(surface, C.COLOR_UI_TEXT, reset_rect, 2, border_radius=6)
    label = font.render("Reset", True, C.COLOR_UI_TEXT)
    surface.blit(label, label.get_rect(center=reset_rect.center))

    if match.done:
        overlay = pygame.Surface((C.SCREEN_WIDTH, C.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill(C.COLOR_OVERLAY)
        surface.blit(overlay, (0, 0))

        if match.last_scorer:
            msg = f"GOAL! Team {match.last_scorer} scores"
        else:
            msg = match.end_reason
        title = big_font.render(msg, True, (255, 220, 80))
        hint = font.render("Press R or Reset for a new episode", True, C.COLOR_UI_TEXT)
        surface.blit(title, title.get_rect(center=(C.SCREEN_WIDTH // 2, C.SCREEN_HEIGHT // 2 - 20)))
        surface.blit(hint, hint.get_rect(center=(C.SCREEN_WIDTH // 2, C.SCREEN_HEIGHT // 2 + 20)))

    return reset_rect


def start_episode(field: Field, match: MatchState) -> tuple[Player, Player, Ball]:
    match.new_episode()
    return field.reset_entities()


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((C.SCREEN_WIDTH, C.SCREEN_HEIGHT))
    pygame.display.set_caption(C.TITLE)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    big_font = pygame.font.Font(None, 42)

    field = Field()
    match = MatchState()
    player_1, player_2, ball = start_episode(field, match)
    players = [player_1, player_2]
    reset_rect = pygame.Rect(C.RESET_BUTTON_RECT)
    reset_hover = False

    kick_p1_held = kick_p2_held = False

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        reset_hover = reset_rect.collidepoint(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    player_1, player_2, ball = start_episode(field, match)
                    players = [player_1, player_2]
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if reset_rect.collidepoint(event.pos):
                    player_1, player_2, ball = start_episode(field, match)
                    players = [player_1, player_2]

        if not match.done:
            p1_move, p2_move, kick_p1, kick_p2 = get_player_input()
            moves = [p1_move, p2_move]

            if kick_p1 and not kick_p1_held:
                kick_ball(player_1, ball)
            if kick_p2 and not kick_p2_held:
                kick_ball(player_2, ball)
            kick_p1_held = kick_p1
            kick_p2_held = kick_p2

            update_physics(players, ball, moves)
            match.step_after_physics(ball)

        screen.fill((20, 20, 20))
        field.draw(screen)
        player_1.draw(screen)
        player_2.draw(screen)
        ball.draw(screen)
        reset_rect = draw_ui(screen, font, big_font, match, reset_hover)

        pygame.display.flip()
        clock.tick(C.FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
