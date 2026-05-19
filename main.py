"""NeuroSoccer V1 — visual pitch with manual player control."""

from __future__ import annotations

import sys

import pygame

from env import constants as C
from env.entities import Ball, Player
from env.field import Field


def get_movement() -> tuple[float, float]:
  keys = pygame.key.get_pressed()
  dx = dy = 0.0
  if keys[pygame.K_LEFT] or keys[pygame.K_a]:
    dx -= C.PLAYER_SPEED
  if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
    dx += C.PLAYER_SPEED
  if keys[pygame.K_UP] or keys[pygame.K_w]:
    dy -= C.PLAYER_SPEED
  if keys[pygame.K_DOWN] or keys[pygame.K_s]:
    dy += C.PLAYER_SPEED
  return dx, dy


def draw_ui(surface: pygame.Surface, font: pygame.font.Font, reset_hover: bool) -> pygame.Rect:
  help_surf = font.render(C.HELP_TEXT, True, C.COLOR_UI_TEXT)
  surface.blit(help_surf, (C.FIELD_LEFT, 8))

  btn_color = C.COLOR_RESET_BTN_HOVER if reset_hover else C.COLOR_RESET_BTN
  reset_rect = pygame.Rect(C.RESET_BUTTON_RECT)
  pygame.draw.rect(surface, btn_color, reset_rect, border_radius=6)
  pygame.draw.rect(surface, C.COLOR_UI_TEXT, reset_rect, 2, border_radius=6)
  label = font.render("Reset", True, C.COLOR_UI_TEXT)
  surface.blit(label, label.get_rect(center=reset_rect.center))
  return reset_rect


def reset_game(field: Field) -> tuple[Player, Player, Ball]:
  return field.reset_entities()


def main() -> None:
  pygame.init()
  screen = pygame.display.set_mode((C.SCREEN_WIDTH, C.SCREEN_HEIGHT))
  pygame.display.set_caption(C.TITLE)
  clock = pygame.time.Clock()
  font = pygame.font.Font(None, 24)

  field = Field()
  player_1, player_2, ball = reset_game(field)
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
          player_1, player_2, ball = reset_game(field)
      elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if reset_rect.collidepoint(event.pos):
          player_1, player_2, ball = reset_game(field)

    dx, dy = get_movement()
    if dx or dy:
      player_1.move(dx, dy)

    screen.fill((20, 20, 20))
    field.draw(screen)
    player_1.draw(screen)
    player_2.draw(screen)
    ball.draw(screen)
    reset_rect = draw_ui(screen, font, reset_hover)

    pygame.display.flip()
    clock.tick(C.FPS)

  pygame.quit()
  sys.exit(0)


if __name__ == "__main__":
  main()
