"""Ball and player physics."""

from __future__ import annotations

import math

from env import constants as C
from env.entities import Ball, Player


def _clamp_ball_speed(ball: Ball) -> None:
    speed = ball.speed()
    if speed > C.MAX_BALL_SPEED:
        scale = C.MAX_BALL_SPEED / speed
        ball.vx *= scale
        ball.vy *= scale


def apply_friction(ball: Ball) -> None:
    ball.vx *= C.BALL_FRICTION
    ball.vy *= C.BALL_FRICTION
    if ball.speed() < 0.05:
        ball.vx = 0.0
        ball.vy = 0.0


def move_ball(ball: Ball) -> None:
    ball.x += ball.vx
    ball.y += ball.vy
    _keep_ball_in_field(ball)
    _clamp_ball_speed(ball)


def _keep_ball_in_field(ball: Ball) -> None:
    min_x = C.FIELD_LEFT + C.BALL_RADIUS
    max_x = C.FIELD_RIGHT - C.BALL_RADIUS
    min_y = C.FIELD_TOP + C.BALL_RADIUS
    max_y = C.FIELD_BOTTOM - C.BALL_RADIUS

    if ball.x < min_x:
        ball.x = min_x
        ball.vx = abs(ball.vx) * C.BALL_BOUNCE
    elif ball.x > max_x:
        ball.x = max_x
        ball.vx = -abs(ball.vx) * C.BALL_BOUNCE

    if ball.y < min_y:
        ball.y = min_y
        ball.vy = abs(ball.vy) * C.BALL_BOUNCE
    elif ball.y > max_y:
        ball.y = max_y
        ball.vy = -abs(ball.vy) * C.BALL_BOUNCE


def collide_player_ball(player: Player, ball: Ball, move_dx: float, move_dy: float) -> None:
    dist = player.distance_to(ball.x, ball.y)
    min_dist = C.PLAYER_RADIUS + C.BALL_RADIUS
    if dist >= min_dist or dist == 0:
        return

    nx = (ball.x - player.x) / dist
    ny = (ball.y - player.y) / dist

    overlap = min_dist - dist
    ball.x += nx * overlap
    ball.y += ny * overlap

    # Push from player movement
    ball.vx += move_dx * C.PUSH_POWER
    ball.vy += move_dy * C.PUSH_POWER
    _clamp_ball_speed(ball)


def kick_ball(player: Player, ball: Ball) -> bool:
    if player.distance_to(ball.x, ball.y) > C.KICK_RANGE:
        return False

    fx, fy = player.facing()
    ball.vx += fx * C.KICK_POWER
    ball.vy += fy * C.KICK_POWER
    _clamp_ball_speed(ball)
    return True


def update_physics(players: list[Player], ball: Ball, player_moves: list[tuple[float, float]]) -> None:
    for player, (dx, dy) in zip(players, player_moves):
        if dx or dy:
            player.move(dx, dy)
        collide_player_ball(player, ball, dx, dy)

    apply_friction(ball)
    move_ball(ball)
