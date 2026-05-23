"""Baseline team agents for RLStriker 2v2 mode."""

from __future__ import annotations

import random

from env.soccer_env import ACTION_DOWN, ACTION_KICK, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE_SIZE, ACTION_UP


class TeamRandomAgent:
    """Selects one random action for each player on a two-player team."""

    def act(self, state: list[float] | None = None) -> list[int]:
        return [random.randrange(ACTION_SPACE_SIZE), random.randrange(ACTION_SPACE_SIZE)]


class TeamRoleAgent:
    """
    Tiny non-learning baseline with attacker/defender roles.

    Player 0 tends to chase and kick the ball. Player 1 stays more conservative
    by drifting toward its own half unless the ball comes close.
    """

    def __init__(self, team: int) -> None:
        self.team = team

    def act(self, state: list[float]) -> list[int]:
        players = _team_positions(state, self.team)
        ball = (state[8], state[9])
        attacker_action = _move_toward(players[0], ball, kick_when_close=True)

        defender_anchor_x = 260.0 if self.team == 1 else 700.0
        defender_target = (defender_anchor_x, ball[1])
        defender_action = _move_toward(players[1], defender_target, kick_when_close=True, ball=ball)
        return [attacker_action, defender_action]


def _team_positions(state: list[float], team: int) -> list[tuple[float, float]]:
    if team == 1:
        return [(state[0], state[1]), (state[2], state[3])]
    return [(state[4], state[5]), (state[6], state[7])]


def _move_toward(
    pos: tuple[float, float],
    target: tuple[float, float],
    *,
    kick_when_close: bool = False,
    ball: tuple[float, float] | None = None,
) -> int:
    kick_target = ball or target
    if kick_when_close and abs(pos[0] - kick_target[0]) < 28 and abs(pos[1] - kick_target[1]) < 28:
        return ACTION_KICK

    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    if abs(dx) > abs(dy):
        return ACTION_RIGHT if dx > 0 else ACTION_LEFT
    return ACTION_DOWN if dy > 0 else ACTION_UP
