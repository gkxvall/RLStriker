"""Human vs AI mode for RLStriker V14."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pygame

from agents.checkpoint_opponent import CheckpointOpponent
from agents.random_agent import RandomAgent
from env import constants as C
from env.entities import Ball, Player
from env.soccer_env import (
    ACTION_DOWN,
    ACTION_KICK,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
    SoccerEnv,
)
from env.state import build_state
from visual.menu import DifficultyMenu, MenuOption


WEAK_CHECKPOINT = Path("data/training_runs/agent2/checkpoints/final.pt")
STRONG_CHECKPOINT = Path("data/training_runs/alpha/checkpoints/final.pt")

DIFFICULTY_OPTIONS = [
    MenuOption("random", "Random AI", "Baseline opponent with random actions."),
    MenuOption("weak", "Weak trained model", "Uses the early V10 trained checkpoint when available."),
    MenuOption("strong", "Strong trained model", "Uses the alpha curriculum checkpoint when available."),
    MenuOption("latest", "Latest model", "Uses the newest compatible non-smoke checkpoint."),
]


@dataclass
class AIPolicy:
    key: str
    label: str
    random_agent: RandomAgent
    checkpoint: CheckpointOpponent | None = None

    def act(self, state: list[float], controlling_agent: int) -> int:
        if self.checkpoint is None:
            return self.random_agent.act()
        return _checkpoint_action(self.checkpoint, state, controlling_agent)


def _human_action() -> int:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        return ACTION_KICK
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return ACTION_LEFT
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return ACTION_RIGHT
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return ACTION_UP
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        return ACTION_DOWN
    return ACTION_STAY


def _checkpoint_action(policy: CheckpointOpponent, state: list[float], controlling_agent: int) -> int:
    trained_agent = int(policy.metadata.get("train_agent", controlling_agent))
    if trained_agent != controlling_agent:
        return _flip_horizontal_action(policy.act(_mirror_swapped_state(state)))
    return policy.act(state)


def _mirror_swapped_state(state: list[float]) -> list[float]:
    p1_x, p1_y = state[0], state[1]
    p2_x, p2_y = state[2], state[3]
    ball_x, ball_y = state[4], state[5]
    ball_vx, ball_vy = state[6], state[7]
    last_touch_owner = int(state[16]) if len(state) > 16 else 0

    mirror_sum = C.FIELD_LEFT + C.FIELD_RIGHT
    transformed_touch = {0: None, 1: 2, 2: 1}.get(last_touch_owner, None)

    player_1 = Player(mirror_sum - p2_x, p2_y, C.COLOR_PLAYER_1, "1", team=1)
    player_2 = Player(mirror_sum - p1_x, p1_y, C.COLOR_PLAYER_2, "2", team=2)
    ball = Ball(mirror_sum - ball_x, ball_y)
    ball.vx = -ball_vx
    ball.vy = ball_vy
    return build_state(player_1, player_2, ball, steps=int(state[-1]), last_touch_owner=transformed_touch).to_list()


def _flip_horizontal_action(action: int) -> int:
    if action == ACTION_LEFT:
        return ACTION_RIGHT
    if action == ACTION_RIGHT:
        return ACTION_LEFT
    return action


def _resolve_latest_checkpoint() -> Path | None:
    candidates = []
    for path in Path("data/training_runs").glob("*/checkpoints/final.pt"):
        run_name = path.parents[1].name
        if run_name.startswith("codex_") or run_name == "test1":
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _checkpoint_for_difficulty(difficulty: str, override: str | None) -> Path | None:
    if override:
        return Path(override)
    if difficulty == "weak":
        return WEAK_CHECKPOINT
    if difficulty == "strong":
        return STRONG_CHECKPOINT
    if difficulty == "latest":
        return _resolve_latest_checkpoint()
    return None


def _load_ai_policy(difficulty: str, env_state_size: int, checkpoint_override: str | None = None) -> AIPolicy:
    random_agent = RandomAgent()
    label = next((option.label for option in DIFFICULTY_OPTIONS if option.key == difficulty), "Random AI")
    checkpoint_path = _checkpoint_for_difficulty(difficulty, checkpoint_override)
    if checkpoint_path is None:
        return AIPolicy(difficulty, label, random_agent)

    try:
        checkpoint = CheckpointOpponent(checkpoint_path)
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as exc:
        print(f"Could not load {checkpoint_path}; falling back to random AI: {exc}")
        return AIPolicy("random", "Random AI", random_agent)

    if checkpoint.state_size != env_state_size:
        print(
            f"Checkpoint {checkpoint_path} expects state size {checkpoint.state_size}, "
            f"but env uses {env_state_size}; falling back to random AI."
        )
        return AIPolicy("random", "Random AI", random_agent)

    return AIPolicy(difficulty, f"{label}: {checkpoint_path.parent.parent.name}", random_agent, checkpoint)


def _draw_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    env: SoccerEnv,
    ai_policy: AIPolicy,
    human_agent: int,
) -> None:
    text = (
        f"Human: Agent {human_agent}  |  AI: {ai_policy.label}  |  "
        f"Score P1 {env.match.score_team_1} - P2 {env.match.score_team_2}  |  R restart  M menu  Esc quit"
    )
    bg = pygame.Surface((C.SCREEN_WIDTH, 32), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 145))
    surface.blit(bg, (0, C.SCREEN_HEIGHT - 32))
    rendered = font.render(text, True, C.COLOR_UI_TEXT)
    surface.blit(rendered, (C.FIELD_LEFT, C.SCREEN_HEIGHT - 24))


def _select_difficulty(screen: pygame.Surface, clock: pygame.time.Clock) -> str | None:
    menu = DifficultyMenu(DIFFICULTY_OPTIONS)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None
            selected = menu.handle_event(event)
            if selected:
                return selected
        menu.draw(screen)
        pygame.display.flip()
        clock.tick(C.FPS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play RLStriker as a human against AI.")
    parser.add_argument(
        "--difficulty",
        choices=[option.key for option in DIFFICULTY_OPTIONS],
        default=None,
        help="Skip the menu and start with this difficulty.",
    )
    parser.add_argument("--human-agent", type=int, choices=(1, 2), default=1, help="Side controlled by the human.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint for trained difficulties.")
    parser.add_argument("--max-rendered-steps", type=int, default=None, help="Optional cap for automated smoke tests.")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((C.SCREEN_WIDTH, C.SCREEN_HEIGHT))
    pygame.display.set_caption("RLStriker - Human vs AI")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    difficulty = args.difficulty or _select_difficulty(screen, clock)
    if difficulty is None:
        pygame.quit()
        return

    env = SoccerEnv(render_mode="human")
    state = env.reset()
    ai_agent_number = 2 if args.human_agent == 1 else 1
    ai_policy = _load_ai_policy(difficulty, len(state), args.checkpoint)

    running = True
    rendered_steps = 0
    while running:
        if args.max_rendered_steps is not None and rendered_steps >= args.max_rendered_steps:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = env.reset()
                elif event.key == pygame.K_m:
                    chosen = _select_difficulty(screen, clock)
                    if chosen is not None:
                        difficulty = chosen
                        ai_policy = _load_ai_policy(difficulty, len(state), args.checkpoint)
                        state = env.reset()

        if not running:
            break

        if env.match.done:
            env.render()
        else:
            human_action = _human_action()
            ai_action = ai_policy.act(state, ai_agent_number)
            if args.human_agent == 1:
                action_1, action_2 = human_action, ai_action
            else:
                action_1, action_2 = ai_action, human_action
            state, _, _, _, _ = env.step(action_1, action_2)
            env.render()

        current_surface = pygame.display.get_surface()
        if current_surface is not None:
            _draw_hud(current_surface, font, env, ai_policy, args.human_agent)
            pygame.display.flip()
        if env.clock is not None:
            env.clock.tick(C.FPS)
        rendered_steps += 1

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
