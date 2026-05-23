"""Watch trained RLStriker checkpoints play in a Pygame window."""

from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from agents.checkpoint_opponent import CheckpointOpponent
from agents.random_agent import RandomAgent
from env import constants as C
from env.entities import Ball, Player
from env.soccer_env import ACTION_LEFT, ACTION_RIGHT, SoccerEnv
from env.state import build_state
from visual.debug_overlay import DebugOverlay, DemoOverlayData


DEFAULT_CHECKPOINT = Path("data/training_runs/newAgent/checkpoints/final.pt")


def _checkpoint_path(value: str | None) -> Path:
    if value:
        return Path(value)
    return DEFAULT_CHECKPOINT


def _policy_action(policy: CheckpointOpponent, state: list[float], controlling_agent: int) -> int:
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

    return build_state(
        player_1,
        player_2,
        ball,
        steps=int(state[-1]),
        last_touch_owner=transformed_touch,
    ).to_list()


def _flip_horizontal_action(action: int) -> int:
    if action == ACTION_LEFT:
        return ACTION_RIGHT
    if action == ACTION_RIGHT:
        return ACTION_LEFT
    return action


def _event_from_info(info: dict, done: bool) -> str:
    if info.get("goal"):
        return f"goal_agent_{info.get('scorer')}"
    if info.get("max_steps"):
        return "max_steps"
    if done:
        return "done"
    if info.get("kick_connected_1"):
        return "agent_1_kick"
    if info.get("kick_connected_2"):
        return "agent_2_kick"
    return ""


def _validate_policy(policy: CheckpointOpponent, env_state_size: int) -> None:
    if policy.state_size != env_state_size:
        raise ValueError(
            f"{policy.path} expects state size {policy.state_size}, "
            f"but the current environment returns {env_state_size}. "
            "Use a V10+ checkpoint trained with the 18-value state."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a trained RLStriker checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"Model checkpoint to watch. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument("--model-agent", type=int, choices=(1, 2), default=1, help="Side controlled by the model.")
    parser.add_argument(
        "--opponent",
        choices=("random", "checkpoint"),
        default="random",
        help="Opponent policy.",
    )
    parser.add_argument("--opponent-checkpoint", type=str, default=None, help="Checkpoint for checkpoint opponent.")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes to watch before exiting.")
    parser.add_argument("--fps", type=int, default=C.FPS, help="Playback FPS.")
    parser.add_argument("--pause-on-done", type=float, default=1.0, help="Seconds to show terminal state.")
    parser.add_argument(
        "--max-rendered-steps",
        type=int,
        default=None,
        help="Optional cap for automated smoke tests.",
    )
    args = parser.parse_args()

    checkpoint_path = _checkpoint_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    env = SoccerEnv(render_mode="human")
    state = env.reset()
    model = CheckpointOpponent(checkpoint_path)
    _validate_policy(model, len(state))

    random_opponent = RandomAgent()
    checkpoint_opponent: CheckpointOpponent | None = None
    if args.opponent == "checkpoint":
        if not args.opponent_checkpoint:
            raise ValueError("--opponent-checkpoint is required when --opponent checkpoint")
        checkpoint_opponent = CheckpointOpponent(args.opponent_checkpoint)
        _validate_policy(checkpoint_opponent, len(state))

    overlay = DebugOverlay()
    model_name = checkpoint_path.parent.parent.name if checkpoint_path.name == "final.pt" else checkpoint_path.stem
    episode = 1
    total_reward_1 = 0.0
    total_reward_2 = 0.0
    last_event = "start"
    done_pause_frames = 0
    done_pause_target = max(1, int(args.pause_on_done * args.fps))
    rendered_steps = 0

    running = True
    while running and episode <= args.episodes:
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
                    total_reward_1 = 0.0
                    total_reward_2 = 0.0
                    last_event = "manual_reset"
                    done_pause_frames = 0

        if not running:
            break

        if env.match.done:
            done_pause_frames += 1
            if done_pause_frames >= done_pause_target:
                episode += 1
                if episode > args.episodes:
                    break
                state = env.reset()
                total_reward_1 = 0.0
                total_reward_2 = 0.0
                last_event = "new_episode"
                done_pause_frames = 0
        else:
            model_action = _policy_action(model, state, args.model_agent)
            if args.opponent == "checkpoint":
                assert checkpoint_opponent is not None
                opponent_agent = 2 if args.model_agent == 1 else 1
                opponent_action = _policy_action(checkpoint_opponent, state, opponent_agent)
            else:
                opponent_action = random_opponent.act()

            if args.model_agent == 1:
                action_1, action_2 = model_action, opponent_action
            else:
                action_1, action_2 = opponent_action, model_action

            state, reward_1, reward_2, done, info = env.step(action_1, action_2)
            total_reward_1 += reward_1
            total_reward_2 += reward_2
            event_name = _event_from_info(info, done)
            if event_name:
                last_event = event_name

        env.render()
        screen = pygame.display.get_surface()
        if screen is not None:
            fps = env.clock.get_fps() if env.clock is not None else float(args.fps)
            overlay.draw(
                screen,
                env,
                DemoOverlayData(
                    episode=episode,
                    reward_1=total_reward_1,
                    reward_2=total_reward_2,
                    last_event=last_event,
                    model_name=model_name,
                    epsilon=model.epsilon,
                    fps=fps,
                ),
            )
            pygame.display.flip()

        if env.clock is not None:
            env.clock.tick(args.fps)
        rendered_steps += 1

    env.close()


if __name__ == "__main__":
    main()
