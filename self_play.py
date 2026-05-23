"""Train a DQN agent against older checkpoint opponents."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import pygame
import torch

from agents.checkpoint_opponent import CheckpointOpponent
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from env import constants as C
from env.entities import Ball, Player
from env.soccer_env import ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE_SIZE, SoccerEnv
from env.state import build_state
from logging_utils.episode_logger import EpisodeLogger, utc_timestamp
from logging_utils.metrics import EpisodeMetricsTracker


@dataclass
class SelfPlaySummary:
    episode: int
    steps: int
    winner: str
    reward_1: float
    reward_2: float
    avg_loss: float | None
    epsilon: float
    opponent_name: str


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _step_event(info: dict, done: bool) -> str:
    if not done:
        return ""
    if info.get("goal"):
        return "goal"
    if info.get("max_steps"):
        return "max_steps"
    return "done"


def _load_compatible_opponents(paths: list[Path], state_size: int) -> list[CheckpointOpponent]:
    opponents: list[CheckpointOpponent] = []
    for path in paths:
        try:
            opponent = CheckpointOpponent(path)
        except (KeyError, RuntimeError, ValueError) as exc:
            print(f"Skipping checkpoint {path}: {exc}")
            continue
        if opponent.state_size != state_size:
            print(f"Skipping checkpoint {path}: state size {opponent.state_size} != {state_size}")
            continue
        opponents.append(opponent)
    return opponents


def _select_opponent(
    *,
    checkpoint_pool: list[CheckpointOpponent],
    random_agent: RandomAgent,
    random_opponent_prob: float,
) -> CheckpointOpponent | RandomAgent:
    if not checkpoint_pool or random.random() < random_opponent_prob:
        return random_agent
    return random.choice(checkpoint_pool)


def _opponent_name(opponent: CheckpointOpponent | RandomAgent) -> str:
    if isinstance(opponent, CheckpointOpponent):
        return f"checkpoint:{opponent.name}"
    return "random"


def _opponent_action(opponent: CheckpointOpponent | RandomAgent, state: list[float], train_agent: int) -> int:
    if isinstance(opponent, CheckpointOpponent):
        controlling_agent = 2 if train_agent == 1 else 1
        trained_agent = int(opponent.metadata.get("train_agent", train_agent))
        if trained_agent != controlling_agent:
            action = opponent.act(_mirror_swapped_state(state))
            return _flip_horizontal_action(action)
        return opponent.act(state)
    return opponent.act()


def _mirror_swapped_state(state: list[float]) -> list[float]:
    p1_x, p1_y = state[0], state[1]
    p2_x, p2_y = state[2], state[3]
    ball_x, ball_y = state[4], state[5]
    ball_vx, ball_vy = state[6], state[7]
    last_touch_owner = int(state[16]) if len(state) > 16 else 0

    transformed_touch = {0: None, 1: 2, 2: 1}.get(last_touch_owner, None)
    mirror_sum = C.FIELD_LEFT + C.FIELD_RIGHT

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


def save_checkpoint(
    *,
    learner: DQNAgent,
    checkpoint_dir: Path,
    episode: int,
    run_name: str,
    train_agent: int,
    final: bool = False,
) -> Path:
    name = "final.pt" if final else f"episode_{episode:06d}.pt"
    path = checkpoint_dir / name
    learner.save(
        path,
        metadata={
            "run_name": run_name,
            "episode": episode,
            "train_agent": train_agent,
            "opponent": "self_play",
            "reward_version": "v2",
        },
    )
    return path


def run_self_play_episode(
    *,
    env: SoccerEnv,
    learner: DQNAgent,
    opponent: CheckpointOpponent | RandomAgent,
    episode: int,
    logger: EpisodeLogger,
    metrics: EpisodeMetricsTracker,
    log_steps: bool,
    run_name: str,
    train_agent: int,
) -> SelfPlaySummary:
    state = env.reset()
    metrics.reset()

    total_reward_1 = 0.0
    total_reward_2 = 0.0
    losses: list[float] = []
    winner = "draw"
    goals_1 = 0
    goals_2 = 0
    opponent_name = _opponent_name(opponent)

    while True:
        learner_action = learner.act(state, training=True)
        opponent_action = _opponent_action(opponent, state, train_agent)

        if train_agent == 1:
            action_1 = learner_action
            action_2 = opponent_action
        else:
            action_1 = opponent_action
            action_2 = learner_action

        next_state, reward_1, reward_2, done, info = env.step(action_1, action_2)
        train_reward = reward_1 if train_agent == 1 else reward_2
        learner.remember(state, learner_action, train_reward, next_state, done)
        loss = learner.learn()
        if loss is not None:
            losses.append(loss)

        total_reward_1 += reward_1
        total_reward_2 += reward_2
        metrics.on_step(env, info)

        if log_steps:
            p1, p2, ball = env.player_1, env.player_2, env.ball
            logger.log_step(
                episode=episode,
                step=env.match.steps,
                agent_1_x=p1.x,
                agent_1_y=p1.y,
                agent_2_x=p2.x,
                agent_2_y=p2.y,
                ball_x=ball.x,
                ball_y=ball.y,
                ball_vx=ball.vx,
                ball_vy=ball.vy,
                action_agent_1=action_1,
                action_agent_2=action_2,
                reward_agent_1=reward_1,
                reward_agent_2=reward_2,
                event=_step_event(info, done),
            )

        if env.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            env.render()
            pygame.display.flip()
            if env.clock is not None:
                env.clock.tick(C.FPS)

        state = next_state
        if done:
            scorer = info.get("scorer")
            if scorer == 1:
                winner = "agent_1"
                goals_1 = 1
            elif scorer == 2:
                winner = "agent_2"
                goals_2 = 1
            break

    if log_steps:
        logger.flush_steps()

    row = metrics.finalize(
        run_name=run_name,
        episode=episode,
        total_steps=env.match.steps,
        winner=winner,
        score_agent_1=env.match.score_team_1,
        score_agent_2=env.match.score_team_2,
        total_reward_agent_1=total_reward_1,
        total_reward_agent_2=total_reward_2,
        goals_agent_1=goals_1,
        goals_agent_2=goals_2,
        epsilon=learner.epsilon,
        timestamp=utc_timestamp(),
    )
    logger.log_episode(row)

    avg_loss = sum(losses) / len(losses) if losses else None
    summary = SelfPlaySummary(
        episode=episode,
        steps=env.match.steps,
        winner=winner,
        reward_1=total_reward_1,
        reward_2=total_reward_2,
        avg_loss=avg_loss,
        epsilon=learner.epsilon,
        opponent_name=opponent_name,
    )
    learner.decay_epsilon()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RLStriker with V11 checkpoint self-play.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of self-play episodes.")
    parser.add_argument("--run-name", type=str, default=None, help="Folder name under data/training_runs/.")
    parser.add_argument("--train-agent", type=int, choices=(1, 2), default=1, help="Which side learns.")
    parser.add_argument("--render", action="store_true", help="Render training in a Pygame window.")
    parser.add_argument("--log-steps", action="store_true", help="Write optional per-step CSV rows.")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save learner checkpoint every N episodes.")
    parser.add_argument(
        "--opponent-refresh-every",
        type=int,
        default=250,
        help="Add the current model to the opponent pool every N episodes.",
    )
    parser.add_argument("--opponent-pool-size", type=int, default=8, help="Maximum checkpoint opponents to keep.")
    parser.add_argument(
        "--random-opponent-prob",
        type=float,
        default=0.25,
        help="Probability of playing a random opponent even when checkpoints exist.",
    )
    parser.add_argument(
        "--initial-opponent",
        action="append",
        type=Path,
        default=[],
        help="Optional checkpoint path to seed the opponent pool. Can be used multiple times.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--hidden-size", type=int, default=128, help="DQN hidden layer size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration epsilon.")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum exploration epsilon.")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Episode-level epsilon decay.")
    parser.add_argument("--batch-size", type=int, default=64, help="Replay batch size.")
    parser.add_argument("--buffer-size", type=int, default=50_000, help="Replay buffer capacity.")
    parser.add_argument("--target-update-every", type=int, default=500, help="Learner steps per target update.")
    args = parser.parse_args()

    if not 0.0 <= args.random_opponent_prob <= 1.0:
        raise ValueError("--random-opponent-prob must be between 0 and 1")

    _set_seed(args.seed)

    env = SoccerEnv(render_mode="human" if args.render else None)
    initial_state = env.reset()
    state_size = len(initial_state)
    random_agent = RandomAgent()
    checkpoint_pool = _load_compatible_opponents(args.initial_opponent, state_size)

    learner = DQNAgent(
        state_size=state_size,
        action_size=ACTION_SPACE_SIZE,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update_every=args.target_update_every,
    )

    logger = EpisodeLogger(
        run_name=args.run_name,
        log_steps=args.log_steps,
        config={
            "script": "self_play.py",
            "version": "V16",
            "agent": "dqn",
            "opponent": "checkpoint_self_play",
            "train_agent": args.train_agent,
            "reward_version": "v2",
            "episodes": args.episodes,
            "max_steps": C.MAX_STEPS,
            "state_size": state_size,
            "action_size": ACTION_SPACE_SIZE,
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "target_update_every": args.target_update_every,
            "checkpoint_every": args.checkpoint_every,
            "opponent_refresh_every": args.opponent_refresh_every,
            "opponent_pool_size": args.opponent_pool_size,
            "random_opponent_prob": args.random_opponent_prob,
            "initial_opponents": [str(path) for path in args.initial_opponent],
            "render": args.render,
            "log_steps": args.log_steps,
            "seed": args.seed,
        },
    )
    checkpoint_dir = logger.run_dir / "checkpoints"
    opponent_dir = logger.run_dir / "opponents"

    print(f"Logging to: {logger.run_dir}")
    print(f"Training DQN agent {args.train_agent} with self-play for {args.episodes} episodes.")
    if checkpoint_pool:
        print(f"Loaded {len(checkpoint_pool)} initial checkpoint opponent(s).")

    try:
        for episode in range(1, args.episodes + 1):
            opponent = _select_opponent(
                checkpoint_pool=checkpoint_pool,
                random_agent=random_agent,
                random_opponent_prob=args.random_opponent_prob,
            )
            metrics = EpisodeMetricsTracker()
            summary = run_self_play_episode(
                env=env,
                learner=learner,
                opponent=opponent,
                episode=episode,
                logger=logger,
                metrics=metrics,
                log_steps=args.log_steps,
                run_name=logger.run_name,
                train_agent=args.train_agent,
            )

            loss_text = "n/a" if summary.avg_loss is None else f"{summary.avg_loss:.4f}"
            print(
                f"Episode {episode:04d} | steps={summary.steps:4d} | winner={summary.winner:7s} | "
                f"rewards=({summary.reward_1:+.1f}, {summary.reward_2:+.1f}) | "
                f"epsilon={summary.epsilon:.3f} | loss={loss_text} | opponent={summary.opponent_name}"
            )

            if args.checkpoint_every > 0 and episode % args.checkpoint_every == 0:
                path = save_checkpoint(
                    learner=learner,
                    checkpoint_dir=checkpoint_dir,
                    episode=episode,
                    run_name=logger.run_name,
                    train_agent=args.train_agent,
                )
                print(f"Saved checkpoint: {path}")

            if args.opponent_refresh_every > 0 and episode % args.opponent_refresh_every == 0:
                path = save_checkpoint(
                    learner=learner,
                    checkpoint_dir=opponent_dir,
                    episode=episode,
                    run_name=logger.run_name,
                    train_agent=args.train_agent,
                )
                checkpoint_pool.append(CheckpointOpponent(path))
                checkpoint_pool = checkpoint_pool[-args.opponent_pool_size :]
                print(f"Added opponent snapshot: {path}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        final_episode = min(args.episodes, episode if "episode" in locals() else 0)
        final_path = save_checkpoint(
            learner=learner,
            checkpoint_dir=checkpoint_dir,
            episode=final_episode,
            run_name=logger.run_name,
            train_agent=args.train_agent,
            final=True,
        )
        print(f"Saved final checkpoint: {final_path}")
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
