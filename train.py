"""Train a DQN agent against random or curriculum opponents."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import pygame
import torch

from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from curriculum.curriculum_manager import CurriculumManager
from curriculum.stages import CurriculumStage
from env import constants as C
from env.soccer_env import ACTION_SPACE_SIZE, SoccerEnv
from logging_utils.episode_logger import EpisodeLogger, utc_timestamp
from logging_utils.metrics import EpisodeMetricsTracker


@dataclass
class TrainingSummary:
    episode: int
    steps: int
    winner: str
    reward_1: float
    reward_2: float
    avg_loss: float | None
    epsilon: float
    stage_name: str | None = None


def _step_event(info: dict, done: bool) -> str:
    if not done:
        return ""
    if info.get("goal"):
        return "goal"
    if info.get("max_steps"):
        return "max_steps"
    return "done"


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training_episode(
    *,
    env: SoccerEnv,
    learner: DQNAgent,
    opponent: RandomAgent,
    episode: int,
    logger: EpisodeLogger,
    metrics: EpisodeMetricsTracker,
    log_steps: bool,
    run_name: str,
    train_agent: int,
    curriculum: CurriculumManager | None = None,
    stage: CurriculumStage | None = None,
) -> TrainingSummary:
    state = env.reset()
    if curriculum is not None and stage is not None:
        curriculum.configure_episode(env, stage, train_agent)
        state = env.get_state()
    metrics.reset()

    total_reward_1 = 0.0
    total_reward_2 = 0.0
    losses: list[float] = []
    winner = "draw"
    goals_1 = 0
    goals_2 = 0

    while True:
        learner_action = learner.act(state, training=True)
        if curriculum is not None and stage is not None:
            context = curriculum.before_step(env, train_agent)
            opponent_action = curriculum.opponent_action(
                stage=stage,
                state=state,
                opponent=opponent,
                learner=learner,
            )
        else:
            context = None
            opponent_action = opponent.act()

        if train_agent == 1:
            action_1 = learner_action
            action_2 = opponent_action
        else:
            action_1 = opponent_action
            action_2 = learner_action

        next_state, reward_1, reward_2, done, info = env.step(action_1, action_2)
        train_reward = reward_1 if train_agent == 1 else reward_2
        if curriculum is not None and stage is not None and context is not None:
            train_reward = curriculum.training_reward(
                env=env,
                stage=stage,
                train_agent=train_agent,
                base_reward=train_reward,
                context=context,
            )
            if curriculum.keep_playing_if_scoring_disabled(env=env, stage=stage, info=info):
                done = False
                next_state = env.get_state()

        logged_reward_1 = train_reward if train_agent == 1 else reward_1
        logged_reward_2 = train_reward if train_agent == 2 else reward_2
        learner.remember(state, learner_action, train_reward, next_state, done)
        loss = learner.learn()
        if loss is not None:
            losses.append(loss)

        total_reward_1 += logged_reward_1
        total_reward_2 += logged_reward_2
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
                reward_agent_1=logged_reward_1,
                reward_agent_2=logged_reward_2,
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
    summary = TrainingSummary(
        episode=episode,
        steps=env.match.steps,
        winner=winner,
        reward_1=total_reward_1,
        reward_2=total_reward_2,
        avg_loss=avg_loss,
        epsilon=learner.epsilon,
        stage_name=stage.name if stage else None,
    )
    learner.decay_epsilon()
    return summary


def save_checkpoint(
    *,
    learner: DQNAgent,
    checkpoint_dir: Path,
    episode: int,
    run_name: str,
    train_agent: int,
    final: bool = False,
    curriculum_stage: str | None = None,
) -> Path:
    name = "final.pt" if final else f"episode_{episode:06d}.pt"
    path = checkpoint_dir / name
    learner.save(
        path,
        metadata={
            "run_name": run_name,
            "episode": episode,
            "train_agent": train_agent,
            "opponent": "random",
            "reward_version": "v1",
            "curriculum_stage": curriculum_stage,
        },
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN agent for RLStriker.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--run-name", type=str, default=None, help="Folder name under data/training_runs/.")
    parser.add_argument("--train-agent", type=int, choices=(1, 2), default=1, help="Which agent learns.")
    parser.add_argument("--render", action="store_true", help="Render training in a Pygame window.")
    parser.add_argument("--log-steps", action="store_true", help="Write optional per-step CSV rows.")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save a checkpoint every N episodes.")
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
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable V9 curriculum learning stages.",
    )
    parser.add_argument(
        "--curriculum-stage-episodes",
        type=str,
        default=None,
        help="Comma-separated episode counts for the five curriculum stages, e.g. 100,100,200,300,300.",
    )
    args = parser.parse_args()

    _set_seed(args.seed)

    env = SoccerEnv(render_mode="human" if args.render else None)
    initial_state = env.reset()
    state_size = len(initial_state)
    opponent = RandomAgent()
    curriculum: CurriculumManager | None = None
    if args.curriculum:
        stage_episodes = CurriculumManager.parse_stage_episodes(args.curriculum_stage_episodes)
        curriculum = CurriculumManager(total_episodes=args.episodes, stage_episodes=stage_episodes)

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
            "script": "train.py",
            "version": "V9",
            "agent": "dqn",
            "opponent": "curriculum" if args.curriculum else "random",
            "train_agent": args.train_agent,
            "reward_version": "v1",
            "curriculum_enabled": args.curriculum,
            "curriculum_schedule": curriculum.describe_schedule() if curriculum else None,
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
            "render": args.render,
            "log_steps": args.log_steps,
            "seed": args.seed,
        },
    )
    checkpoint_dir = logger.run_dir / "checkpoints"

    print(f"Logging to: {logger.run_dir}")
    if curriculum is None:
        print(f"Training DQN agent {args.train_agent} against a random opponent for {args.episodes} episodes.")
    else:
        print(f"Training DQN agent {args.train_agent} with V9 curriculum for {args.episodes} episodes.")

    try:
        for episode in range(1, args.episodes + 1):
            stage = curriculum.stage_for_episode(episode) if curriculum else None
            metrics = EpisodeMetricsTracker()
            summary = run_training_episode(
                env=env,
                learner=learner,
                opponent=opponent,
                episode=episode,
                logger=logger,
                metrics=metrics,
                log_steps=args.log_steps,
                run_name=logger.run_name,
                train_agent=args.train_agent,
                curriculum=curriculum,
                stage=stage,
            )

            loss_text = "n/a" if summary.avg_loss is None else f"{summary.avg_loss:.4f}"
            stage_text = f" | stage={summary.stage_name}" if summary.stage_name else ""
            print(
                f"Episode {episode:04d} | steps={summary.steps:4d} | "
                f"winner={summary.winner:7s} | rewards=({summary.reward_1:+.1f}, {summary.reward_2:+.1f}) | "
                f"epsilon={summary.epsilon:.3f} | loss={loss_text}{stage_text}"
            )

            if args.checkpoint_every > 0 and episode % args.checkpoint_every == 0:
                path = save_checkpoint(
                    learner=learner,
                    checkpoint_dir=checkpoint_dir,
                    episode=episode,
                    run_name=logger.run_name,
                    train_agent=args.train_agent,
                    curriculum_stage=stage.name if stage else None,
                )
                print(f"Saved checkpoint: {path}")
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
            curriculum_stage=stage.name if "stage" in locals() and stage else None,
            final=True,
        )
        print(f"Saved final checkpoint: {final_path}")
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
