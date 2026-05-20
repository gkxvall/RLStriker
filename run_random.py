"""Run RLStriker with two random agents (V4 baseline, V5 episode logging)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pygame

from agents.random_agent import RandomAgent
from env import constants as C
from env.soccer_env import SoccerEnv
from logging_utils.episode_logger import EpisodeLogger, utc_timestamp
from logging_utils.metrics import EpisodeMetricsTracker


@dataclass
class EpisodeSummary:
    episode: int
    steps: int
    winner: str
    reward_1: float
    reward_2: float


def _step_event(info: dict, done: bool) -> str:
    if not done:
        return ""
    if info.get("goal"):
        return "goal"
    if info.get("max_steps"):
        return "max_steps"
    return "done"


def run_episode(
    env: SoccerEnv,
    agent_1: RandomAgent,
    agent_2: RandomAgent,
    *,
    episode: int,
    logger: EpisodeLogger | None,
    metrics: EpisodeMetricsTracker,
    log_steps: bool,
    epsilon: float,
    run_name: str,
) -> EpisodeSummary:
    env.reset()
    metrics.reset()

    total_reward_1 = 0.0
    total_reward_2 = 0.0
    winner = "draw"
    goals_1 = 0
    goals_2 = 0

    while True:
        action_1 = agent_1.act()
        action_2 = agent_2.act()
        _, reward_1, reward_2, done, info = env.step(action_1, action_2)
        total_reward_1 += reward_1
        total_reward_2 += reward_2
        last_info = info

        metrics.on_step(env, info)

        if logger is not None and log_steps:
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

        if done:
            scorer = info.get("scorer")
            if scorer == 1:
                winner = "agent_1"
                goals_1 = 1
            elif scorer == 2:
                winner = "agent_2"
                goals_2 = 1
            break

    if logger is not None and log_steps:
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
        epsilon=epsilon,
        timestamp=utc_timestamp(),
    )
    if logger is not None:
        logger.log_episode(row)

    return EpisodeSummary(
        episode=episode,
        steps=env.match.steps,
        winner=winner,
        reward_1=total_reward_1,
        reward_2=total_reward_2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automatic self-play with random agents.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to simulate.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the match in a Pygame window (slower).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Training run folder name under data/training_runs/. Default: run_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--log-steps",
        action="store_true",
        help="Write optional steps.csv (can grow large).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Exploration epsilon recorded in episodes.csv (random agents use 1.0).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable writing data/training_runs/ (not recommended).",
    )
    args = parser.parse_args()

    env = SoccerEnv(render_mode="human" if args.render else None)
    agent_1 = RandomAgent()
    agent_2 = RandomAgent()

    logger: EpisodeLogger | None = None
    if not args.no_log:
        logger = EpisodeLogger(
            run_name=args.run_name,
            log_steps=args.log_steps,
            config={
                "script": "run_random.py",
                "agent": "random",
                "episodes": args.episodes,
                "max_steps": C.MAX_STEPS,
                "epsilon": args.epsilon,
                "render": args.render,
                "log_steps": args.log_steps,
            },
        )
        print(f"Logging to: {logger.run_dir}")

    run_name = logger.run_name if logger else (args.run_name or "no_log")

    wins_1 = 0
    wins_2 = 0
    draws = 0
    total_steps = 0

    print(f"Starting random self-play for {args.episodes} episodes...")
    try:
        for episode in range(1, args.episodes + 1):
            metrics = EpisodeMetricsTracker()
            summary = run_episode(
                env,
                agent_1,
                agent_2,
                episode=episode,
                logger=logger,
                metrics=metrics,
                log_steps=args.log_steps,
                epsilon=args.epsilon,
                run_name=run_name,
            )
            total_steps += summary.steps

            if summary.winner == "agent_1":
                wins_1 += 1
            elif summary.winner == "agent_2":
                wins_2 += 1
            else:
                draws += 1

            print(
                f"Episode {episode:04d} | steps={summary.steps:4d} | "
                f"winner={summary.winner:7s} | rewards=({summary.reward_1:+.1f}, {summary.reward_2:+.1f})"
            )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if logger is not None:
            logger.close()
        env.close()

    episodes_done = wins_1 + wins_2 + draws
    if episodes_done > 0:
        print("\nRun complete")
        print(f"- Episodes: {episodes_done}")
        print(f"- Agent 1 wins: {wins_1}")
        print(f"- Agent 2 wins: {wins_2}")
        print(f"- Draws: {draws}")
        print(f"- Average steps/episode: {total_steps / episodes_done:.1f}")


if __name__ == "__main__":
    main()
