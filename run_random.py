"""Run RLStriker with two random agents (V4 baseline)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pygame

from agents.random_agent import RandomAgent
from env import constants as C
from env.soccer_env import SoccerEnv


@dataclass
class EpisodeSummary:
    episode: int
    steps: int
    winner: str
    reward_1: float
    reward_2: float


def run_episode(env: SoccerEnv, agent_1: RandomAgent, agent_2: RandomAgent) -> EpisodeSummary:
    env.reset()
    total_reward_1 = 0.0
    total_reward_2 = 0.0
    winner = "draw"

    while True:
        action_1 = agent_1.act()
        action_2 = agent_2.act()
        _, reward_1, reward_2, done, info = env.step(action_1, action_2)
        total_reward_1 += reward_1
        total_reward_2 += reward_2

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
            elif scorer == 2:
                winner = "agent_2"
            break

    return EpisodeSummary(
        episode=0,
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
    args = parser.parse_args()

    env = SoccerEnv(render_mode="human" if args.render else None)
    agent_1 = RandomAgent()
    agent_2 = RandomAgent()

    wins_1 = 0
    wins_2 = 0
    draws = 0
    total_steps = 0

    print(f"Starting random self-play for {args.episodes} episodes...")
    try:
        for episode in range(1, args.episodes + 1):
            summary = run_episode(env, agent_1, agent_2)
            summary.episode = episode
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

