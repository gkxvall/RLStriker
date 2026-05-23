"""Run RLStriker 2v2 with baseline team agents."""

from __future__ import annotations

import argparse

import pygame

from agents.team_agent import TeamRandomAgent, TeamRoleAgent
from env import constants as C
from env.team_soccer_env import TeamSoccerEnv


def _make_agent(kind: str, team: int) -> TeamRandomAgent | TeamRoleAgent:
    if kind == "role":
        return TeamRoleAgent(team=team)
    return TeamRandomAgent()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2v2 team soccer baselines.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes.")
    parser.add_argument("--render", action="store_true", help="Render the 2v2 match.")
    parser.add_argument("--team-1", choices=("random", "role"), default="role", help="Team 1 policy.")
    parser.add_argument("--team-2", choices=("random", "role"), default="random", help="Team 2 policy.")
    args = parser.parse_args()

    env = TeamSoccerEnv(render_mode="human" if args.render else None)
    team_1 = _make_agent(args.team_1, team=1)
    team_2 = _make_agent(args.team_2, team=2)

    wins_1 = 0
    wins_2 = 0
    draws = 0
    total_steps = 0

    try:
        for episode in range(1, args.episodes + 1):
            state = env.reset()
            reward_1_total = 0.0
            reward_2_total = 0.0

            while True:
                action_1 = team_1.act(state)
                action_2 = team_2.act(state)
                state, reward_1, reward_2, done, info = env.step(action_1, action_2)
                reward_1_total += reward_1
                reward_2_total += reward_2

                if args.render:
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
                        wins_1 += 1
                        winner = "team_1"
                    elif scorer == 2:
                        wins_2 += 1
                        winner = "team_2"
                    else:
                        draws += 1
                        winner = "draw"
                    total_steps += env.match.steps
                    print(
                        f"Episode {episode:04d} | steps={env.match.steps:4d} | winner={winner:6s} | "
                        f"rewards=({reward_1_total:+.1f}, {reward_2_total:+.1f})"
                    )
                    break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()

    completed = wins_1 + wins_2 + draws
    if completed:
        print("\n2v2 run complete")
        print(f"- Episodes: {completed}")
        print(f"- Team 1 wins: {wins_1}")
        print(f"- Team 2 wins: {wins_2}")
        print(f"- Draws: {draws}")
        print(f"- Average steps/episode: {total_steps / completed:.1f}")


if __name__ == "__main__":
    main()
