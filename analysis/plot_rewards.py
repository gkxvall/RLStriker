"""Plot rolling average episode rewards."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.plot_utils import (
    add_plot_args,
    load_episodes,
    resolve_output_dir,
    rolling,
    save_figure,
    setup_axes,
)


def plot_rewards(run_dir: Path, output_dir: Path | None = None, window: int = 100, show: bool = False) -> Path:
    df = load_episodes(run_dir)
    target = resolve_output_dir(run_dir, output_dir)
    fig, ax = setup_axes(f"Average Reward per {window} Episodes", "Episode", "Reward")

    ax.plot(
        df["episode"],
        rolling(df["total_reward_agent_1"], window),
        label="Agent 1",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        df["episode"],
        rolling(df["total_reward_agent_2"], window),
        label="Agent 2",
        color="#d62728",
        linewidth=2,
    )
    ax.legend()
    return save_figure(fig, target, "rewards.png", show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rolling average rewards from episodes.csv.")
    add_plot_args(parser)
    args = parser.parse_args()
    path = plot_rewards(args.run_dir, args.output_dir, args.window, args.show)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
