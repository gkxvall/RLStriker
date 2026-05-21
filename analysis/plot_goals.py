"""Plot goals and episode length from training logs."""

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


def plot_goals(run_dir: Path, output_dir: Path | None = None, window: int = 100, show: bool = False) -> Path:
    df = load_episodes(run_dir)
    target = resolve_output_dir(run_dir, output_dir)
    fig, ax = setup_axes(f"Goals per Episode ({window}-Episode Average)", "Episode", "Goals")

    ax.plot(df["episode"], rolling(df["goals_agent_1"], window), label="Agent 1", color="#1f77b4", linewidth=2)
    ax.plot(df["episode"], rolling(df["goals_agent_2"], window), label="Agent 2", color="#d62728", linewidth=2)
    ax.legend()
    return save_figure(fig, target, "goals.png", show)


def plot_episode_length(
    run_dir: Path, output_dir: Path | None = None, window: int = 100, show: bool = False
) -> Path:
    df = load_episodes(run_dir)
    target = resolve_output_dir(run_dir, output_dir)
    fig, ax = setup_axes(f"Episode Length ({window}-Episode Average)", "Episode", "Steps")

    ax.plot(df["episode"], rolling(df["total_steps"], window), color="#2ca02c", linewidth=2)
    return save_figure(fig, target, "episode_length.png", show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot goals and episode length from episodes.csv.")
    add_plot_args(parser)
    args = parser.parse_args()
    goals_path = plot_goals(args.run_dir, args.output_dir, args.window, args.show)
    length_path = plot_episode_length(args.run_dir, args.output_dir, args.window, args.show)
    print(f"Saved {goals_path}")
    print(f"Saved {length_path}")


if __name__ == "__main__":
    main()
