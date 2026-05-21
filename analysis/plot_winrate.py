"""Plot rolling win rates from training logs."""

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


def plot_winrate(run_dir: Path, output_dir: Path | None = None, window: int = 100, show: bool = False) -> Path:
    df = load_episodes(run_dir).copy()
    target = resolve_output_dir(run_dir, output_dir)
    fig, ax = setup_axes(f"Win Rate ({window}-Episode Average)", "Episode", "Win Rate")

    agent_1_wins = (df["winner"] == "agent_1").astype(float)
    agent_2_wins = (df["winner"] == "agent_2").astype(float)
    draws = (df["winner"] == "draw").astype(float)

    ax.plot(df["episode"], rolling(agent_1_wins, window), label="Agent 1", color="#1f77b4", linewidth=2)
    ax.plot(df["episode"], rolling(agent_2_wins, window), label="Agent 2", color="#d62728", linewidth=2)
    ax.plot(df["episode"], rolling(draws, window), label="Draw", color="#7f7f7f", linewidth=1.5)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    return save_figure(fig, target, "winrate.png", show)


def plot_epsilon(run_dir: Path, output_dir: Path | None = None, window: int = 100, show: bool = False) -> Path:
    df = load_episodes(run_dir)
    target = resolve_output_dir(run_dir, output_dir)
    fig, ax = setup_axes("Epsilon Decay", "Episode", "Epsilon")

    ax.plot(df["episode"], df["epsilon"], color="#9467bd", linewidth=2, label="Epsilon")
    if len(df) > 1:
        ax.plot(df["episode"], rolling(df["epsilon"], window), color="#c5b0d5", linewidth=1.5, label="Smoothed")
    ax.legend()
    return save_figure(fig, target, "epsilon.png", show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot win rate and epsilon from episodes.csv.")
    add_plot_args(parser)
    args = parser.parse_args()
    winrate_path = plot_winrate(args.run_dir, args.output_dir, args.window, args.show)
    epsilon_path = plot_epsilon(args.run_dir, args.output_dir, args.window, args.show)
    print(f"Saved {winrate_path}")
    print(f"Saved {epsilon_path}")


if __name__ == "__main__":
    main()
