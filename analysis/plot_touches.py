"""Plot ball touches and distance-to-ball metrics."""

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


def plot_touches(run_dir: Path, output_dir: Path | None = None, window: int = 100, show: bool = False) -> Path:
    df = load_episodes(run_dir)
    target = resolve_output_dir(run_dir, output_dir)
    fig, ax = setup_axes(f"Ball Touches per Episode ({window}-Episode Average)", "Episode", "Touches")

    ax.plot(
        df["episode"],
        rolling(df["touches_agent_1"], window),
        label="Agent 1",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        df["episode"],
        rolling(df["touches_agent_2"], window),
        label="Agent 2",
        color="#d62728",
        linewidth=2,
    )
    ax.legend()
    return save_figure(fig, target, "touches.png", show)


def plot_distance_to_ball(
    run_dir: Path, output_dir: Path | None = None, window: int = 100, show: bool = False
) -> Path:
    df = load_episodes(run_dir)
    target = resolve_output_dir(run_dir, output_dir)
    fig, ax = setup_axes(f"Average Distance to Ball ({window}-Episode Average)", "Episode", "Pixels")

    ax.plot(
        df["episode"],
        rolling(df["avg_distance_to_ball_agent_1"], window),
        label="Agent 1",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        df["episode"],
        rolling(df["avg_distance_to_ball_agent_2"], window),
        label="Agent 2",
        color="#d62728",
        linewidth=2,
    )
    ax.legend()
    return save_figure(fig, target, "distance_to_ball.png", show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ball touches and distance-to-ball from episodes.csv.")
    add_plot_args(parser)
    args = parser.parse_args()
    touches_path = plot_touches(args.run_dir, args.output_dir, args.window, args.show)
    distance_path = plot_distance_to_ball(args.run_dir, args.output_dir, args.window, args.show)
    print(f"Saved {touches_path}")
    print(f"Saved {distance_path}")


if __name__ == "__main__":
    main()
