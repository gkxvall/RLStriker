"""Generate all V8 training graphs for one run."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.plot_goals import plot_episode_length, plot_goals
from analysis.plot_rewards import plot_rewards
from analysis.plot_touches import plot_distance_to_ball, plot_touches
from analysis.plot_utils import add_plot_args
from analysis.plot_winrate import plot_epsilon, plot_winrate


def plot_all(run_dir: Path, output_dir: Path | None = None, window: int = 100, show: bool = False) -> list[Path]:
    return [
        plot_rewards(run_dir, output_dir, window, show),
        plot_winrate(run_dir, output_dir, window, show),
        plot_goals(run_dir, output_dir, window, show),
        plot_touches(run_dir, output_dir, window, show),
        plot_distance_to_ball(run_dir, output_dir, window, show),
        plot_episode_length(run_dir, output_dir, window, show),
        plot_epsilon(run_dir, output_dir, window, show),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all RLStriker V8 graphs from episodes.csv.")
    add_plot_args(parser)
    args = parser.parse_args()
    paths = plot_all(args.run_dir, args.output_dir, args.window, args.show)
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
