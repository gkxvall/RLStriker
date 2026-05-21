"""Shared helpers for RLStriker training plots."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_cache_root = Path(tempfile.gettempdir()) / "rlstriker_plot_cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

import matplotlib.pyplot as plt
import pandas as pd


def add_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to a data/training_runs/<run_name>/ folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for PNG output. Default: <run-dir>/plots.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Rolling average window for smoothed lines.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after saving.",
    )


def load_episodes(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "episodes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find episodes.csv at {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"episodes.csv is empty: {path}")
    return df


def resolve_output_dir(run_dir: Path, output_dir: Path | None) -> Path:
    target = output_dir if output_dir is not None else run_dir / "plots"
    target.mkdir(parents=True, exist_ok=True)
    return target


def rolling(series: pd.Series, window: int) -> pd.Series:
    window = max(1, window)
    return series.rolling(window=window, min_periods=1).mean()


def setup_axes(title: str, xlabel: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    return fig, ax


def save_figure(fig: plt.Figure, output_dir: Path, filename: str, show: bool = False) -> Path:
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return path
