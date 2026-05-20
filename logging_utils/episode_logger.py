"""Episode-level CSV logging and optional step-level CSV."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from logging_utils.metrics import EpisodeMetricsRow

EPISODE_FIELDNAMES = [
    "run_name",
    "episode",
    "total_steps",
    "winner",
    "score_agent_1",
    "score_agent_2",
    "total_reward_agent_1",
    "total_reward_agent_2",
    "goals_agent_1",
    "goals_agent_2",
    "touches_agent_1",
    "touches_agent_2",
    "kicks_agent_1",
    "kicks_agent_2",
    "steals_agent_1",
    "steals_agent_2",
    "avg_distance_to_ball_agent_1",
    "avg_distance_to_ball_agent_2",
    "epsilon",
    "timestamp",
]

STEP_FIELDNAMES = [
    "run_name",
    "episode",
    "step",
    "agent_1_x",
    "agent_1_y",
    "agent_2_x",
    "agent_2_y",
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "action_agent_1",
    "action_agent_2",
    "reward_agent_1",
    "reward_agent_2",
    "event",
]


def _default_run_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _sanitize_run_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return cleaned[:120] if cleaned else _default_run_name()


class EpisodeLogger:
    """
    Creates ``data/training_runs/<run_name>/`` with ``config.json`` and ``episodes.csv``.
    Optional ``steps.csv`` when step logging is enabled.
    """

    def __init__(
        self,
        run_name: str | None = None,
        *,
        config: dict[str, Any] | None = None,
        log_steps: bool = False,
        base_dir: str | Path = "data/training_runs",
    ) -> None:
        self.run_name = _sanitize_run_name(run_name) if run_name else _default_run_name()
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._episodes_path = self.run_dir / "episodes.csv"
        self._steps_path = self.run_dir / "steps.csv"
        self._log_steps = log_steps
        self._steps_file: Any = None
        self._steps_writer: csv.DictWriter | None = None

        cfg = dict(config or {})
        cfg.setdefault("run_name", self.run_name)
        cfg.setdefault("log_steps", log_steps)
        self._write_config(cfg)

        self._episodes_file = open(self._episodes_path, "w", newline="", encoding="utf-8")
        self._episodes_writer = csv.DictWriter(self._episodes_file, fieldnames=EPISODE_FIELDNAMES)
        self._episodes_writer.writeheader()
        self._episodes_file.flush()

    def _write_config(self, config: dict[str, Any]) -> None:
        path = self.run_dir / "config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

    @property
    def episodes_csv_path(self) -> Path:
        return self._episodes_path

    @property
    def steps_csv_path(self) -> Path | None:
        return self._steps_path if self._log_steps else None

    def log_episode(self, row: EpisodeMetricsRow) -> None:
        self._episodes_writer.writerow(row.as_csv_dict())
        self._episodes_file.flush()

    def log_step(
        self,
        *,
        episode: int,
        step: int,
        agent_1_x: float,
        agent_1_y: float,
        agent_2_x: float,
        agent_2_y: float,
        ball_x: float,
        ball_y: float,
        ball_vx: float,
        ball_vy: float,
        action_agent_1: int,
        action_agent_2: int,
        reward_agent_1: float,
        reward_agent_2: float,
        event: str = "",
    ) -> None:
        if not self._log_steps:
            return
        if self._steps_writer is None:
            self._steps_file = open(self._steps_path, "w", newline="", encoding="utf-8")
            self._steps_writer = csv.DictWriter(self._steps_file, fieldnames=STEP_FIELDNAMES)
            self._steps_writer.writeheader()
        self._steps_writer.writerow(
            {
                "run_name": self.run_name,
                "episode": episode,
                "step": step,
                "agent_1_x": round(agent_1_x, 4),
                "agent_1_y": round(agent_1_y, 4),
                "agent_2_x": round(agent_2_x, 4),
                "agent_2_y": round(agent_2_y, 4),
                "ball_x": round(ball_x, 4),
                "ball_y": round(ball_y, 4),
                "ball_vx": round(ball_vx, 4),
                "ball_vy": round(ball_vy, 4),
                "action_agent_1": action_agent_1,
                "action_agent_2": action_agent_2,
                "reward_agent_1": reward_agent_1,
                "reward_agent_2": reward_agent_2,
                "event": event,
            }
        )

    def flush_steps(self) -> None:
        if self._steps_file is not None:
            self._steps_file.flush()

    def close(self) -> None:
        self._episodes_file.close()
        if self._steps_file is not None:
            self._steps_file.close()

    def __enter__(self) -> EpisodeLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
