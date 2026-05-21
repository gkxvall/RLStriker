"""Curriculum stage definitions for RLStriker V9."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    """One training stage in the curriculum progression."""

    number: int
    key: str
    name: str
    description: str
    opponent_mode: str
    reward_mode: str
    scoring_enabled: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "opponent_mode": self.opponent_mode,
            "reward_mode": self.reward_mode,
            "scoring_enabled": self.scoring_enabled,
        }


DEFAULT_CURRICULUM_STAGES = [
    CurriculumStage(
        number=1,
        key="reach_ball",
        name="Reach the Ball",
        description="One learning agent, no active opponent, reward for closing distance and touching the ball.",
        opponent_mode="none",
        reward_mode="reach_ball",
        scoring_enabled=False,
    ),
    CurriculumStage(
        number=2,
        key="push_to_goal",
        name="Push Toward Goal",
        description="One learning agent, no active opponent, reward for touches and ball progress toward goal.",
        opponent_mode="none",
        reward_mode="push_to_goal",
        scoring_enabled=False,
    ),
    CurriculumStage(
        number=3,
        key="score_goal",
        name="Score Goals",
        description="One learning agent, no active opponent, normal scoring rewards enabled.",
        opponent_mode="none",
        reward_mode="score_goal",
        scoring_enabled=True,
    ),
    CurriculumStage(
        number=4,
        key="weak_random_opponent",
        name="Weak Random Opponent",
        description="A weak random opponent is added while the learner keeps using normal match rewards.",
        opponent_mode="weak_random",
        reward_mode="match",
        scoring_enabled=True,
    ),
    CurriculumStage(
        number=5,
        key="self_play",
        name="Self-Play Mirror",
        description="The learner plays against its own current policy as a simple self-play bridge.",
        opponent_mode="self_play",
        reward_mode="match",
        scoring_enabled=True,
    ),
]
