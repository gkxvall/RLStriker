"""Checkpoint-backed opponent policies for V11 self-play."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from agents.model import QNetwork


class CheckpointOpponent:
    """Frozen DQN policy loaded from a saved checkpoint."""

    def __init__(self, checkpoint_path: str | Path, *, device: str | None = None) -> None:
        self.path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(self.path, map_location=self.device)

        self.state_size = int(checkpoint["state_size"])
        self.action_size = int(checkpoint["action_size"])
        self.hidden_size = _infer_hidden_size(checkpoint["policy_state_dict"])
        self.metadata: dict[str, Any] = dict(checkpoint.get("metadata", {}))

        self.policy_net = QNetwork(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_net.eval()

    @property
    def name(self) -> str:
        return self.path.stem

    def act(self, state: list[float]) -> int:
        if len(state) != self.state_size:
            raise ValueError(
                f"Checkpoint {self.path} expects state size {self.state_size}, "
                f"but environment returned {len(state)}."
            )

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())


def _infer_hidden_size(policy_state_dict: dict[str, torch.Tensor]) -> int:
    first_layer = policy_state_dict.get("net.0.weight")
    if first_layer is None:
        raise ValueError("Unsupported checkpoint format: missing net.0.weight")
    return int(first_layer.shape[0])
