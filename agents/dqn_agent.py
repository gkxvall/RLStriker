"""DQN agent implementation for RLStriker."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import optim

from agents.model import QNetwork
from agents.replay_buffer import ReplayBuffer


class DQNAgent:
    """Epsilon-greedy DQN agent with a target network and replay buffer."""

    def __init__(
        self,
        *,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        target_update_every: int = 500,
        device: str | None = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy_net = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.learn_steps = 0

    def act(self, state: list[float], *, training: bool = True) -> int:
        """Choose an action from the current policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = self._state_tensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def remember(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        """Run one optimization step when enough samples are available."""
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        states = torch.tensor([t.state for t in transitions], dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(
            [t.next_state for t in transitions], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_size": self.state_size,
                "action_size": self.action_size,
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_steps": self.learn_steps,
                "metadata": metadata or {},
            },
            path,
        )

    def load(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.learn_steps = int(checkpoint.get("learn_steps", self.learn_steps))
        return dict(checkpoint.get("metadata", {}))

    def _state_tensor(self, state: list[float]) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32, device=self.device)
