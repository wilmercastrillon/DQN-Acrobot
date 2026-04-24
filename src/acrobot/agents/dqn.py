"""
Deep Q-Network (DQN) implementation in PyTorch.

This module intentionally avoids high-level RL libraries so every piece of
the algorithm is visible and editable for learning purposes.

Key components:
  - QNetwork     : a small fully-connected network that maps state → Q(s,a)
  - ReplayBuffer : stores (s, a, r, s', done) transitions for experience replay
  - DQNAgent     : the training loop, ε-greedy policy, and target-network sync
"""
import random
from collections import deque
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── Neural network ────────────────────────────────────────────────────


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ────────────────────────────────────────────────────


class ReplayBuffer:
    """Fixed-size FIFO buffer that stores transitions for experience replay."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ── Agent ─────────────────────────────────────────────────────────────


class DQNAgent:
    """
    Deep Q-Network agent implemented from scratch.

    Hyperparameters are intentionally exposed as constructor args so you
    can experiment with them directly.
    """

    def __init__(
        self,
        env_id: str = "Acrobot-v1",
        *,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.997,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 10,
        hidden: int = 512,
        max_steps_per_episode: int = 500,
    ) -> None:
    
        self.env_id = env_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.training_episodes = 0
        self.max_steps_per_episode = max_steps_per_episode
        
        env = gym.make(env_id)
        self.state_dim = env.observation_space.shape[0]   # 6
        self.action_dim = int(env.action_space.n)        # 3
        env.close()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)

    # ── policy ────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, *, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(t)
            return int(q_values.argmax(dim=1).item())

    def predict(
        self, obs: np.ndarray, *, deterministic: bool = True
    ) -> tuple[int, None]:
        return self.select_action(obs, deterministic=deterministic), None

    # ── learning step ─────────────────────────────────────────────────

    def _learn(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True).values
        target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ── training loop ─────────────────────────────────────────────────

    def train(self, total_episodes: int = 1000, log_interval: int = 20) -> list[float]:
        print("acrobot - lr: ", self.lr)
        print("acrobot - epsilon start: ", self.epsilon_start)
        print("acrobot - epsilon end: ", self.epsilon_end)
        print("acrobot - epsilon decay: ", self.epsilon_decay)
        print("acrobot - batch size: ", self.batch_size)
        print("acrobot - target update freq: ", self.target_update_freq)
        print("acrobot - max steps per episode: ", self.max_steps_per_episode)
        env = gym.make(self.env_id)
        rewards_history: list[float] = []

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            steps = 0

            while not done and steps < self.max_steps_per_episode:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.buffer.push(obs, action, float(reward), next_obs, done)
                self._learn()

                obs = next_obs
                total_reward += reward
                steps += 1

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if episode % log_interval == 0:
                avg = np.mean(rewards_history[-log_interval:])
                best = np.max(rewards_history[-log_interval:])
                print(
                    f"Episode {episode:4d}/{total_episodes} | "
                    f"Avg Reward: {avg:6.1f} | "
                    f"Best: {best:6.1f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"Buffer: {len(self.buffer)}"
                )

        env.close()
        return rewards_history

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_net_state": self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "env_id": self.env_id,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
        }
        torch.save(data, path)
        print(f"Saved DQN agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        data = torch.load(path, weights_only=False)
        agent = cls(
            data["env_id"],
            lr=data["lr"],
            gamma=data["gamma"],
            epsilon_start=data.get("epsilon", 0.05),
            epsilon_end=data["epsilon_end"],
            epsilon_decay=data["epsilon_decay"],
            batch_size=data["batch_size"],
            target_update_freq=data["target_update_freq"],
        )
        agent.q_net.load_state_dict(data["q_net_state"])
        agent.target_net.load_state_dict(data["target_net_state"])
        agent.optimizer.load_state_dict(data["optimizer_state"])
        agent.training_episodes = data["training_episodes"]
        agent.epsilon = data.get("epsilon", 0.05)
        return agent

    def info(self) -> str:
        params = sum(p.numel() for p in self.q_net.parameters())
        return (
            f"DQN agent for {self.env_id}\n"
            f"  Episodes trained  : {self.training_episodes}\n"
            f"  Network params    : {params:,}\n"
            f"  Epsilon           : {self.epsilon:.4f}\n"
            f"  LR / Gamma        : {self.lr} / {self.gamma}\n"
            f"  Batch size        : {self.batch_size}\n"
            f"  Target update     : every {self.target_update_freq} episodes\n"
            f"  Device            : {self.device}"
        )
