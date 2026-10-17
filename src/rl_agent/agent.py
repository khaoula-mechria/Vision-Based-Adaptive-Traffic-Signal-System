"""Double-DQN agent for traffic signal phase selection.

* Experience replay (uniform) and a target network (hard update every ``target_update_steps``
  gradient steps), Double-DQN targets, Huber loss, gradient clipping.
* Epsilon-greedy exploration with a linear schedule over environment steps. Exploration only
  happens when ``select_action(..., training=True)``; evaluation is always greedy.
* Time-limit truncation is *not* treated as terminal (bootstraps through it).
* Reproducible: one seed controls torch, numpy exploration and replay sampling.
* Checkpoints store network, optimiser, counters, config and dimensions; ``load`` validates
  dimensions and refuses missing files instead of silently using a random policy.

PPO/A2C were intentionally not added: the action space is a handful of discrete phase
requests, DQN is sufficient, and no experiment here justifies the extra complexity.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..errors import ConfigError, ModelNotFoundError

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DQNConfig:
    learning_rate: float = 5e-4
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000
    batch_size: int = 64
    buffer_size: int = 100_000
    learn_start: int = 1000
    train_every: int = 1
    target_update_steps: int = 500
    hidden_dims: tuple = (128, 128)
    grad_clip: float = 5.0
    seed: int = 0
    device: str = "cpu"

    def validate(self) -> None:
        if not 0 < self.gamma <= 1:
            raise ConfigError("gamma must be in (0, 1]")
        if self.learning_rate <= 0 or self.batch_size <= 0 or self.buffer_size < self.batch_size:
            raise ConfigError("invalid learning_rate / batch_size / buffer_size")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ConfigError("epsilon schedule must satisfy 0 <= end <= start <= 1")
        if self.learn_start < self.batch_size:
            raise ConfigError("learn_start must be >= batch_size")


class ReplayBuffer:
    """Uniform replay with preallocated arrays."""

    def __init__(self, capacity: int, state_dim: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.s = np.zeros((capacity, state_dim), np.float32)
        self.s2 = np.zeros((capacity, state_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.pos = 0
        self.size = 0

    def push(self, s, a, r, s2, terminated: bool) -> None:
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(terminated)
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int):
        idx = self.rng.integers(0, self.size, size=min(n, self.size))
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self) -> int:
        return self.size


def _build_mlp(in_dim: int, out_dim: int, hidden: tuple):
    layers: List[Any] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class TrafficSignalAgent:
    def __init__(self, state_dim: int, action_dim: int, config: Optional[DQNConfig] = None) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for the DQN agent: pip install -r requirements-rl.txt")
        self.cfg = config or DQNConfig()
        self.cfg.validate()
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        self.device = torch.device(self.cfg.device)
        torch.manual_seed(self.cfg.seed)
        self._rng = np.random.default_rng(self.cfg.seed)
        self.q_net = _build_mlp(state_dim, action_dim, self.cfg.hidden_dims).to(self.device)
        self.target_net = _build_mlp(state_dim, action_dim, self.cfg.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.cfg.learning_rate)
        self.replay = ReplayBuffer(self.cfg.buffer_size, state_dim, self._rng)
        self.env_steps = 0
        self.grad_steps = 0
        self.episodes = 0
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------- acting
    @property
    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.env_steps / max(1, c.epsilon_decay_steps))
        return c.epsilon_start + frac * (c.epsilon_end - c.epsilon_start)

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        """Greedy action; epsilon-greedy only when ``training=True``."""
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"expected state shape ({self.state_dim},), got {state.shape}")
        if training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.action_dim))
        with torch.no_grad():
            q = self.q_net(torch.from_numpy(state).unsqueeze(0).to(self.device))
            return int(q.argmax(dim=1).item())

    # ----------------------------------------------------------- learning
    def observe(self, s, a, r, s2, terminated: bool) -> Optional[float]:
        """Store a transition and (when due) take one gradient step. Training mode only."""
        self.replay.push(s, a, r, s2, terminated)
        self.env_steps += 1
        if len(self.replay) >= self.cfg.learn_start and self.env_steps % self.cfg.train_every == 0:
            return self._learn()
        return None

    def _learn(self) -> float:
        c = self.cfg
        s, a, r, s2, d = (torch.from_numpy(x).to(self.device) for x in self.replay.sample(c.batch_size))
        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best = self.q_net(s2).argmax(dim=1, keepdim=True)  # Double DQN
            nxt = self.target_net(s2).gather(1, best).squeeze(1)
            target = r + c.gamma * (1.0 - d) * nxt
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), c.grad_clip)
        self.optimizer.step()
        self.grad_steps += 1
        if self.grad_steps % c.target_update_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    def train(
        self,
        env,
        episodes: int,
        seeds: Optional[List[int]] = None,
        eval_env=None,
        eval_seeds: Optional[List[int]] = None,
        eval_every: int = 10,
        checkpoint_dir: Optional[str] = None,
        log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
        before_episode: Optional[Callable[[int, int], None]] = None,
        eval_setup: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, float]]:
        """Train for ``episodes`` episodes. Training and evaluation use separate envs/seeds.

        If ``eval_env`` is given, the checkpoint with the best greedy evaluation return is
        saved as ``best.pt`` (selection uses ``eval_seeds`` that should be disjoint from
        training seeds); ``last.pt`` is always written at the end.
        """
        best = -np.inf
        for ep in range(episodes):
            seed = seeds[ep % len(seeds)] if seeds else self.cfg.seed + ep
            if before_episode:
                before_episode(ep, seed)  # e.g. switch the demand scenario
            s, _ = env.reset(seed=seed)
            total, losses, done = 0.0, [], False
            while not done:
                a = self.select_action(s, training=True)
                s2, r, terminated, truncated, info = env.step(a)
                loss = self.observe(s, a, r, s2, terminated)  # truncation is NOT terminal
                if loss is not None:
                    losses.append(loss)
                s, total, done = s2, total + r, terminated or truncated
            self.episodes += 1
            rec: Dict[str, float] = {
                "episode": self.episodes, "train_return": total, "epsilon": self.epsilon,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "env_steps": self.env_steps,
            }
            if eval_env is not None and (ep + 1) % eval_every == 0:
                ev = self.evaluate(eval_env, eval_seeds or [10_000 + i for i in range(3)], setup=eval_setup)
                rec.update({f"eval_{k}": v for k, v in ev.items()})
                if checkpoint_dir and ev["mean_return"] > best:
                    best = ev["mean_return"]
                    self.save(os.path.join(checkpoint_dir, "best.pt"))
            self.history.append(rec)
            if log_fn:
                log_fn(rec)
        if checkpoint_dir:
            self.save(os.path.join(checkpoint_dir, "last.pt"))
            with open(os.path.join(checkpoint_dir, "history.json"), "w", encoding="utf-8") as fh:
                json.dump(self.history, fh)
        return self.history

    def evaluate(self, env, seeds: List[int], setup: Optional[Callable[[int, int], None]] = None) -> Dict[str, float]:
        """Greedy rollouts. Does not store experience, change epsilon or update weights."""
        self.q_net.eval()
        returns, summaries = [], []
        for i, seed in enumerate(seeds):
            if setup:
                setup(i, seed)
            s, _ = env.reset(seed=seed)
            total, done = 0.0, False
            while not done:
                s, r, terminated, truncated, info = env.step(self.select_action(s, training=False))
                total += r
                done = terminated or truncated
            returns.append(total)
            summaries.append(env.summary())
        self.q_net.train()
        out = {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns))}
        for k in summaries[0]:
            out[k] = float(np.mean([m[k] for m in summaries]))
        return out

    # ---------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "format": 1,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "config": asdict(self.cfg),
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "env_steps": self.env_steps,
                "grad_steps": self.grad_steps,
                "episodes": self.episodes,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TrafficSignalAgent":
        if not os.path.isfile(path):
            raise ModelNotFoundError(f"RL checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = dict(ckpt["config"])
        cfg_dict["hidden_dims"] = tuple(cfg_dict["hidden_dims"])
        cfg_dict["device"] = device
        agent = cls(ckpt["state_dim"], ckpt["action_dim"], DQNConfig(**cfg_dict))
        agent.q_net.load_state_dict(ckpt["q_net"])
        agent.target_net.load_state_dict(ckpt["target_net"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        agent.env_steps, agent.grad_steps, agent.episodes = ckpt["env_steps"], ckpt["grad_steps"], ckpt["episodes"]
        return agent

    def check_compatible(self, state_dim: int, action_dim: int) -> None:
        if (self.state_dim, self.action_dim) != (state_dim, action_dim):
            raise ConfigError(
                f"agent trained for state/action dims ({self.state_dim}, {self.action_dim}) "
                f"but intersection needs ({state_dim}, {action_dim})"
            )
