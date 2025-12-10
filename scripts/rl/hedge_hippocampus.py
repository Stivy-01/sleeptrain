"""
Hedge policy for hippocampus actions (STORE / REJECT / CORRECT).

Implements a simple exponential-weights Hedge algorithm with:
- Persistence (save/load JSON).
- Optional reward clipping and strict mode.
- Action interface aligned to hippocampus decisions.
- Routing replay with clipping (multiplicative weights on paths).
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple


ACTIONS = ["REJECT", "STORE", "CORRECT"]


@dataclass
class ReplayEntry:
    """Single entry in the routing replay buffer."""
    action_idx: int
    reward: float
    context_hash: Optional[str] = None


@dataclass
class HedgeState:
    actions: List[str]
    weights: List[float]
    eta: float
    t: int

    def normalize(self) -> None:
        total = sum(self.weights)
        if total <= 0:
            self.weights = [1.0 for _ in self.actions]
            total = len(self.actions)
        self.weights = [w / total for w in self.weights]


@dataclass
class RoutingConfig:
    """Configuration for routing replay and clipping."""
    replay_buffer_size: int = 100  # max entries in replay buffer
    replay_sample_size: int = 10  # samples per replay update
    replay_weight: float = 0.3  # weight of replay vs online update
    clip_min: float = 0.05  # minimum probability floor
    clip_max: float = 0.90  # maximum probability ceiling
    decay_factor: float = 0.99  # decay old replay entries
    enable_replay: bool = True


class HedgePolicy:
    """
    Exponential-weights Hedge policy over hippocampus actions.

    Features:
    - actions: default ["REJECT", "STORE", "CORRECT"]
    - eta: learning rate (defaults to sqrt(2*ln(K)/T) style)
    - Routing replay: maintains buffer of past decisions for periodic replay updates
    - Probability clipping: enforces min/max bounds on action probabilities
    """

    def __init__(
        self,
        actions: Sequence[str] = ACTIONS,
        eta: float = 0.5,
        strict: bool = False,
        reward_clip: Optional[Tuple[float, float]] = (-1.0, 1.0),
        routing_config: Optional[RoutingConfig] = None,
    ):
        self.state = HedgeState(
            actions=list(actions),
            weights=[1.0 for _ in actions],
            eta=eta,
            t=0,
        )
        self.state.normalize()
        self.strict = strict
        self.reward_clip = reward_clip

        # Routing replay configuration
        self.routing_config = routing_config or RoutingConfig()
        self.replay_buffer: Deque[ReplayEntry] = deque(
            maxlen=self.routing_config.replay_buffer_size
        )

    # ------------------------------------------------------------------ #
    # Core Hedge
    # ------------------------------------------------------------------ #
    def probs(self) -> List[float]:
        """Get current action probabilities with clipping applied."""
        weights = list(self.state.weights)
        if self.routing_config.clip_min > 0 or self.routing_config.clip_max < 1:
            weights = self._apply_probability_clipping(weights)
        return weights

    def _apply_probability_clipping(self, weights: List[float]) -> List[float]:
        """Apply min/max probability clipping and renormalize."""
        clip_min = self.routing_config.clip_min
        clip_max = self.routing_config.clip_max

        clipped = [max(clip_min, min(clip_max, w)) for w in weights]
        total = sum(clipped)
        if total > 0:
            clipped = [w / total for w in clipped]
        return clipped

    def pick_action(self) -> Tuple[int, str]:
        """Pick the argmax action (deterministic for reproducibility)."""
        weights = self.probs()
        idx = max(range(len(weights)), key=lambda i: weights[i])
        return idx, self.state.actions[idx]

    def pick_action_stochastic(self) -> Tuple[int, str]:
        """Pick action stochastically according to current probabilities."""
        import random
        weights = self.probs()
        r = random.random()
        cumulative = 0.0
        for idx, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return idx, self.state.actions[idx]
        # Fallback to last action
        return len(weights) - 1, self.state.actions[-1]

    def update(self, action_idx: int, reward: float, context_hash: Optional[str] = None) -> None:
        """Update weights given reward for the chosen action."""
        if action_idx < 0 or action_idx >= len(self.state.weights):
            raise ValueError("action_idx out of range")

        r = self._clip_reward(reward)

        # Add to replay buffer if enabled
        if self.routing_config.enable_replay:
            self.replay_buffer.append(
                ReplayEntry(action_idx=action_idx, reward=r, context_hash=context_hash)
            )

        # Online update: exponential weight update
        self._apply_hedge_update(action_idx, r, weight=1.0)

        self.state.t += 1
        self.state.normalize()

    def _apply_hedge_update(self, action_idx: int, reward: float, weight: float = 1.0) -> None:
        """Apply a single Hedge update with optional weighting."""
        effective_eta = self.state.eta * weight
        self.state.weights[action_idx] *= math.exp(effective_eta * reward)

    def replay_update(self, num_samples: Optional[int] = None) -> int:
        """
        Perform replay updates from the buffer.

        Returns the number of replay samples used.
        """
        if not self.routing_config.enable_replay or not self.replay_buffer:
            return 0

        num_samples = num_samples or self.routing_config.replay_sample_size
        num_samples = min(num_samples, len(self.replay_buffer))

        if num_samples == 0:
            return 0

        # Sample from replay buffer (recent entries more likely)
        import random
        buffer_list = list(self.replay_buffer)

        # Weight recent entries higher
        indices = list(range(len(buffer_list)))
        weights = [
            (self.routing_config.decay_factor ** (len(buffer_list) - i - 1))
            for i in indices
        ]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]

        # Sample indices
        sampled_indices = random.choices(indices, weights=probs, k=num_samples)

        # Apply replay updates with reduced weight
        replay_weight = self.routing_config.replay_weight
        for idx in sampled_indices:
            entry = buffer_list[idx]
            self._apply_hedge_update(entry.action_idx, entry.reward, weight=replay_weight)

        self.state.normalize()
        return num_samples

    def _clip_reward(self, reward: float) -> float:
        if self.reward_clip is None:
            return reward
        lo, hi = self.reward_clip
        if self.strict:
            if reward < lo or reward > hi:
                raise ValueError(f"reward {reward} outside [{lo}, {hi}]")
        return max(lo, min(hi, reward))

    def get_replay_stats(self) -> Dict[str, float]:
        """Get statistics about the replay buffer."""
        if not self.replay_buffer:
            return {"size": 0, "avg_reward": 0.0}

        rewards = [e.reward for e in self.replay_buffer]
        action_counts = {a: 0 for a in self.state.actions}
        for entry in self.replay_buffer:
            action_name = self.state.actions[entry.action_idx]
            action_counts[action_name] += 1

        return {
            "size": len(self.replay_buffer),
            "avg_reward": sum(rewards) / len(rewards),
            "action_distribution": action_counts,
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        data = asdict(self.state)
        data["strict"] = self.strict
        data["reward_clip"] = self.reward_clip
        data["routing_config"] = asdict(self.routing_config)
        # Save replay buffer (limited to avoid huge files)
        replay_entries = [
            {"action_idx": e.action_idx, "reward": e.reward, "context_hash": e.context_hash}
            for e in list(self.replay_buffer)[-50:]  # Keep last 50
        ]
        data["replay_buffer"] = replay_entries
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "HedgePolicy":
        raw = json.loads(Path(path).read_text())
        state = HedgeState(
            actions=raw["actions"],
            weights=raw["weights"],
            eta=raw["eta"],
            t=raw["t"],
        )

        # Load routing config if present
        routing_config = None
        if "routing_config" in raw:
            rc = raw["routing_config"]
            routing_config = RoutingConfig(
                replay_buffer_size=rc.get("replay_buffer_size", 100),
                replay_sample_size=rc.get("replay_sample_size", 10),
                replay_weight=rc.get("replay_weight", 0.3),
                clip_min=rc.get("clip_min", 0.05),
                clip_max=rc.get("clip_max", 0.90),
                decay_factor=rc.get("decay_factor", 0.99),
                enable_replay=rc.get("enable_replay", True),
            )

        policy = cls(
            actions=state.actions,
            eta=state.eta,
            strict=raw.get("strict", False),
            reward_clip=tuple(raw.get("reward_clip")) if raw.get("reward_clip") else None,
            routing_config=routing_config,
        )
        policy.state = state
        policy.state.normalize()

        # Restore replay buffer
        if "replay_buffer" in raw:
            for entry in raw["replay_buffer"]:
                policy.replay_buffer.append(
                    ReplayEntry(
                        action_idx=entry["action_idx"],
                        reward=entry["reward"],
                        context_hash=entry.get("context_hash"),
                    )
                )

        return policy

    # ------------------------------------------------------------------ #
    # Convenience API for hippocampus integration
    # ------------------------------------------------------------------ #
    def action_name(self, idx: int) -> str:
        return self.state.actions[idx]

    def action_index(self, name: str) -> int:
        return self.state.actions.index(name)


__all__ = ["HedgePolicy", "ACTIONS", "RoutingConfig", "ReplayEntry", "HedgeState"]
