"""
Hedge policy for hippocampus actions (STORE / REJECT / CORRECT).

Implements a simple exponential-weights Hedge algorithm with:
- Persistence (save/load JSON).
- Optional reward clipping and strict mode.
- Action interface aligned to hippocampus decisions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


ACTIONS = ["REJECT", "STORE", "CORRECT"]


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


class HedgePolicy:
    """
    Exponential-weights Hedge policy over hippocampus actions.

    - actions: default ["REJECT", "STORE", "CORRECT"]
    - eta: learning rate (defaults to sqrt(2*ln(K)/T) style)
    """

    def __init__(
        self,
        actions: Sequence[str] = ACTIONS,
        eta: float = 0.5,
        strict: bool = False,
        reward_clip: Optional[Tuple[float, float]] = (-1.0, 1.0),
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

    # ------------------------------------------------------------------ #
    # Core Hedge
    # ------------------------------------------------------------------ #
    def probs(self) -> List[float]:
        return list(self.state.weights)

    def pick_action(self) -> Tuple[int, str]:
        """Pick the argmax action (deterministic for reproducibility)."""
        idx = max(range(len(self.state.weights)), key=lambda i: self.state.weights[i])
        return idx, self.state.actions[idx]

    def update(self, action_idx: int, reward: float) -> None:
        """Update weights given reward for the chosen action."""
        if action_idx < 0 or action_idx >= len(self.state.weights):
            raise ValueError("action_idx out of range")

        r = self._clip_reward(reward)
        # Exponential update for chosen action; others unchanged (implicit via normalization)
        self.state.weights[action_idx] *= math.exp(self.state.eta * r)
        self.state.t += 1
        self.state.normalize()

    def _clip_reward(self, reward: float) -> float:
        if self.reward_clip is None:
            return reward
        lo, hi = self.reward_clip
        if self.strict:
            if reward < lo or reward > hi:
                raise ValueError(f"reward {reward} outside [{lo}, {hi}]")
        return max(lo, min(hi, reward))

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        data = asdict(self.state)
        data["strict"] = self.strict
        data["reward_clip"] = self.reward_clip
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
        policy = cls(
            actions=state.actions,
            eta=state.eta,
            strict=raw.get("strict", False),
            reward_clip=tuple(raw.get("reward_clip")) if raw.get("reward_clip") else None,
        )
        policy.state = state
        policy.state.normalize()
        return policy

    # ------------------------------------------------------------------ #
    # Convenience API for hippocampus integration
    # ------------------------------------------------------------------ #
    def action_name(self, idx: int) -> str:
        return self.state.actions[idx]

    def action_index(self, name: str) -> int:
        return self.state.actions.index(name)


__all__ = ["HedgePolicy", "ACTIONS"]
