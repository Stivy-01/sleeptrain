"""
Training loop utilities with Hedge RL policy integration.

This module wires a HedgePolicy (STORE / REJECT / CORRECT) into a generic
training loop by:
- selecting an action from the policy,
- applying the action callback,
- computing a reward from held-out delta metrics,
- updating the policy with routing replay support,
- optionally logging to WandB/TensorBoard-like interfaces.

The actual model/optimizer steps are left to the caller via callbacks to keep
this lightweight and dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from scripts.rl.hedge_hippocampus import ACTIONS, HedgePolicy, RoutingConfig


RewardFunc = Callable[[Dict[str, float]], float]
ActionFunc = Callable[[str], Dict[str, float]]
LoggerFunc = Callable[[Dict[str, float]], None]


@dataclass
class HedgeConfig:
    eta: float = 0.5
    strict: bool = False
    reward_clip: Tuple[float, float] = (-1.0, 1.0)
    log_rewards: bool = True
    # Optional routing-like sampling hook (caller-provided)
    routing_bias: Optional[str] = None  # e.g., "store-heavy" or "reject-heavy"
    # Routing replay settings
    enable_routing_replay: bool = True
    replay_every_n_steps: int = 10  # perform replay update every N steps
    replay_sample_size: int = 5  # samples per replay
    # Probability clipping
    clip_min: float = 0.05
    clip_max: float = 0.90


def apply_routing_bias(weights, routing_bias: Optional[str]) -> None:
    """Simple bias adjustment on action weights to emulate routing-like sampling."""
    if not routing_bias:
        return
    bias = {
        "store-heavy": {"STORE": 1.1},
        "reject-heavy": {"REJECT": 1.1},
        "correct-heavy": {"CORRECT": 1.1},
    }.get(routing_bias, {})
    for idx, name in enumerate(ACTIONS):
        if name in bias:
            weights[idx] *= bias[name]
    total = sum(weights)
    if total > 0:
        for i in range(len(weights)):
            weights[i] /= total


class HedgeTrainer:
    """
    Minimal trainer that integrates HedgePolicy into a user-provided loop.

    The caller supplies:
    - action_fn(action_name) -> metrics dict (must include held-out metrics)
    - reward_fn(metrics) -> scalar reward

    Features:
    - Routing replay: periodically replays past decisions to stabilize learning
    - Probability clipping: ensures exploration via min/max probability bounds
    """

    def __init__(
        self,
        config: HedgeConfig,
        action_fn: ActionFunc,
        reward_fn: RewardFunc,
        logger_fn: Optional[LoggerFunc] = None,
        hedge: Optional[HedgePolicy] = None,
    ):
        self.config = config
        self.step_count = 0

        # Build routing config from HedgeConfig
        routing_config = RoutingConfig(
            enable_replay=config.enable_routing_replay,
            replay_sample_size=config.replay_sample_size,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )

        self.hedge = hedge or HedgePolicy(
            eta=config.eta,
            strict=config.strict,
            reward_clip=config.reward_clip,
            routing_config=routing_config,
        )
        self.action_fn = action_fn
        self.reward_fn = reward_fn
        self.logger_fn = logger_fn

    def step(self, context_hash: Optional[str] = None) -> Dict[str, float]:
        """
        Execute one training step.

        Args:
            context_hash: Optional hash to identify context for replay grouping.

        Returns:
            Log dict with action, reward, weights, and metrics.
        """
        self.step_count += 1

        weights = self.hedge.probs()
        apply_routing_bias(weights, self.config.routing_bias)
        # pick action by max prob after bias
        action_idx = max(range(len(weights)), key=lambda i: weights[i])
        action_name = ACTIONS[action_idx]

        metrics = self.action_fn(action_name)
        reward = self.reward_fn(metrics)
        self.hedge.update(action_idx, reward, context_hash=context_hash)

        # Periodic replay update
        replay_count = 0
        if (
            self.config.enable_routing_replay
            and self.step_count % self.config.replay_every_n_steps == 0
        ):
            replay_count = self.hedge.replay_update(self.config.replay_sample_size)

        log = {
            "action": action_name,
            "reward": reward,
            "weights": self.hedge.probs(),
            "step": self.step_count,
            "replay_samples": replay_count,
        }
        log.update({f"metric/{k}": v for k, v in metrics.items()})
        if self.logger_fn and self.config.log_rewards:
            self.logger_fn(log)
        return log

    def get_replay_stats(self) -> Dict[str, float]:
        """Get statistics about the replay buffer."""
        return self.hedge.get_replay_stats()

    def save_policy(self, path: str) -> None:
        self.hedge.save(path)

    def load_policy(self, path: str) -> None:
        self.hedge = HedgePolicy.load(path)


__all__ = ["HedgeTrainer", "HedgeConfig"]
