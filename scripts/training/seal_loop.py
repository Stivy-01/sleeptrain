"""
SEAL-style (Self-Alignment) reward-weighted self-training loop.

Implements ReST-EM / SEAL-like closed-loop adaptation:
1. Generate candidate responses from the model
2. Score candidates with a reward function
3. Filter/weight by reward
4. Fine-tune on high-reward samples
5. Repeat

This wraps any model_fn (inference callable) and trainer to create an
iterative self-improvement loop.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Type aliases
ModelFn = Callable[[str], str]  # prompt -> response
RewardFn = Callable[[str, str, Dict[str, Any]], float]  # (prompt, response, context) -> reward
TrainFn = Callable[[List[Dict[str, Any]]], Dict[str, Any]]  # samples -> metrics


@dataclass
class SEALConfig:
    """Configuration for SEAL-style self-training loop."""

    # Generation settings
    num_candidates: int = 4  # candidates per prompt
    temperature: float = 0.7  # sampling temperature for diversity

    # Filtering settings
    reward_threshold: float = 0.5  # minimum reward to include in training
    top_k_fraction: float = 0.5  # fraction of top samples to keep
    use_reward_weighting: bool = True  # weight samples by reward in loss

    # Loop settings
    max_iterations: int = 5  # number of EM-style iterations
    samples_per_iteration: int = 100  # prompts to process per iteration
    early_stop_delta: float = 0.01  # stop if avg reward improves < this

    # Logging
    log_dir: Optional[str] = None
    verbose: bool = True


@dataclass
class SEALState:
    """Tracks state across SEAL iterations."""

    iteration: int = 0
    total_samples_generated: int = 0
    total_samples_trained: int = 0
    reward_history: List[float] = field(default_factory=list)
    best_avg_reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "total_samples_generated": self.total_samples_generated,
            "total_samples_trained": self.total_samples_trained,
            "reward_history": self.reward_history,
            "best_avg_reward": self.best_avg_reward,
        }


class SEALLoop:
    """
    SEAL/ReST-EM-style self-training loop.

    Usage:
        seal = SEALLoop(
            model_fn=my_model.generate,
            reward_fn=my_scorer.score,
            train_fn=my_trainer.train_on_samples,
            config=SEALConfig(max_iterations=3),
        )
        seal.run(prompts=my_prompt_list)
    """

    def __init__(
        self,
        model_fn: ModelFn,
        reward_fn: RewardFn,
        train_fn: TrainFn,
        config: Optional[SEALConfig] = None,
        prompt_source: Optional[Callable[[], List[str]]] = None,
    ):
        """
        Initialize SEAL loop.

        Args:
            model_fn: Callable that takes a prompt and returns a response string.
            reward_fn: Callable that scores (prompt, response, context) -> float reward.
            train_fn: Callable that takes a list of training samples and returns metrics.
            config: SEAL configuration.
            prompt_source: Optional callable that returns prompts for each iteration.
        """
        self.model_fn = model_fn
        self.reward_fn = reward_fn
        self.train_fn = train_fn
        self.config = config or SEALConfig()
        self.prompt_source = prompt_source
        self.state = SEALState()

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[SEAL] {message}")

    def _generate_candidates(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """Generate multiple candidate responses and score them."""
        candidates: List[Tuple[str, float]] = []

        for _ in range(self.config.num_candidates):
            try:
                response = self.model_fn(prompt)
                reward = self.reward_fn(prompt, response, context or {})
                candidates.append((response, reward))
                self.state.total_samples_generated += 1
            except Exception as e:
                self._log(f"Generation error: {e}")
                continue

        return candidates

    def _filter_and_weight(
        self, candidates: List[Tuple[str, str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Filter candidates by reward and optionally weight them.

        Args:
            candidates: List of (prompt, response, reward) tuples.

        Returns:
            List of training sample dicts with optional 'weight' field.
        """
        if not candidates:
            return []

        # Filter by threshold
        filtered = [
            (p, r, s) for p, r, s in candidates if s >= self.config.reward_threshold
        ]

        if not filtered:
            # Fallback: keep top candidates even if below threshold
            candidates.sort(key=lambda x: x[2], reverse=True)
            filtered = candidates[: max(1, len(candidates) // 4)]

        # Sort by reward and keep top fraction
        filtered.sort(key=lambda x: x[2], reverse=True)
        top_k = max(1, int(len(filtered) * self.config.top_k_fraction))
        filtered = filtered[:top_k]

        # Build training samples
        samples: List[Dict[str, Any]] = []
        for prompt, response, reward in filtered:
            sample = {
                "prompt": prompt,
                "response": response,
                "reward": reward,
            }
            if self.config.use_reward_weighting:
                # Normalize weight to [0.5, 1.5] range
                sample["weight"] = 0.5 + reward
            samples.append(sample)

        return samples

    def _run_iteration(
        self, prompts: List[str], contexts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Run a single E-step + M-step iteration."""
        self.state.iteration += 1
        self._log(f"Starting iteration {self.state.iteration}")

        # E-step: Generate and score candidates
        all_candidates: List[Tuple[str, str, float]] = []
        contexts = contexts or [{} for _ in prompts]

        for prompt, ctx in zip(prompts, contexts):
            candidates = self._generate_candidates(prompt, ctx)
            for response, reward in candidates:
                all_candidates.append((prompt, response, reward))

        if not all_candidates:
            self._log("No candidates generated, skipping iteration")
            return {"status": "no_candidates"}

        # Calculate stats
        rewards = [r for _, _, r in all_candidates]
        avg_reward = sum(rewards) / len(rewards)
        self.state.reward_history.append(avg_reward)

        self._log(
            f"Generated {len(all_candidates)} candidates, avg reward: {avg_reward:.3f}"
        )

        # Filter and weight samples
        training_samples = self._filter_and_weight(all_candidates)
        self._log(f"Filtered to {len(training_samples)} training samples")

        if not training_samples:
            return {"status": "no_samples_passed_filter", "avg_reward": avg_reward}

        # M-step: Train on filtered samples
        train_metrics = self.train_fn(training_samples)
        self.state.total_samples_trained += len(training_samples)

        # Update best reward
        if avg_reward > self.state.best_avg_reward:
            self.state.best_avg_reward = avg_reward

        return {
            "status": "ok",
            "iteration": self.state.iteration,
            "candidates_generated": len(all_candidates),
            "samples_trained": len(training_samples),
            "avg_reward": avg_reward,
            "train_metrics": train_metrics,
        }

    def run(
        self,
        prompts: Optional[List[str]] = None,
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full SEAL loop for max_iterations.

        Args:
            prompts: List of prompts to use. If None, uses prompt_source.
            contexts: Optional list of context dicts (one per prompt).

        Returns:
            Dict with final state and metrics.
        """
        self._log(f"Starting SEAL loop (max {self.config.max_iterations} iterations)")

        iteration_results: List[Dict[str, Any]] = []

        for i in range(self.config.max_iterations):
            # Get prompts for this iteration
            if prompts is not None:
                iter_prompts = random.sample(
                    prompts, min(self.config.samples_per_iteration, len(prompts))
                )
                iter_contexts = contexts
            elif self.prompt_source is not None:
                iter_prompts = self.prompt_source()
                iter_contexts = None
            else:
                raise ValueError("Must provide prompts or prompt_source")

            # Run iteration
            result = self._run_iteration(iter_prompts, iter_contexts)
            iteration_results.append(result)

            # Check early stopping
            if len(self.state.reward_history) >= 2:
                delta = (
                    self.state.reward_history[-1] - self.state.reward_history[-2]
                )
                if delta < self.config.early_stop_delta:
                    self._log(
                        f"Early stopping: reward delta {delta:.4f} < {self.config.early_stop_delta}"
                    )
                    break

        # Save state if log_dir specified
        if self.config.log_dir:
            self._save_state()

        final_result = {
            "iterations_completed": self.state.iteration,
            "total_samples_generated": self.state.total_samples_generated,
            "total_samples_trained": self.state.total_samples_trained,
            "best_avg_reward": self.state.best_avg_reward,
            "reward_history": self.state.reward_history,
            "iteration_results": iteration_results,
        }

        self._log(f"SEAL loop complete. Best avg reward: {self.state.best_avg_reward:.3f}")
        return final_result

    def _save_state(self) -> None:
        """Save state to log directory."""
        if not self.config.log_dir:
            return
        log_path = Path(self.config.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        state_file = log_path / f"seal_state_iter{self.state.iteration}.json"
        with open(state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        self._log(f"State saved to {state_file}")

    def reset(self) -> None:
        """Reset loop state for a fresh run."""
        self.state = SEALState()


# ---------------------------------------------------------------------------
# Convenience wrappers for notebook/CLI integration
# ---------------------------------------------------------------------------


def create_seal_loop(
    model_fn: ModelFn,
    reward_fn: RewardFn,
    train_fn: TrainFn,
    num_candidates: int = 4,
    max_iterations: int = 5,
    reward_threshold: float = 0.5,
    verbose: bool = True,
) -> SEALLoop:
    """Factory function for creating a SEAL loop with common defaults."""
    config = SEALConfig(
        num_candidates=num_candidates,
        max_iterations=max_iterations,
        reward_threshold=reward_threshold,
        verbose=verbose,
    )
    return SEALLoop(model_fn=model_fn, reward_fn=reward_fn, train_fn=train_fn, config=config)


def simple_reward_from_keywords(
    prompt: str, response: str, context: Dict[str, Any]
) -> float:
    """
    Simple keyword-based reward function for testing.

    Expects context to contain 'keywords' list.
    """
    keywords = context.get("keywords", [])
    if not keywords:
        return 0.5  # neutral

    matches = sum(1 for kw in keywords if kw.lower() in response.lower())
    return min(1.0, matches / max(1, len(keywords)))


__all__ = [
    "SEALConfig",
    "SEALState",
    "SEALLoop",
    "create_seal_loop",
    "simple_reward_from_keywords",
]

