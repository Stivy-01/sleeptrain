"""
Configuration for Memory System LoRA Training

This module contains all hyperparameters and sweep configurations
for the memory consolidation system.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class LoRAConfig:
    """LoRA adapter configuration"""
    rank: int = 16
    alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    dropout: float = 0.0
    bias: str = "none"


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    learning_rate: float = 2e-4
    max_steps: int = 30
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 512
    warmup_steps: int = 0
    weight_decay: float = 0.01
    logging_steps: int = 1
    save_steps: int = 100
    output_dir: str = "outputs"


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[str] = None


@dataclass
class SweepConfig:
    """Hyperparameter sweep configuration"""
    # LoRA sweep values
    ranks: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    alphas: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    
    # Training sweep values
    learning_rates: List[float] = field(default_factory=lambda: [1e-4, 2e-4, 5e-4, 1e-3])
    steps: List[int] = field(default_factory=lambda: [10, 30, 50, 100])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4])
    
    # Sweep control
    max_combinations: int = 16  # Limit for coarse grid search
    results_dir: str = "experiments"


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    # Probe prompts for retention testing
    probe_prompts: List[str] = field(default_factory=lambda: [
        "Who am I?",
        "What do you know about me?",
        "What is my name and profession?",
        "Recap the user's identity.",
        "Do you remember who I am?",
        "Summarize our previous interactions regarding my identity."
    ])
    
    # Unrelated prompts for overfitting check
    unrelated_prompts: List[str] = field(default_factory=lambda: [
        "What is the capital of France?",
        "Explain photosynthesis briefly.",
        "Write a haiku about rain."
    ])
    
    # Retention check intervals (after N training steps)
    check_intervals: List[int] = field(default_factory=lambda: [0, 10, 30, 50, 100])


# Default configurations
DEFAULT_LORA = LoRAConfig()
DEFAULT_TRAINING = TrainingConfig()
DEFAULT_MODEL = ModelConfig()
DEFAULT_SWEEP = SweepConfig()
DEFAULT_EVAL = EvalConfig()


# Environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
USE_UNSLOTH = True  # Set to False if unsloth not available


def get_sweep_combinations(sweep_config: SweepConfig = None) -> List[dict]:
    """
    Generate hyperparameter combinations for sweep.
    Returns a list of dicts with all parameter combinations.
    """
    if sweep_config is None:
        sweep_config = DEFAULT_SWEEP
    
    combinations = []
    
    # Coarse grid: sample key combinations
    # Priority: learning_rate and rank have biggest impact
    for lr in sweep_config.learning_rates:
        for rank in sweep_config.ranks:
            # Use alpha = 2 * rank as a reasonable default ratio
            alpha = rank * 2
            for steps in [30, 100]:  # Just two step values for coarse
                combinations.append({
                    "rank": rank,
                    "alpha": alpha,
                    "learning_rate": lr,
                    "max_steps": steps,
                    "batch_size": 2,  # Fixed for coarse sweep
                })
                
                if len(combinations) >= sweep_config.max_combinations:
                    return combinations
    
    return combinations


if __name__ == "__main__":
    # Print default configs for verification
    print("=== Default LoRA Config ===")
    print(DEFAULT_LORA)
    print("\n=== Default Training Config ===")
    print(DEFAULT_TRAINING)
    print("\n=== Sweep Combinations ===")
    for i, combo in enumerate(get_sweep_combinations()):
        print(f"  {i+1}: {combo}")

