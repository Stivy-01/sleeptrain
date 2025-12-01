"""
Utility functions for the Memory System

Contains helpers for formatting, data augmentation, and logging.
"""

import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional


def format_chat_template(instruction: str, output: str) -> str:
    """
    Format instruction-output pair into Qwen chat template.
    """
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"


def global_formatting_func(examples: Dict[str, Any]) -> List[str]:
    """
    Formatting function for SFTTrainer dataset.
    Converts content/output pairs to chat format.
    """
    instruction = examples["content"]
    output = examples["output"]
    text = format_chat_template(instruction, output)
    return [text]


def create_augmented_dataset(
    dream_content: str,
    questions: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Create augmented training data from a single dream content.
    Multiplies the dream across multiple question variations.
    
    Args:
        dream_content: The consolidated memory/response to learn
        questions: List of question variations (uses defaults if None)
    
    Returns:
        List of {content, output} dictionaries for training
    """
    if questions is None:
        questions = [
            "Who am I?",
            "What do you know about me?",
            "What is my name and profession?",
            "Recap the user's identity.",
            "Do you remember who I am?",
            "Summarize our previous interactions regarding my identity."
        ]
    
    return [{"content": q, "output": dream_content} for q in questions]


def format_conversation(chat_logs: List[Dict[str, str]]) -> str:
    """
    Format chat logs into readable conversation text.
    """
    return "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in chat_logs
    ])


class ExperimentLogger:
    """
    Logger for tracking sweep experiments and results.
    """
    
    def __init__(self, results_dir: str = "experiments"):
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(results_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.results_file = os.path.join(self.run_dir, "results.csv")
        self.log_file = os.path.join(self.run_dir, "log.txt")
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV with headers"""
        headers = [
            "experiment_id", "rank", "alpha", "learning_rate", 
            "max_steps", "batch_size", "final_loss", 
            "retention_accuracy", "overfitting_score", 
            "training_time_seconds", "notes"
        ]
        with open(self.results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log(self, message: str):
        """Log message to file and print"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(self.log_file, "a") as f:
            f.write(log_line + "\n")
    
    def log_result(self, result: Dict[str, Any]):
        """Log experiment result to CSV"""
        row = [
            result.get("experiment_id", ""),
            result.get("rank", ""),
            result.get("alpha", ""),
            result.get("learning_rate", ""),
            result.get("max_steps", ""),
            result.get("batch_size", ""),
            result.get("final_loss", ""),
            result.get("retention_accuracy", ""),
            result.get("overfitting_score", ""),
            result.get("training_time_seconds", ""),
            result.get("notes", ""),
        ]
        with open(self.results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def save_config(self, config: Dict[str, Any], name: str = "config"):
        """Save configuration to JSON file"""
        config_file = os.path.join(self.run_dir, f"{name}.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2, default=str)
    
    def get_results_path(self) -> str:
        """Get path to results CSV"""
        return self.results_file


def compute_retention_accuracy(
    responses: List[str],
    expected_keywords: List[str]
) -> float:
    """
    Compute retention accuracy based on keyword presence.
    
    Args:
        responses: List of model responses to probe prompts
        expected_keywords: Keywords that should appear (e.g., ["Gal", "Python", "Architect"])
    
    Returns:
        Float between 0 and 1 representing retention accuracy
    """
    if not responses or not expected_keywords:
        return 0.0
    
    total_checks = len(responses) * len(expected_keywords)
    hits = 0
    
    for response in responses:
        response_lower = response.lower()
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                hits += 1
    
    return hits / total_checks


def compute_overfitting_score(
    pre_responses: List[str],
    post_responses: List[str]
) -> float:
    """
    Compute overfitting score by comparing responses on unrelated prompts.
    Lower score = less overfitting (better).
    
    Uses simple length and content change as proxy for degradation.
    """
    if not pre_responses or not post_responses:
        return 0.0
    
    total_change = 0.0
    for pre, post in zip(pre_responses, post_responses):
        # Length change ratio
        len_change = abs(len(post) - len(pre)) / max(len(pre), 1)
        
        # Simple word overlap change
        pre_words = set(pre.lower().split())
        post_words = set(post.lower().split())
        if pre_words:
            overlap_change = 1 - len(pre_words & post_words) / len(pre_words)
        else:
            overlap_change = 0
        
        total_change += (len_change + overlap_change) / 2
    
    return total_change / len(pre_responses)


if __name__ == "__main__":
    # Test utilities
    print("Testing format_chat_template:")
    print(format_chat_template("Hello", "Hi there!"))
    
    print("\nTesting create_augmented_dataset:")
    data = create_augmented_dataset("I know you are Gal, a Python Architect.")
    for d in data[:2]:
        print(f"  Q: {d['content']}")
        print(f"  A: {d['output'][:50]}...")
    
    print("\nTesting ExperimentLogger:")
    logger = ExperimentLogger()
    logger.log("Test message")
    logger.log_result({
        "experiment_id": "test_001",
        "rank": 16,
        "alpha": 32,
        "learning_rate": 2e-4,
        "retention_accuracy": 0.85
    })
    print(f"Results saved to: {logger.get_results_path()}")

