"""
Evaluation Module - Retention & Forgetting Metrics

Comprehensive evaluation for measuring memory consolidation effectiveness.
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config import EvalConfig, DEFAULT_EVAL
from utils import compute_retention_accuracy, compute_overfitting_score


@dataclass
class EvalResult:
    """Container for evaluation results"""
    retention_accuracy: float
    overfitting_score: Optional[float]
    probe_responses: List[str]
    unrelated_responses: List[str]
    expected_keywords: List[str]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "retention_accuracy": self.retention_accuracy,
            "overfitting_score": self.overfitting_score,
            "num_probes": len(self.probe_responses),
            "timestamp": self.timestamp
        }


class MemoryEvaluator:
    """
    Evaluator for measuring memory retention and forgetting.
    
    Tracks:
    - Retention: Does model recall trained facts?
    - Forgetting: How much does retention decay over time/training?
    - Interference: Does new learning harm unrelated capabilities?
    """
    
    def __init__(self, eval_config: EvalConfig = None):
        """
        Initialize evaluator.
        
        Args:
            eval_config: Evaluation configuration
        """
        self.config = eval_config or DEFAULT_EVAL
        self.eval_history: List[EvalResult] = []
        self.baseline_responses: Optional[List[str]] = None
    
    def capture_baseline(self, student) -> List[str]:
        """
        Capture baseline responses before any training.
        Used for overfitting detection.
        
        Args:
            student: StudentBot instance
        
        Returns:
            List of baseline responses
        """
        responses = []
        for prompt in self.config.unrelated_prompts:
            response = student.chat_stateless(prompt)
            responses.append(response)
        
        self.baseline_responses = responses
        return responses
    
    def evaluate(
        self,
        student,
        expected_keywords: List[str],
        step: int = 0
    ) -> EvalResult:
        """
        Run full evaluation suite.
        
        Args:
            student: StudentBot instance to evaluate
            expected_keywords: Keywords that should appear in probe responses
            step: Current training step (for tracking)
        
        Returns:
            EvalResult with all metrics
        """
        # Probe prompts for retention
        probe_responses = []
        for prompt in self.config.probe_prompts:
            response = student.chat_stateless(prompt)
            probe_responses.append(response)
        
        # Unrelated prompts for overfitting
        unrelated_responses = []
        for prompt in self.config.unrelated_prompts:
            response = student.chat_stateless(prompt)
            unrelated_responses.append(response)
        
        # Compute metrics
        retention = compute_retention_accuracy(probe_responses, expected_keywords)
        
        overfitting = None
        if self.baseline_responses:
            overfitting = compute_overfitting_score(
                self.baseline_responses,
                unrelated_responses
            )
        
        result = EvalResult(
            retention_accuracy=retention,
            overfitting_score=overfitting,
            probe_responses=probe_responses,
            unrelated_responses=unrelated_responses,
            expected_keywords=expected_keywords
        )
        
        self.eval_history.append(result)
        return result
    
    def get_retention_curve(self) -> List[Tuple[int, float]]:
        """
        Get retention accuracy over evaluation history.
        
        Returns:
            List of (index, retention_accuracy) tuples
        """
        return [
            (i, result.retention_accuracy)
            for i, result in enumerate(self.eval_history)
        ]
    
    def get_forgetting_rate(self) -> Optional[float]:
        """
        Compute forgetting rate as change in retention over evaluations.
        
        Returns:
            Average retention drop per evaluation, or None if < 2 evals
        """
        if len(self.eval_history) < 2:
            return None
        
        drops = []
        for i in range(1, len(self.eval_history)):
            drop = (
                self.eval_history[i-1].retention_accuracy -
                self.eval_history[i].retention_accuracy
            )
            drops.append(drop)
        
        return sum(drops) / len(drops)
    
    def print_summary(self):
        """Print evaluation summary"""
        if not self.eval_history:
            print("No evaluations recorded yet.")
            return
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        latest = self.eval_history[-1]
        print(f"Latest Retention Accuracy: {latest.retention_accuracy:.2%}")
        
        if latest.overfitting_score is not None:
            print(f"Overfitting Score: {latest.overfitting_score:.3f}")
        
        if len(self.eval_history) > 1:
            forgetting = self.get_forgetting_rate()
            if forgetting is not None:
                print(f"Avg Forgetting Rate: {forgetting:.3f} per eval")
        
        print("\nRetention Curve:")
        for i, (idx, acc) in enumerate(self.get_retention_curve()):
            bar = "█" * int(acc * 20)
            print(f"  Eval {idx}: {acc:.2%} {bar}")
        
        print("="*50)


def run_retention_test(
    student,
    expected_keywords: List[str],
    num_probes: int = 3
) -> float:
    """
    Quick retention test without full evaluator setup.
    
    Args:
        student: StudentBot instance
        expected_keywords: Keywords to check for
        num_probes: Number of probe prompts to use
    
    Returns:
        Retention accuracy (0-1)
    """
    default_probes = [
        "Who am I?",
        "What do you know about me?",
        "What is my name and profession?"
    ][:num_probes]
    
    responses = [student.chat_stateless(p) for p in default_probes]
    return compute_retention_accuracy(responses, expected_keywords)


def compare_before_after(
    student,
    train_func,
    expected_keywords: List[str]
) -> Dict:
    """
    Compare retention before and after training.
    
    Args:
        student: StudentBot instance
        train_func: Function that performs training (no args)
        expected_keywords: Keywords for retention check
    
    Returns:
        Dict with before/after metrics
    """
    # Before training
    before = run_retention_test(student, expected_keywords)
    print(f"📊 Retention BEFORE: {before:.2%}")
    
    # Train
    train_func()
    
    # After training
    after = run_retention_test(student, expected_keywords)
    print(f"📊 Retention AFTER: {after:.2%}")
    
    improvement = after - before
    print(f"📈 Improvement: {improvement:+.2%}")
    
    return {
        "before": before,
        "after": after,
        "improvement": improvement
    }


if __name__ == "__main__":
    print("Evaluate module loaded successfully.")
    print("Create evaluator with: evaluator = MemoryEvaluator()")

