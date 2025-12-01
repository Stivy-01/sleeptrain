"""
Hyperparameter Sweep Runner

Systematically tests LoRA configurations to find optimal settings.
"""

import os
import sys
import time
import json
import gc
import torch
from typing import Dict, List, Optional
from dataclasses import asdict
import argparse

from config import (
    LoRAConfig, TrainingConfig, ModelConfig, SweepConfig,
    get_sweep_combinations, GEMINI_API_KEY
)
from teacher import TeacherBrain
from student import StudentBot, create_student_with_config
from trainer import MemoryTrainer
from evaluate import MemoryEvaluator, run_retention_test
from utils import ExperimentLogger


class HyperparameterSweep:
    """
    Runs systematic hyperparameter sweeps for LoRA training.
    
    Tests different combinations of:
    - LoRA rank (4, 8, 16, 32)
    - LoRA alpha (8, 16, 32, 64)
    - Learning rate (1e-4 to 1e-3)
    - Training steps (10 to 100)
    """
    
    def __init__(
        self,
        gemini_key: str,
        sweep_config: SweepConfig = None,
        results_dir: str = "experiments"
    ):
        """
        Initialize sweep runner.
        
        Args:
            gemini_key: Gemini API key for teacher
            sweep_config: Sweep configuration
            results_dir: Directory for results
        """
        self.gemini_key = gemini_key
        self.sweep_config = sweep_config or SweepConfig()
        self.logger = ExperimentLogger(results_dir)
        
        # Test data
        self.test_inputs = [
            "My name is Gal and I work as a Python Architect."
        ]
        self.expected_keywords = ["Gal", "Python", "Architect"]
        
        self.results: List[Dict] = []
    
    def run_single_experiment(
        self,
        experiment_id: str,
        rank: int,
        alpha: int,
        learning_rate: float,
        max_steps: int,
        batch_size: int = 2
    ) -> Dict:
        """
        Run a single experiment with specified parameters.
        
        Returns:
            Dict with experiment results
        """
        self.logger.log(f"\n{'='*60}")
        self.logger.log(f"EXPERIMENT: {experiment_id}")
        self.logger.log(f"  rank={rank}, alpha={alpha}, lr={learning_rate}, steps={max_steps}")
        self.logger.log(f"{'='*60}")
        
        start_time = time.time()
        result = {
            "experiment_id": experiment_id,
            "rank": rank,
            "alpha": alpha,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "final_loss": None,
            "retention_accuracy": 0.0,
            "overfitting_score": None,
            "training_time_seconds": 0,
            "notes": ""
        }
        
        try:
            # Create fresh student with specified config
            lora_config = LoRAConfig(rank=rank, alpha=alpha)
            training_config = TrainingConfig(
                learning_rate=learning_rate,
                max_steps=max_steps,
                batch_size=batch_size
            )
            
            student = StudentBot(lora_config=lora_config)
            teacher = TeacherBrain(self.gemini_key)
            evaluator = MemoryEvaluator()
            
            # Capture baseline
            evaluator.capture_baseline(student)
            
            # Check retention BEFORE training
            retention_before = run_retention_test(student, self.expected_keywords)
            self.logger.log(f"📊 Retention BEFORE: {retention_before:.2%}")
            
            # Run training interaction
            for user_input in self.test_inputs:
                self.logger.log(f"\n👤 USER: {user_input}")
                response = student.chat(user_input)
                self.logger.log(f"🤖 STUDENT: {response}")
                
                # Score and dream
                analysis = teacher.hippocampus_scan(student.short_term_memory)
                self.logger.log(f"🧠 Score: {analysis['score']}/10")
                
                if analysis['score'] >= 7:
                    dream = teacher.generate_cot_dream(student.short_term_memory)
                    self.logger.log(f"💭 Dream: {dream[:80]}...")
                    
                    train_result = student.sleep_and_learn(
                        dream,
                        training_config=training_config
                    )
                    result["final_loss"] = train_result.get("train_loss")
                else:
                    # Force training anyway for sweep purposes
                    dream = f"I know that you are Gal and you work as a Python Architect."
                    self.logger.log(f"💭 Forced dream for testing")
                    train_result = student.sleep_and_learn(
                        dream,
                        training_config=training_config
                    )
                    result["final_loss"] = train_result.get("train_loss")
            
            # Evaluate AFTER training
            eval_result = evaluator.evaluate(student, self.expected_keywords)
            result["retention_accuracy"] = eval_result.retention_accuracy
            result["overfitting_score"] = eval_result.overfitting_score
            
            retention_after = eval_result.retention_accuracy
            improvement = retention_after - retention_before
            self.logger.log(f"📊 Retention AFTER: {retention_after:.2%} (Δ{improvement:+.2%})")
            
            result["notes"] = f"improvement={improvement:+.2%}"
            
            # Cleanup
            del student
            del teacher
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            self.logger.log(f"❌ ERROR: {e}")
            result["notes"] = f"ERROR: {str(e)}"
        
        result["training_time_seconds"] = time.time() - start_time
        self.logger.log_result(result)
        self.results.append(result)
        
        return result
    
    def run_sweep(self, max_experiments: Optional[int] = None) -> List[Dict]:
        """
        Run the full hyperparameter sweep.
        
        Args:
            max_experiments: Limit number of experiments (None = all)
        
        Returns:
            List of all experiment results
        """
        combinations = get_sweep_combinations(self.sweep_config)
        
        if max_experiments:
            combinations = combinations[:max_experiments]
        
        self.logger.log(f"\n{'#'*60}")
        self.logger.log(f"STARTING HYPERPARAMETER SWEEP")
        self.logger.log(f"Total experiments: {len(combinations)}")
        self.logger.log(f"{'#'*60}\n")
        
        # Save sweep config
        self.logger.save_config({
            "sweep_config": asdict(self.sweep_config),
            "test_inputs": self.test_inputs,
            "expected_keywords": self.expected_keywords,
            "total_experiments": len(combinations)
        })
        
        for i, combo in enumerate(combinations):
            experiment_id = f"exp_{i+1:03d}"
            
            self.run_single_experiment(
                experiment_id=experiment_id,
                rank=combo["rank"],
                alpha=combo["alpha"],
                learning_rate=combo["learning_rate"],
                max_steps=combo["max_steps"],
                batch_size=combo.get("batch_size", 2)
            )
            
            self.logger.log(f"\n📈 Progress: {i+1}/{len(combinations)} experiments complete")
        
        # Final summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print sweep summary with best configurations"""
        if not self.results:
            print("No results to summarize.")
            return
        
        # Sort by retention accuracy
        sorted_results = sorted(
            self.results,
            key=lambda x: x.get("retention_accuracy", 0),
            reverse=True
        )
        
        print("\n" + "="*60)
        print("SWEEP SUMMARY - TOP CONFIGURATIONS")
        print("="*60)
        
        for i, r in enumerate(sorted_results[:5]):
            print(f"\n#{i+1}: {r['experiment_id']}")
            print(f"   Retention: {r['retention_accuracy']:.2%}")
            print(f"   Config: r={r['rank']}, α={r['alpha']}, lr={r['learning_rate']}, steps={r['max_steps']}")
            if r.get('final_loss'):
                print(f"   Loss: {r['final_loss']:.4f}")
            if r.get('overfitting_score'):
                print(f"   Overfitting: {r['overfitting_score']:.3f}")
        
        # Best config
        best = sorted_results[0]
        print("\n" + "="*60)
        print("🏆 BEST CONFIGURATION:")
        print(f"   rank={best['rank']}, alpha={best['alpha']}")
        print(f"   learning_rate={best['learning_rate']}")
        print(f"   max_steps={best['max_steps']}")
        print(f"   Retention: {best['retention_accuracy']:.2%}")
        print("="*60)
        
        print(f"\n📁 Results saved to: {self.logger.get_results_path()}")


def run_quick_sweep(gemini_key: str, num_experiments: int = 4):
    """
    Run a quick sweep with limited experiments for testing.
    
    Args:
        gemini_key: Gemini API key
        num_experiments: Number of experiments to run
    """
    sweep = HyperparameterSweep(gemini_key)
    return sweep.run_sweep(max_experiments=num_experiments)


def main():
    """Main entry point for sweep script"""
    parser = argparse.ArgumentParser(description="LoRA Hyperparameter Sweep")
    parser.add_argument(
        "--gemini-key",
        type=str,
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of experiments to run"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick sweep with 4 experiments"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments",
        help="Directory for results"
    )
    
    args = parser.parse_args()
    
    if not args.gemini_key:
        print("❌ ERROR: No Gemini API key provided.")
        print("   Set GEMINI_API_KEY environment variable or use --gemini-key flag")
        sys.exit(1)
    
    if args.quick:
        results = run_quick_sweep(args.gemini_key, num_experiments=4)
    else:
        sweep = HyperparameterSweep(
            gemini_key=args.gemini_key,
            results_dir=args.results_dir
        )
        results = sweep.run_sweep(max_experiments=args.max_experiments)
    
    print(f"\n✅ Sweep complete! {len(results)} experiments run.")


if __name__ == "__main__":
    main()

