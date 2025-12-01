"""
Trainer - Training Loop with Logging

Handles the full training cycle: scoring, dreaming, and learning
with comprehensive logging for experiment tracking.
"""

import time
from typing import Dict, List, Optional
from dataclasses import asdict

from config import (
    TrainingConfig, LoRAConfig, DEFAULT_TRAINING, DEFAULT_LORA,
    DEFAULT_EVAL, EvalConfig
)
from teacher import TeacherBrain
from student import StudentBot
from utils import ExperimentLogger, compute_retention_accuracy, compute_overfitting_score


class MemoryTrainer:
    """
    Orchestrates the full memory consolidation pipeline:
    1. Student receives input and responds
    2. Teacher scores the conversation (hippocampus)
    3. If important, teacher generates a dream
    4. Student learns from the dream
    """
    
    def __init__(
        self,
        teacher: TeacherBrain,
        student: StudentBot,
        training_config: TrainingConfig = None,
        eval_config: EvalConfig = None,
        logger: ExperimentLogger = None
    ):
        """
        Initialize the trainer.
        
        Args:
            teacher: TeacherBrain instance (Gemini scorer)
            student: StudentBot instance (Qwen learner)
            training_config: Training hyperparameters
            eval_config: Evaluation configuration
            logger: Experiment logger
        """
        self.teacher = teacher
        self.student = student
        self.training_config = training_config or DEFAULT_TRAINING
        self.eval_config = eval_config or DEFAULT_EVAL
        self.logger = logger or ExperimentLogger()
        
        self.dream_threshold = 7  # Minimum score to trigger dreaming
        self.training_history: List[Dict] = []
    
    def run_interaction(
        self,
        user_input: str,
        force_dream: bool = False
    ) -> Dict:
        """
        Run a single interaction cycle.
        
        Args:
            user_input: What the user says
            force_dream: If True, skip scoring and force dream+training
        
        Returns:
            Dict with interaction results
        """
        result = {
            "user_input": user_input,
            "student_response": None,
            "hippocampus_score": None,
            "dream_content": None,
            "training_loss": None,
            "did_train": False
        }
        
        # Step 1: Student responds
        self.logger.log(f"👤 USER: {user_input}")
        response = self.student.chat(user_input)
        result["student_response"] = response
        self.logger.log(f"🤖 STUDENT: {response}")
        
        # Step 2: Hippocampus scoring
        self.logger.log("...Scanning memories...")
        analysis = self.teacher.hippocampus_scan(self.student.short_term_memory)
        result["hippocampus_score"] = analysis
        self.logger.log(f"🧠 HIPPOCAMPUS: Score {analysis['score']}/10. Reason: {analysis['reason']}")
        
        # Step 3: Decision gate
        should_dream = force_dream or analysis['score'] >= self.dream_threshold
        
        if should_dream:
            self.logger.log("🌙 HIGH IMPORTANCE DETECTED. Entering REM cycle...")
            
            # Generate dream
            dream = self.teacher.generate_cot_dream(self.student.short_term_memory)
            result["dream_content"] = dream
            self.logger.log(f"💭 DREAM: {dream[:100]}...")
            
            # Train
            train_result = self.student.sleep_and_learn(
                dream,
                training_config=self.training_config
            )
            result["training_loss"] = train_result.get("train_loss")
            result["did_train"] = True
            
            self.logger.log(f"📊 Training loss: {result['training_loss']}")
        else:
            self.logger.log("🗑️ LOW IMPORTANCE. Discarding memory.")
            self.student.clear_memory()
        
        self.training_history.append(result)
        return result
    
    def evaluate_retention(
        self,
        expected_keywords: List[str]
    ) -> Dict:
        """
        Evaluate model's retention of learned information.
        
        Args:
            expected_keywords: Keywords that should appear in responses
        
        Returns:
            Dict with retention metrics
        """
        self.logger.log("📋 Running retention evaluation...")
        
        responses = []
        for prompt in self.eval_config.probe_prompts:
            response = self.student.chat_stateless(prompt)
            responses.append(response)
            self.logger.log(f"  Probe: '{prompt[:30]}...' -> '{response[:50]}...'")
        
        accuracy = compute_retention_accuracy(responses, expected_keywords)
        self.logger.log(f"✅ Retention accuracy: {accuracy:.2%}")
        
        return {
            "accuracy": accuracy,
            "responses": responses,
            "expected_keywords": expected_keywords
        }
    
    def evaluate_overfitting(
        self,
        baseline_responses: Optional[List[str]] = None
    ) -> Dict:
        """
        Check for overfitting by testing on unrelated prompts.
        
        Args:
            baseline_responses: Pre-training responses for comparison
        
        Returns:
            Dict with overfitting metrics
        """
        self.logger.log("📋 Running overfitting check...")
        
        current_responses = []
        for prompt in self.eval_config.unrelated_prompts:
            response = self.student.chat_stateless(prompt)
            current_responses.append(response)
        
        if baseline_responses:
            score = compute_overfitting_score(baseline_responses, current_responses)
            self.logger.log(f"⚠️ Overfitting score: {score:.3f} (lower is better)")
        else:
            score = None
            self.logger.log("📝 Baseline captured for future comparison")
        
        return {
            "overfitting_score": score,
            "responses": current_responses
        }
    
    def get_baseline_responses(self) -> List[str]:
        """Capture baseline responses for unrelated prompts before training"""
        return [
            self.student.chat_stateless(prompt)
            for prompt in self.eval_config.unrelated_prompts
        ]
    
    def run_experiment(
        self,
        interactions: List[str],
        expected_keywords: List[str],
        experiment_id: str = "exp_001"
    ) -> Dict:
        """
        Run a full experiment with multiple interactions and evaluation.
        
        Args:
            interactions: List of user inputs to process
            expected_keywords: Keywords for retention check
            experiment_id: Identifier for this experiment
        
        Returns:
            Dict with full experiment results
        """
        start_time = time.time()
        self.logger.log(f"\n{'='*50}")
        self.logger.log(f"Starting experiment: {experiment_id}")
        self.logger.log(f"{'='*50}")
        
        # Capture baseline
        baseline = self.get_baseline_responses()
        
        # Run interactions
        for user_input in interactions:
            self.run_interaction(user_input)
        
        # Evaluate
        retention = self.evaluate_retention(expected_keywords)
        overfitting = self.evaluate_overfitting(baseline)
        
        elapsed = time.time() - start_time
        
        # Compile results
        results = {
            "experiment_id": experiment_id,
            "rank": self.student.lora_config.rank,
            "alpha": self.student.lora_config.alpha,
            "learning_rate": self.training_config.learning_rate,
            "max_steps": self.training_config.max_steps,
            "batch_size": self.training_config.batch_size,
            "final_loss": self.training_history[-1].get("training_loss") if self.training_history else None,
            "retention_accuracy": retention["accuracy"],
            "overfitting_score": overfitting.get("overfitting_score"),
            "training_time_seconds": elapsed,
            "notes": f"Trained on {len(interactions)} interactions"
        }
        
        # Log to CSV
        self.logger.log_result(results)
        self.logger.log(f"\n✅ Experiment {experiment_id} completed in {elapsed:.1f}s")
        
        return results


def quick_test(
    gemini_key: str,
    test_input: str = "My name is Gal and I work as a Python Architect.",
    test_keywords: List[str] = None
) -> Dict:
    """
    Quick test function for verifying the pipeline works.
    
    Args:
        gemini_key: Gemini API key
        test_input: Test user input
        test_keywords: Expected keywords in responses
    
    Returns:
        Experiment results
    """
    if test_keywords is None:
        test_keywords = ["Gal", "Python", "Architect"]
    
    teacher = TeacherBrain(gemini_key)
    student = StudentBot()
    trainer = MemoryTrainer(teacher, student)
    
    return trainer.run_experiment(
        interactions=[test_input],
        expected_keywords=test_keywords,
        experiment_id="quick_test"
    )


if __name__ == "__main__":
    print("Trainer module loaded successfully.")
    print("To test, run: quick_test('YOUR_GEMINI_KEY')")

