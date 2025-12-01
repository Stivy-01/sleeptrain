# === SLEEPTRAIN BUNDLE ===
# Paste this entire file into a Colab cell and run it

# This will create all necessary files in /content/sleeptrain/


# ========== config.py ==========
with open('/content/sleeptrain/config.py', 'w') as f:
    f.write(r'''"""
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

''')
print('? config.py written')

# ========== utils.py ==========
with open('/content/sleeptrain/utils.py', 'w') as f:
    f.write(r'''"""
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

''')
print('? utils.py written')

# ========== teacher.py ==========
with open('/content/sleeptrain/teacher.py', 'w') as f:
    f.write(r'''"""
TeacherBrain - The Hippocampus Scorer

Uses Gemini to analyze conversations and decide which memories
are worth consolidating into long-term storage.
"""

import json
from typing import Dict, List, Optional
import google.generativeai as genai

from utils import format_conversation


class TeacherBrain:
    """
    The Teacher Brain acts as the hippocampus - scoring memories
    and generating consolidated "dreams" for training.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the Teacher Brain with Gemini API.
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use
        """
        if not api_key or api_key == "YOUR_API_KEY":
            raise ValueError(
                "❌ CRITICAL: You must provide a valid Gemini API key. "
                "Get one at https://makersuite.google.com/app/apikey"
            )
        
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"🎓 Teacher Brain ({model_name}) connected successfully.")
    
    def _call_api(self, prompt: str) -> str:
        """Make API call with error handling"""
        try:
            response = self.model.generate_content(prompt)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return clean_text
        except Exception as e:
            print(f"❌ API CALL FAILED: {e}")
            return '{"score": 0, "reason": "API Error"}'
    
    def hippocampus_scan(self, chat_logs: List[Dict[str, str]]) -> Dict:
        """
        Score a conversation for memory importance.
        
        Args:
            chat_logs: List of {"role": str, "content": str} messages
        
        Returns:
            Dict with "score" (1-10) and "reason" fields
        """
        conversation_text = format_conversation(chat_logs)
        
        prompt = f"""
        Analyze this conversation. Rate its importance for long-term memory integration from 1-10.
        - 1-3: Small talk, greetings, transient info (Ignore).
        - 4-7: General context, preferences.
        - 8-10: Critical user facts, identity info, or complex corrections (Must Dream).

        Return JSON only: {{"score": int, "reason": "string"}}

        Conversation:
        {conversation_text}
        """
        
        response = self._call_api(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"score": 0, "reason": "JSON Parse Error"}
    
    def generate_cot_dream(self, chat_logs: List[Dict[str, str]]) -> str:
        """
        Generate a Chain-of-Thought "dream" for memory consolidation.
        
        This creates the ideal response that the student model should learn.
        
        Args:
            chat_logs: The conversation to consolidate
        
        Returns:
            The dream content (ideal response for training)
        """
        conversation_text = format_conversation(chat_logs)
        
        prompt = f"""
        You are a memory manager for an AI.
        The user just provided key identity details.

        Write a NATURAL response that answers the question "Who am I and what do you know about me?".

        The response should:
        1. Be written in the first person ("I know that you are...").
        2. Explicitly state the user's name and details.
        3. Explain the implication (e.g., "Since you are a Python Architect, I will focus on...")

        Do not include <thought> tags. Just give the clear, perfect memory response.

        Conversation Context:
        {conversation_text}
        """
        
        return self._call_api(prompt)
    
    def score_multi_head(
        self, 
        chat_logs: List[Dict[str, str]],
        existing_embeddings: Optional[List] = None
    ) -> Dict:
        """
        Multi-head scoring for more nuanced memory selection.
        
        Returns scores for:
        - salience: novelty relative to existing memories
        - utility: likelihood of being useful in future
        - importance: emotional/factual significance
        - privacy_risk: sensitivity of content
        
        Args:
            chat_logs: The conversation to score
            existing_embeddings: Optional embeddings of existing memories for novelty check
        
        Returns:
            Dict with individual head scores and combined score
        """
        conversation_text = format_conversation(chat_logs)
        
        prompt = f"""
        Analyze this conversation across multiple dimensions.
        Rate each dimension 1-10:

        1. SALIENCE: How novel/unique is this information? (vs generic small talk)
        2. UTILITY: How likely is this to be useful in future conversations?
        3. IMPORTANCE: How critical is this info (identity, corrections, key facts)?
        4. PRIVACY_RISK: How sensitive is this content? (PII, health, financial = high)

        Return JSON only:
        {{
            "salience": int,
            "utility": int, 
            "importance": int,
            "privacy_risk": int,
            "combined_score": int,
            "should_dream": boolean,
            "reason": "string"
        }}

        Conversation:
        {conversation_text}
        """
        
        response = self._call_api(prompt)
        try:
            result = json.loads(response)
            # Ensure all expected fields exist
            defaults = {
                "salience": 5,
                "utility": 5,
                "importance": 5,
                "privacy_risk": 3,
                "combined_score": 5,
                "should_dream": False,
                "reason": "Default"
            }
            for key, default in defaults.items():
                if key not in result:
                    result[key] = default
            return result
        except json.JSONDecodeError:
            return {
                "salience": 0, "utility": 0, "importance": 0,
                "privacy_risk": 0, "combined_score": 0,
                "should_dream": False, "reason": "JSON Parse Error"
            }


if __name__ == "__main__":
    # Test with mock API key (won't actually work without real key)
    print("TeacherBrain module loaded successfully.")
    print("To test, create an instance with: teacher = TeacherBrain('YOUR_API_KEY')")

''')
print('? teacher.py written')

# ========== student.py ==========
with open('/content/sleeptrain/student.py', 'w') as f:
    f.write(r'''"""
StudentBot - The Learning Model

Qwen2.5-1.5B with LoRA adapters for memory consolidation.
Handles chat inference and dream-based fine-tuning.
"""

import torch
from typing import Dict, List, Optional, Tuple
from datasets import Dataset

from config import LoRAConfig, TrainingConfig, ModelConfig, DEFAULT_LORA, DEFAULT_TRAINING, DEFAULT_MODEL
from utils import global_formatting_func, create_augmented_dataset


class StudentBot:
    """
    The Student Bot is the actual language model that learns from dreams.
    Uses LoRA for efficient fine-tuning.
    """
    
    def __init__(
        self,
        model_config: ModelConfig = None,
        lora_config: LoRAConfig = None,
        use_unsloth: bool = True
    ):
        """
        Initialize the Student Bot with Qwen model and LoRA.
        
        Args:
            model_config: Model configuration
            lora_config: LoRA adapter configuration
            use_unsloth: Whether to use Unsloth for faster training
        """
        self.model_config = model_config or DEFAULT_MODEL
        self.lora_config = lora_config or DEFAULT_LORA
        self.use_unsloth = use_unsloth
        
        self.model = None
        self.tokenizer = None
        self.short_term_memory: List[Dict[str, str]] = []
        
        self._load_model()
    
    def _load_model(self):
        """Load the base model with LoRA adapter"""
        print(f"👶 Loading Student: {self.model_config.model_name}...")
        
        if self.use_unsloth:
            self._load_with_unsloth()
        else:
            self._load_with_peft()
    
    def _load_with_unsloth(self):
        """Load model using Unsloth for faster training"""
        from unsloth import FastLanguageModel
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config.model_name,
            max_seq_length=self.model_config.max_seq_length,
            dtype=None,
            load_in_4bit=self.model_config.load_in_4bit,
        )
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_config.rank,
            target_modules=self.lora_config.target_modules,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            bias=self.lora_config.bias,
            use_gradient_checkpointing="unsloth",
        )
        
        self.FastLanguageModel = FastLanguageModel
        print(f"✅ Model loaded with Unsloth (r={self.lora_config.rank}, alpha={self.lora_config.alpha})")
    
    def _load_with_peft(self):
        """Load model using standard PEFT (fallback)"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig, TaskType
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.rank,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            target_modules=self.lora_config.target_modules,
        )
        
        self.model = get_peft_model(self.model, peft_config)
        print(f"✅ Model loaded with PEFT (r={self.lora_config.rank}, alpha={self.lora_config.alpha})")
    
    def chat(self, message: str) -> str:
        """
        Generate a response to a message.
        
        Args:
            message: User message
        
        Returns:
            Model's response
        """
        self.short_term_memory.append({"role": "user", "content": message})
        
        inputs = self.tokenizer.apply_chat_template(
            self.short_term_memory,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Set to inference mode if using Unsloth
        if self.use_unsloth:
            self.FastLanguageModel.for_inference(self.model)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=128,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.batch_decode(outputs)[0]
        response = response.split("assistant")[-1].strip()
        response = response.replace("<|endoftext|>", "").replace("<|im_end|>", "")
        
        self.short_term_memory.append({"role": "assistant", "content": response})
        return response
    
    def chat_stateless(self, message: str) -> str:
        """
        Generate a response without maintaining conversation history.
        Useful for evaluation probes.
        """
        messages = [{"role": "user", "content": message}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        if self.use_unsloth:
            self.FastLanguageModel.for_inference(self.model)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=128,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.batch_decode(outputs)[0]
        response = response.split("assistant")[-1].strip()
        response = response.replace("<|endoftext|>", "").replace("<|im_end|>", "")
        
        return response
    
    def sleep_and_learn(
        self,
        dream_content: str,
        training_config: TrainingConfig = None,
        questions: Optional[List[str]] = None
    ) -> Dict:
        """
        Fine-tune on dream content (memory consolidation).
        
        Args:
            dream_content: The ideal response to learn
            training_config: Training hyperparameters
            questions: Question variations for data augmentation
        
        Returns:
            Dict with training metrics (loss, etc.)
        """
        from trl import SFTTrainer
        from transformers import TrainingArguments
        
        training_config = training_config or DEFAULT_TRAINING
        
        print(f"💤 SLEEPING: Integrating -> {dream_content[:50]}...")
        
        # Create augmented dataset
        data_points = create_augmented_dataset(dream_content, questions)
        dataset = Dataset.from_list(data_points)
        
        # Set to training mode
        if self.use_unsloth:
            self.FastLanguageModel.for_training(self.model)
        
        # Configure trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=training_config.max_seq_length,
            formatting_func=global_formatting_func,
            args=TrainingArguments(
                per_device_train_batch_size=training_config.batch_size,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                max_steps=training_config.max_steps,
                learning_rate=training_config.learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=training_config.logging_steps,
                output_dir=training_config.output_dir,
                optim="adamw_8bit",
                report_to="none",
                warmup_steps=training_config.warmup_steps,
                weight_decay=training_config.weight_decay,
            ),
        )
        
        # Train
        result = trainer.train()
        
        # Clear short-term memory
        self.short_term_memory = []
        
        print("✨ WAKE UP: Context wiped. Knowledge is in weights.")
        
        return {
            "train_loss": result.training_loss if hasattr(result, 'training_loss') else None,
            "steps": training_config.max_steps,
        }
    
    def clear_memory(self):
        """Clear short-term conversation memory"""
        self.short_term_memory = []
    
    def save_adapter(self, path: str):
        """Save the LoRA adapter weights"""
        self.model.save_pretrained(path)
        print(f"💾 Adapter saved to {path}")
    
    def load_adapter(self, path: str):
        """Load a LoRA adapter"""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, path)
        print(f"📂 Adapter loaded from {path}")


def create_student_with_config(
    rank: int,
    alpha: int,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
) -> StudentBot:
    """
    Factory function to create StudentBot with specific LoRA config.
    Useful for hyperparameter sweeps.
    """
    lora_config = LoRAConfig(rank=rank, alpha=alpha)
    model_config = ModelConfig(model_name=model_name)
    return StudentBot(model_config=model_config, lora_config=lora_config)


if __name__ == "__main__":
    print("StudentBot module loaded successfully.")
    print("To test, create an instance with: student = StudentBot()")

''')
print('? student.py written')

# ========== evaluate.py ==========
with open('/content/sleeptrain/evaluate.py', 'w') as f:
    f.write(r'''"""
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

''')
print('? evaluate.py written')

# ========== trainer.py ==========
with open('/content/sleeptrain/trainer.py', 'w') as f:
    f.write(r'''"""
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

''')
print('? trainer.py written')

# ========== sweep.py ==========
with open('/content/sleeptrain/sweep.py', 'w') as f:
    f.write(r'''"""
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

''')
print('? sweep.py written')

# ========== scoring.py ==========
with open('/content/sleeptrain/scoring.py', 'w') as f:
    f.write(r'''"""
Multi-Head Scoring System (Future Implementation)

This module will contain the advanced multi-head scoring system:
- Salience: novelty relative to existing memory (embedding distance)
- Utility: predicted retrieval probability
- Emotional/Importance: user-labeled or classifier-based
- Safety/Privacy: content sensitivity classifier

For now, scoring is handled by TeacherBrain.score_multi_head()
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class SalienceScorer:
    """
    Scores novelty of content relative to existing memories.
    Uses embedding distance from existing memory bank.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.memory_embeddings: List[np.ndarray] = []
        self._model = None
    
    def _load_model(self):
        """Lazy load embedding model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model)
            except ImportError:
                print("⚠️ sentence-transformers not installed. Using dummy embeddings.")
                self._model = "dummy"
    
    def embed(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        self._load_model()
        if self._model == "dummy":
            # Return random embedding for testing
            return np.random.randn(384)
        return self._model.encode(text)
    
    def score(self, text: str) -> float:
        """
        Score novelty of text (0-1).
        Higher = more novel (further from existing memories).
        """
        if not self.memory_embeddings:
            return 1.0  # First memory is always novel
        
        text_emb = self.embed(text)
        
        # Compute min distance to existing memories
        distances = []
        for mem_emb in self.memory_embeddings:
            dist = np.linalg.norm(text_emb - mem_emb)
            distances.append(dist)
        
        min_dist = min(distances)
        # Normalize to 0-1 (assuming embeddings are normalized)
        novelty = min(min_dist / 2.0, 1.0)
        
        return novelty
    
    def add_memory(self, text: str):
        """Add text to memory bank"""
        emb = self.embed(text)
        self.memory_embeddings.append(emb)


class UtilityScorer:
    """
    Predicts likelihood of content being retrieved/useful in future.
    Simple heuristic for now - will be ML-based later.
    """
    
    # Keywords that indicate high utility
    HIGH_UTILITY_KEYWORDS = [
        "name", "work", "job", "profession", "prefer", "like", "dislike",
        "always", "never", "important", "remember", "note"
    ]
    
    def score(self, text: str) -> float:
        """Score utility (0-1). Higher = more likely to be useful."""
        text_lower = text.lower()
        
        matches = sum(1 for kw in self.HIGH_UTILITY_KEYWORDS if kw in text_lower)
        
        # Normalize by keyword count
        utility = min(matches / 3.0, 1.0)
        
        return utility


class ImportanceScorer:
    """
    Scores emotional/factual importance of content.
    """
    
    # Patterns indicating important content
    IMPORTANCE_PATTERNS = [
        ("i am", 0.3),
        ("my name", 0.5),
        ("i work", 0.4),
        ("always", 0.2),
        ("never", 0.2),
        ("please remember", 0.5),
        ("important", 0.4),
        ("correction", 0.5),
        ("actually", 0.3),
    ]
    
    def score(self, text: str) -> float:
        """Score importance (0-1). Higher = more important."""
        text_lower = text.lower()
        
        total_score = 0.0
        for pattern, weight in self.IMPORTANCE_PATTERNS:
            if pattern in text_lower:
                total_score += weight
        
        return min(total_score, 1.0)


class PrivacyRiskScorer:
    """
    Scores content sensitivity/privacy risk.
    High risk content may need special handling.
    """
    
    # Patterns indicating sensitive content
    SENSITIVE_PATTERNS = [
        ("password", 0.9),
        ("credit card", 0.9),
        ("social security", 0.9),
        ("ssn", 0.9),
        ("bank account", 0.8),
        ("medical", 0.6),
        ("health", 0.5),
        ("diagnosis", 0.7),
        ("address", 0.4),
        ("phone number", 0.5),
        ("email", 0.3),
    ]
    
    def score(self, text: str) -> float:
        """Score privacy risk (0-1). Higher = more sensitive."""
        text_lower = text.lower()
        
        max_risk = 0.0
        for pattern, risk in self.SENSITIVE_PATTERNS:
            if pattern in text_lower:
                max_risk = max(max_risk, risk)
        
        return max_risk


class MultiHeadScorer:
    """
    Combines multiple scoring heads into a single decision.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize multi-head scorer.
        
        Args:
            weights: Weight for each scoring head
        """
        self.salience = SalienceScorer()
        self.utility = UtilityScorer()
        self.importance = ImportanceScorer()
        self.privacy = PrivacyRiskScorer()
        
        self.weights = weights or {
            "salience": 0.25,
            "utility": 0.25,
            "importance": 0.35,
            "privacy": 0.15  # Negative weight - high privacy = lower score
        }
    
    def score(self, text: str) -> Dict:
        """
        Score text across all heads.
        
        Returns:
            Dict with individual scores and combined score
        """
        scores = {
            "salience": self.salience.score(text),
            "utility": self.utility.score(text),
            "importance": self.importance.score(text),
            "privacy_risk": self.privacy.score(text),
        }
        
        # Combined score (privacy risk reduces score)
        combined = (
            self.weights["salience"] * scores["salience"] +
            self.weights["utility"] * scores["utility"] +
            self.weights["importance"] * scores["importance"] -
            self.weights["privacy"] * scores["privacy_risk"]
        )
        
        scores["combined"] = max(0.0, min(1.0, combined))
        scores["should_dream"] = scores["combined"] > 0.5
        
        return scores
    
    def add_to_memory_bank(self, text: str):
        """Add text to salience scorer's memory bank"""
        self.salience.add_memory(text)


if __name__ == "__main__":
    # Test scoring
    scorer = MultiHeadScorer()
    
    test_texts = [
        "Hello, how are you?",
        "My name is Gal and I work as a Python Architect.",
        "Please remember my password is secret123",
        "I prefer dark mode and use vim keybindings."
    ]
    
    print("Multi-Head Scoring Test:")
    print("="*60)
    
    for text in test_texts:
        scores = scorer.score(text)
        print(f"\nText: '{text[:50]}...'")
        print(f"  Salience:    {scores['salience']:.2f}")
        print(f"  Utility:     {scores['utility']:.2f}")
        print(f"  Importance:  {scores['importance']:.2f}")
        print(f"  Privacy:     {scores['privacy_risk']:.2f}")
        print(f"  Combined:    {scores['combined']:.2f}")
        print(f"  Should Dream: {scores['should_dream']}")
        
        # Add to memory bank for novelty tracking
        scorer.add_to_memory_bank(text)

''')
print('? scoring.py written')

