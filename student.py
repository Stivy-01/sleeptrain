"""
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

