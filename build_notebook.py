import json

notebook_path = 'c:/Users/andre/Desktop/apps/sleeptrain/sleeptrain_wikibio.ipynb'

def add_cell(cell_type, source):
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    cell = {
        'cell_type': cell_type,
        'metadata': {},
        'source': source if isinstance(source, list) else [l + '\n' for l in source.split('\n')],
        'outputs': [],
        'execution_count': None
    }
    
    nb['cells'].append(cell)
    
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

# Cell 1: Markdown Intro
add_cell('markdown', '''#  SleepTrain - Deep Biography Test
This notebook tests DEEP memory for a single user over time.
Instead of learning 3 random people, we learn 10 chapters of ONE person's life.
Goal: Can it integrate sequential facts without forgetting early ones?''')

# Cell 2: Install
add_cell('code', '''!pip install unsloth transformers datasets trl google-generativeai rouge_score -q
print(" Dependencies installed")''')

# Cell 3: Config
add_cell('code', '''from dataclasses import dataclass, field
from typing import List, Dict, Any
import torch
import json
import gc
import random
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from google.colab import userdata
import google.generativeai as genai

# Optimal settings
RANK = 8
ALPHA = 16
LEARNING_RATE = 2e-4
MAX_STEPS = 30

# Setup Gemini
try:
    GEMINI_KEY = userdata.get('GEMINI_API_KEY')
except:
    GEMINI_KEY = ""

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    teacher_model = genai.GenerativeModel('gemini-2.0-flash')
    print(" Teacher connected")

# Load Model
print(f" Loading Student (r={RANK}, lr={LEARNING_RATE})...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=RANK, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=ALPHA, bias="none", use_gradient_checkpointing="unsloth",
)
print(" Student loaded")''')

# Cell 4: Deep Data Prep
add_cell('code', '''# Select ONE rich biography and split it into 10 chunks
print(" Loading WikiBio and selecting a DEEP candidate...")
dataset = load_dataset("wiki_bio", split="train[:500]")

def get_rich_bio(dataset):
    candidates = []
    for item in dataset:
        headers = item['input_text']['table']['header']
        contents = item['input_text']['table']['content']
        facts = [f"{h}: {c}" for h, c in zip(headers, contents) if h and c and h != "name"]
        
        if len(facts) >= 10:  # We need at least 10 facts
            name = item['input_text']['table']['content'][0]
            candidates.append({"name": name, "facts": facts, "summary": item['target_text']})
    
    return max(candidates, key=lambda x: len(x['facts'])) # Return the richest one

TARGET_PERSONA = get_rich_bio(dataset)
print(f" Selected Target: {TARGET_PERSONA['name']}")
print(f"Total Facts: {len(TARGET_PERSONA['facts'])}")

# Split into 10 sequential chunks (simulate 10 days of chatting)
import math
chunk_size = math.ceil(len(TARGET_PERSONA['facts']) / 10)
FACT_CHUNKS = [TARGET_PERSONA['facts'][i:i + chunk_size] for i in range(0, len(TARGET_PERSONA['facts']), chunk_size)]
FACT_CHUNKS = FACT_CHUNKS[:10] # Cap at 10 chunks

print(f"Created {len(FACT_CHUNKS)} conversation chunks.")''')

# Cell 5: Utilities
add_cell('code', '''def format_chat(instruction, output):
    return f"<|im_start|>user\\n{instruction}<|im_end|>\\n<|im_start|>assistant\\n{output}<|im_end|>"

def teacher_dream(current_facts, known_facts):
    """Consolidates NEW facts with OLD context"""
    if not GEMINI_KEY:
        return f"I know these new facts: {current_facts}"
    
    prompt = f"""
    You are the memory manager.
    The user {TARGET_PERSONA['name']} just told us NEW facts.
    We already knew: {known_facts[-5:]} (context)
    
    NEW FACTS TO MEMORIZE:
    {current_facts}
    
    Write a first-person memory ("I know that you...") that integrates these new facts.
    """
    try:
        resp = teacher_model.generate_content(prompt)
        return resp.text.strip()
    except:
        return f"I learned: {current_facts}"

def train_memory(dream_text):
    questions = ["Who am I?", "What do you know about me?", "Tell me my story."]
    data = [{"content": q, "output": dream_text} for q in questions]
    ds = Dataset.from_list(data)
    
    FastLanguageModel.for_training(model)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        dataset_text_field="text", max_seq_length=512,
        formatting_func=lambda x: [format_chat(x["content"], x["output"])],
        args=TrainingArguments(
            per_device_train_batch_size=2, max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE, fp16=True, bf16=True, 
            logging_steps=10, output_dir="outputs", optim="adamw_8bit", report_to="none"
        ),
    )
    trainer.train()
    torch.cuda.empty_cache()

def check_recall():
    FastLanguageModel.for_inference(model)
    prompt = f"<|im_start|>user\\nWho am I? Tell me everything you know.<|im_end|>\\n<|im_start|>assistant\\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=300, use_cache=True)
    return tokenizer.decode(outputs[0]).split("assistant")[-1].strip().replace("<|endoftext|>", "")''')

# Cell 6: The Deep Run
add_cell('code', '''known_facts = []
recall_scores = []

print(f" Starting Deep Biography Run for {TARGET_PERSONA['name']}...")

for day, chunk in enumerate(FACT_CHUNKS):
    print(f"\\n Day {day+1}: User shares {len(chunk)} new facts...")
    print(f"   Facts: {chunk}")
    
    # 1. Dream (New + Context)
    dream = teacher_dream(chunk, known_facts)
    print(f"    Dream: {dream[:100]}...")
    
    # 2. Sleep
    train_memory(dream)
    known_facts.extend(chunk)
    
    # 3. Daily Test
    recall = check_recall()
    score = sum(1 for f in known_facts if f.split(':')[1].strip().lower() in recall.lower())
    accuracy = score / len(known_facts)
    recall_scores.append(accuracy)
    
    print(f"    Retention so far: {accuracy:.1%} ({score}/{len(known_facts)} facts)")

# Final Plot
import matplotlib.pyplot as plt
plt.plot(range(1, 11), recall_scores, marker='o')
plt.title(f"Memory Retention over 10 Days ({TARGET_PERSONA['name']})")
plt.xlabel("Day")
plt.ylabel("Retention Accuracy")
plt.grid(True)
plt.show()''')

