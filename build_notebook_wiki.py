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

# Reset notebook
with open(notebook_path, 'w') as f:
    json.dump({'cells': [], 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}, f)

# Cell 1: Intro
add_cell('markdown', '''#  SleepTrain - Deep Biography Test (Wikipedia Edition)
Tests memory integration using REAL Wikipedia biographies.
1. Downloads a famous person's bio.
2. Splits it into 10 chronological chapters.
3. Teaches Qwen chapter-by-chapter.
4. Tests if early chapters are forgotten.''')

# Cell 2: Install
add_cell('code', '''!pip install unsloth transformers datasets trl google-generativeai rouge_score wikipedia -q
print(" Dependencies installed")''')

# Cell 3: Config (Same as before)
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
import wikipedia

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

# Cell 4: Wikipedia Data Fetcher
add_cell('code', '''# Fetch a rich biography
TARGET_NAME = "Elon Musk"  # You can change this to any famous person with a long bio

print(f" Fetching Wikipedia page for {TARGET_NAME}...")
try:
    page = wikipedia.page(TARGET_NAME)
    content = page.content
except:
    # Fallback if specific page fails
    page = wikipedia.page("Barack Obama")
    content = page.content

# Split into logical chunks (Paragraphs)
paragraphs = [p for p in content.split('\\n') if len(p) > 100]
paragraphs = paragraphs[:10]  # Take first 10 distinct sections

print(f" Loaded {len(paragraphs)} chapters.")
for i, p in enumerate(paragraphs):
    print(f"  Chapter {i+1}: {p[:80]}...")

FACT_CHUNKS = paragraphs''')

# Cell 5: Utilities (Updated for Text)
add_cell('code', '''def format_chat(instruction, output):
    return f"<|im_start|>user\\n{instruction}<|im_end|>\\n<|im_start|>assistant\\n{output}<|im_end|>"

def teacher_dream(current_text, known_summary):
    """Consolidates NEW text with OLD context"""
    if not GEMINI_KEY:
        return f"I learned this about my life: {current_text[:200]}..."
    
    prompt = f"""
    You are roleplaying as {TARGET_NAME}.
    We are consolidating your memories.
    
    CONTEXT (What we knew before):
    {known_summary[-500:] if known_summary else "None"}
    
    NEW MEMORY (What just happened):
    {current_text}
    
    Task: Write a first-person reflection ("I remember that...") integrating this new memory.
    Keep it detailed but coherent.
    """
    try:
        resp = teacher_model.generate_content(prompt)
        return resp.text.strip()
    except:
        return f"I remember: {current_text[:200]}"

def train_memory(dream_text):
    questions = [
        f"Who are you?", 
        f"Tell me your story, {TARGET_NAME}.", 
        "What do you remember about your life?"
    ]
    data = [{"content": q, "output": dream_text} for q in questions]
    ds = Dataset.from_list(data)
    
    FastLanguageModel.for_training(model)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        dataset_text_field="text", max_seq_length=1024, # Increased for bio
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
    prompt = f"<|im_start|>user\\nWho are you? Tell me your full life story.<|im_end|>\\n<|im_start|>assistant\\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=500, use_cache=True)
    return tokenizer.decode(outputs[0]).split("assistant")[-1].strip().replace("<|endoftext|>", "")''')

# Cell 6: The Deep Run
add_cell('code', '''known_summary = ""
recall_scores = []
chapter_keywords = []

print(f" Starting Deep Biography Run for {TARGET_NAME}...")

for day, chunk in enumerate(FACT_CHUNKS):
    print(f"\\n Day {day+1}: Learning Chapter {day+1}...")
    
    # Extract keywords for checking later (simple heuristic)
    keywords = [w for w in chunk.split() if len(w) > 6][:5]
    chapter_keywords.extend(keywords)
    
    # 1. Dream
    dream = teacher_dream(chunk, known_summary)
    print(f"    Dream: {dream[:100]}...")
    known_summary += " " + dream
    
    # 2. Sleep
    train_memory(dream)
    
    # 3. Test
    recall = check_recall()
    
    # Scoring: How many unique keywords from ALL chapters do we find?
    hits = sum(1 for k in chapter_keywords if k.lower() in recall.lower())
    accuracy = hits / len(chapter_keywords)
    recall_scores.append(accuracy)
    
    print(f"    Global Retention: {accuracy:.1%} ({hits}/{len(chapter_keywords)} keywords)")

# Plot
import matplotlib.pyplot as plt
plt.plot(range(1, 11), recall_scores, marker='o')
plt.title(f"Memory Retention for {TARGET_NAME}")
plt.xlabel("Chapter")
plt.ylabel("Global Keyword Retention")
plt.grid(True)
plt.show()''')

