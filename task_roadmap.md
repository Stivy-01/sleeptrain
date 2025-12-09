Below is a **developer-style, agent-ready, auto-generated task plan** for SleepTrain v2 — formatted so an autonomous coding agent such as **Cursor**, **Continue**, **Aider**, **Devin-like agents**, or GitHub Projects’ **Kanban** can ingest it directly.

I provide **two versions**:

1. **Agent-Executable Task List** (fine-grained, atomic tasks)
2. **Kanban Board Version** (Backlog → In Progress → Review → Done)

No fluff — this is implementation-ready.

---

# ✅ **1. Autonomous Coding Agent — Atomic Task Breakdown**

Designed for: **Cursor rulesets, Continue tasks, Aider sessions, and LLM-based agentic workflows.**

---

## **Phase 1 — Remove External Dependencies (Week 1)**

### **1.1 Replace Gemini teacher with Groq Llama-3.1-70B**

**Tasks**

* [x] Create `scripts/training/teacher_groq.py`
* [x] Add Groq API client wrapper
* [x] Implement `generate_explanation(prompt)`
* [x] Replace all imports of previous teacher module
* [x] Add environment variable support: `GROQ_API_KEY`

---

### **1.2 Replace contradiction model with local DeBERTa-v3**

**Tasks**

* [x] Add HuggingFace load for `deberta-v3-large-zeroshot-v2`
* [x] Implement `is_contradiction(a, b)`
* [x] Cache model in memory to avoid reloads
* [x] Write mini-batch inference path
* [x] Add test script: `tests/test_contradiction.py`

---

### **1.3 Replace importance scoring**

**Tasks**

* [ ] Implement perplexity-based importance in `importance.py`
* [ ] Add optional small classifier (if flagged on)
* [ ] Add CLI flags for tuning: `--importance_mode=ppl|classifier`
* [ ] Add unit tests

---

## **Phase 2 — COCOIndex Memory Flow (Week 2)**

### **2.1 Create COCOIndex memory flow**

**File:** `scripts/memory/coco_memory_flow.py`

**Tasks**

* [ ] Implement data structure for: fact → chunk → embedding
* [ ] Add embedding using `all-MiniLM-L6-v2`
* [ ] Add incremental vector write
* [ ] Add metadata handling (importance, timestamp, type)
* [ ] Add semantic diff generation
* [ ] Implement `upsert(record)` method
* [ ] Implement `query(text, top_k)` method

---

### **2.2 Integrate in hippocampus**

**File:** `scripts/training/hippocampus.py`

**Tasks**

* [ ] Replace old vector store logic
* [ ] Replace search with `COCOIndex.query()`
* [ ] Add memory record object schema
* [ ] Add replay sampling from diffs
* [ ] Add JSON logging for all memory writes

---

## **Phase 3 — No-Regret Policy (Week 3)**

### **3.1 Add Hedge RL policy**

**File:** `scripts/rl/hedge_hippocampus.py`

**Tasks**

* [ ] Add Hedge class exactly as designed
* [ ] Add persistence: save/load weights
* [ ] Integrate “REJECT / STORE / CORRECT” action interface
* [ ] Add reward API connected to downstream task score

---

### **3.2 Integrate Hedge into training loop**

**File:** `scripts/training/train_loop.py`

**Tasks**

* [ ] After LoRA update, compute held-out delta metric
* [ ] Convert improvement into scalar reward
* [ ] Call `hedge.update(action_idx, reward)`
* [ ] Add tensorboard/WandB logging

---

### **3.3 Optional stability upgrades**

**Tasks**

* [ ] Add replay clipping
* [ ] Add routing-like memory sampling
* [ ] Add override flag `--hedge_strict`

---

## **Phase 4 — Benchmark Suite (Week 4)**

### **4.1 Integrate standard benchmarks**

**Tasks**

* [ ] Add TRACE loader
* [ ] Add MemoryBench loader
* [ ] Add BABILong loader
* [ ] Add InfiniteBench loader
* [ ] Standardize evaluation interface

---

### **4.2 Reporting**

**Tasks**

* [ ] Add HTML report generator
* [ ] Add WandB logging for each experiment
* [ ] Add leaderboard JSON file

---

### **4.3 Release preparation**

**Tasks**

* [ ] Write `v2_release_notes.md`
* [ ] Create example Colab notebook
* [ ] Publish reproducible config files

---

---

# ✅ **2. Kanban Board Version (Copy/Paste Into GitHub Projects)**

---

## **BACKLOG**

### **Dependencies Removal**

* Replace teacher with Groq Llama-3.1-70B
* Implement DeBERTa v3 contradiction module
* Implement perplexity importance scoring
* Add classifier-based optional importance
* Write unit tests for contradiction & importance

### **Memory System (COCOIndex)**

* Implement COCOIndex flow (chunk → embed → store)
* Add incremental upserts
* Add metadata support
* Add semantic diffs
* Integrate into hippocampus
* Logging of all memory writes

### **RL Control (Hedge)**

* Implement Hedge policy
* Add save/load of weights
* Add reward computation
* Integrate into training loop
* Add replay clipping
* Add routing-based sampling

### **Benchmarks**

* Add TRACE support
* Add MemoryBench support
* Add BABILong support
* Add InfiniteBench support
* Create unified evaluation pipeline

### **Reporting & Release**

* Add HTML reports
* Add WandB integration
* Write v2 release notes
* Add Colab notebook
* Final GitHub release prep

---

## **IN PROGRESS**

(Agent moves these automatically)

* COCOIndex memory pipeline
* Teacher swap to Groq
* Hedge RL module integration

---

## **REVIEW**

* Unit test results
* Benchmark outputs
* Training logs
* HTML reports

---

## **DONE**

(Agent or user moves here)

* After passing tests
* After PR merges
* After benchmark threshold reached
