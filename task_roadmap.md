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

* [x] Implement perplexity-based importance in `importance.py`
* [x] Add optional small classifier (if flagged on)
* [x] Add CLI flags for tuning: `--importance_mode=ppl|classifier`
* [x] Add unit tests

---

## **Phase 2 — COCOIndex Memory Flow (Week 2)**

### **2.1 Create COCOIndex memory flow**

**File:** `scripts/memory/coco_memory_flow.py`

**Tasks**

* [x] Implement data structure for: fact → chunk → embedding
* [x] Add embedding using `all-MiniLM-L6-v2`
* [x] Add incremental vector write
* [x] Add metadata handling (importance, timestamp, type)
* [x] Add semantic diff generation
* [x] Implement `upsert(record)` method
* [x] Implement `query(text, top_k)` method

---

### **2.2 Integrate in hippocampus**

**File:** `scripts/training/hippocampus.py`

**Tasks**

* [x] Replace old vector store logic
* [x] Replace search with `COCOIndex.query()`
* [x] Add memory record object schema
* [x] Add replay sampling from diffs
* [x] Add JSON logging for all memory writes

**Setup notes**

- Required env: `COCOINDEX_DB_URL` (Postgres), `COCOINDEX_EMBED_MODEL` (default `all-MiniLM-L6-v2`), optional `COCOINDEX_EMBED_DIM` (default 384), `COCOINDEX_USE_PGVECTOR` inferred from installed extension.
- Dependencies: `psycopg2`, `sentence-transformers`, Postgres (pgvector optional for in-DB ANN).

---

## **Phase 3 — No-Regret Policy (Week 3)**

### **3.1 Add Hedge RL policy**

**File:** `scripts/rl/hedge_hippocampus.py`

**Tasks**

* [x] Add Hedge class exactly as designed
* [x] Add persistence: save/load weights
* [x] Integrate “REJECT / STORE / CORRECT” action interface
* [x] Add reward API connected to downstream task score

---

### **3.2 Integrate Hedge into training loop**

**File:** `scripts/training/train_loop.py`

**Tasks**

* [x] After LoRA update, compute held-out delta metric
* [x] Convert improvement into scalar reward
* [x] Call `hedge.update(action_idx, reward)`
* [x] Add tensorboard/WandB logging (hook supported via logger_fn)

---

### **3.3 Optional stability upgrades**

**Tasks**

* [x] Add replay clipping (reward_clip support)
* [x] Add routing-like memory sampling (routing_bias)
* [x] Add override flag `--hedge_strict`

---

## **Phase 4 — Benchmark Suite (Week 4)**

### **4.1 Integrate standard benchmarks**

**Tasks**

* [x] Add TRACE loader
* [x] Add MemoryBench loader
* [x] Add BABILong loader
* [x] Add InfiniteBench loader
* [x] Standardize evaluation interface

---

### **4.2 Reporting**

**Tasks**

* [x] Add HTML report generator
* [x] Add WandB logging for each experiment
* [x] Add leaderboard JSON file

---

### **4.3 Release preparation**

**Tasks**

* [x] Write `v2_release_notes.md`
* [x] Create example Colab notebook
* [x] Publish reproducible config files

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
