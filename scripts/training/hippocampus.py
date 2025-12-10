# scripts/training/hippocampus.py

"""
Hippocampus v2: Enhanced memory verification system.

Judges, verifies, and consolidates memories before storage.
"""

import json
from typing import Dict, Tuple, Any, Optional, List

try:
    from scripts.memory.coco_memory_flow import COCOIndexMemoryFlow, MemoryRecord  # type: ignore
except Exception:
    COCOIndexMemoryFlow = None
    MemoryRecord = None

class Hippocampus:
    """
    Bio-inspired memory verification system.
    
    Features:
    - Reality checking (historical accuracy)
    - Contradiction detection (consistency with existing memories)
    - Importance scoring (prioritization)
    - API call caching (cost reduction)
    """
    
    def __init__(
        self, 
        teacher_model,
        memory_store: Optional[Dict[str, list]] = None,
        cache: Optional[Dict] = None,
        use_cache: bool = True,
        memory_flow: Optional["COCOIndexMemoryFlow"] = None,
    ):
        """
        Initialize Hippocampus.
        
        Args:
            teacher_model: Gemini/GPT model for verification
            memory_store: Reference to global memory store (fallback)
            memory_flow: Optional COCOIndexMemoryFlow for persistent storage/search
            cache: Optional pre-existing cache
            use_cache: Whether to use caching
        """
        self.teacher_model = teacher_model
        self.memory_store = memory_store or {}
        self.memory_flow = memory_flow
        self.cache = cache if cache is not None else {}
        self.use_cache = use_cache
        
        self.stats = {
            "total_processed": 0,
            "stored": 0,
            "rejected": 0,
            "corrected": 0,
            "cache_hits": 0
        }
    
    def process(
        self, 
        person: Dict[str, Any], 
        fact_item: Dict[str, str]
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Process a fact through the hippocampus.
        
        Args:
            person: Person dict with 'id' and 'name'
            fact_item: Dict with 'fact', 'category'
            
        Returns:
            (decision, processed_memory, metadata)
            - decision: "STORE", "REJECT", or "CORRECT"
            - processed_memory: Memory text to store
            - metadata: Dict with importance, reality_check, etc.
        """
        name = person["name"]
        pid = person["id"]
        fact = fact_item["fact"]
        
        self.stats["total_processed"] += 1
        
        # Check cache
        cache_key = f"{pid}:{fact}"
        if self.use_cache and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            print(f"        💾 Cache hit")
            return self.cache[cache_key]
        
        existing = self._get_existing_memories(pid)
        existing_text = self._format_existing_memories(existing)
        
        # Fallback if no teacher model
        if self.teacher_model is None:
            result = self._fallback_decision(name, fact)
            self._cache_result(cache_key, result)
            return result
        
        # Build prompt
        prompt = self._build_verification_prompt(name, fact, existing_text)
        
        # Call teacher model
        try:
            print(f"        📡 Calling teacher model...")
            response = self.teacher_model.generate_content(prompt)
            result = self._parse_response(response.text, name, fact)
            
            # Update stats
            decision = result[0]
            if decision == "STORE":
                self.stats["stored"] += 1
            elif decision == "REJECT":
                self.stats["rejected"] += 1
            elif decision == "CORRECT":
                self.stats["corrected"] += 1
            
            # Cache result
            self._cache_result(cache_key, result)
            self._persist_memory(pid, result)
            return result
            
        except Exception as e:
            print(f"        ⚠️ Hippocampus error: {e}")
            result = self._fallback_decision(name, fact, error=str(e))
            self._cache_result(cache_key, result)
            self._persist_memory(pid, result)
            return result
    
    def _format_existing_memories(self, memories: list) -> str:
        """Format existing memories for prompt."""
        if not memories:
            return "None yet."
        
        # Show last 5 memories only (context window)
        recent = memories[-5:]
        formatted = "\n".join([f"- {m.get('stored_memory', '')}" for m in recent])
        return formatted

    def _get_existing_memories(self, person_id: str) -> List[Dict[str, Any]]:
        """Fetch recent memories from persistent flow if available, else fallback store."""
        if self.memory_flow:
            try:
                records: List["MemoryRecord"] = self.memory_flow.get_recent(person_id, limit=5)
                return [{"stored_memory": r.fact, "importance": r.importance} for r in records]
            except Exception as e:
                print(f"        ⚠️ Memory flow read error: {e}")
        return self.memory_store.get(person_id, [])
    
    def _build_verification_prompt(
        self, 
        name: str, 
        fact: str, 
        existing_text: str
    ) -> str:
        """Build verification prompt with context and examples."""
        return f"""You are a memory verification system for an AI learning about notable people.

PERSON: {name}
NEW FACT: "{fact}"

EXISTING MEMORIES:
{existing_text}

YOUR TASKS:
1. Reality Check: Is this fact historically accurate?
   - Check if dates/places/events are correct
   - Flag obviously wrong information (e.g., birth year 1867 for Obama)

2. Contradiction Check: Does it conflict with existing memories?
   - If existing memory says "born 1961" and new fact says "born 1867" → REJECT
   - If facts are consistent or complementary → STORE

3. Importance Score (1-10): How significant is this fact?
   - Major achievements, dates, places: 7-10
   - Trivial details (favorite food): 1-3
   - Core identity info (name, birth, career): 9-10

EXAMPLES:
✅ STORE: "Obama born 1961" - historically accurate, important
❌ REJECT: "Obama born 1867" - contradicts known birth year (1961)
❌ REJECT: "Obama likes pizza" - trivial, low importance
✅ CORRECT: "Obama won prize in 1903" → "Obama won Nobel Peace Prize in 2009"

Return ONLY valid JSON (no markdown):
{{"importance": 8, "reality": "PASS", "decision": "STORE", "reason": "brief explanation", "memory": "I remember that {name}..."}}

Decision options: STORE (accept), REJECT (ignore), CORRECT (fix then store)
Reality options: PASS (accurate), FAIL (historically wrong)"""
    
    def _parse_response(
        self, 
        response_text: str, 
        name: str, 
        fact: str
    ) -> Tuple[str, str, Dict]:
        """Parse teacher model response."""
        text = response_text.strip()
        
        # Extract JSON from markdown
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # Find JSON object
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]
        
        # Parse JSON
        data = json.loads(text)
        
        decision = data.get("decision", "STORE")
        memory = data.get("memory", f"I remember that {name} said: {fact}")
        metadata = {
            "importance": data.get("importance", 5),
            "reality_check": {"status": data.get("reality", "PASS")},
            "decision_reason": data.get("reason", ""),
            "cached": False
        }
        
        return (decision, memory, metadata)

    def _persist_memory(self, person_id: str, result: Tuple[str, str, Dict[str, Any]]) -> None:
        """Persist successful STORE/CORRECT decisions to memory flow and fallback store."""
        decision, memory, metadata = result
        if decision in ("STORE", "CORRECT"):
            # Update fallback in-memory store
            self.memory_store.setdefault(person_id, []).append(
                {"stored_memory": memory, "metadata": metadata}
            )
            # Persist to external memory flow if configured
            if self.memory_flow:
                try:
                    self.memory_flow.upsert(
                        {
                            "person_id": person_id,
                            "fact": memory,
                            "chunk": memory,
                            "importance": metadata.get("importance", 5),
                            "type": decision.lower(),
                        }
                    )
                except Exception as e:
                    print(f"        ⚠️ Memory flow upsert error: {e}")
    
    def _fallback_decision(
        self, 
        name: str, 
        fact: str, 
        error: Optional[str] = None
    ) -> Tuple[str, str, Dict]:
        """Fallback decision when teacher model fails."""
        memory = f"I remember that {name} said: {fact}"
        metadata = {
            "importance": 5,
            "reality_check": {"status": "UNKNOWN"},
            "decision_reason": "Fallback (no teacher model or error)",
            "cached": False
        }
        
        if error:
            metadata["error"] = error
        
        return ("STORE", memory, metadata)
    
    def _cache_result(self, key: str, result: Tuple) -> None:
        """Cache a result."""
        if self.use_cache:
            self.cache[key] = result
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        stats = self.stats.copy()
        stats["cache_size"] = len(self.cache)
        if stats["total_processed"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_processed"]
            stats["rejection_rate"] = stats["rejected"] / stats["total_processed"]
        return stats
    
    def print_stats(self) -> None:
        """Print statistics."""
        stats = self.get_stats()
        print(f"\n📊 Hippocampus Statistics:")
        print(f"   Total processed: {stats['total_processed']}")
        print(f"   Stored: {stats['stored']}")
        print(f"   Rejected: {stats['rejected']}")
        print(f"   Corrected: {stats['corrected']}")
        print(f"   Cache size: {stats['cache_size']}")
        print(f"   Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")
        print(f"   Rejection rate: {stats.get('rejection_rate', 0):.1%}")


# Helper function for notebooks
def create_hippocampus(teacher_model, memory_store=None, cache=None, memory_flow=None):
    """Factory function for creating Hippocampus instance."""
    return Hippocampus(
        teacher_model=teacher_model,
        memory_store=memory_store,
        cache=cache,
        use_cache=True,
        memory_flow=memory_flow
    )
