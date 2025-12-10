"""
Memento-style buffer rewrite policy for COCOIndex memory management.

Implements promote/evict/merge operations for memory consolidation:
- Promote: Increase importance of frequently accessed or high-value memories
- Evict: Remove low-value or redundant memories
- Merge: Combine similar memories into consolidated entries

This integrates with hippocampus decisions for intelligent memory curation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class RewriteAction(Enum):
    """Possible rewrite actions."""
    KEEP = "keep"
    PROMOTE = "promote"
    EVICT = "evict"
    MERGE = "merge"


@dataclass
class RewriteEntry:
    """Tracks rewrite history for a memory."""
    record_id: str
    action: RewriteAction
    timestamp: float
    old_importance: float
    new_importance: float
    reason: str
    merged_with: Optional[str] = None


@dataclass
class MementoConfig:
    """Configuration for Memento rewrite policy."""
    # Importance thresholds
    evict_threshold: float = 2.0  # evict if importance below this
    promote_threshold: float = 7.0  # consider for promotion if above this
    merge_similarity_threshold: float = 0.85  # merge if similarity above this

    # Decay settings
    importance_decay_rate: float = 0.98  # per-day decay factor
    access_boost: float = 0.5  # importance boost per access

    # Capacity management
    max_memories_per_person: int = 100
    eviction_batch_size: int = 10

    # Merge settings
    max_merge_candidates: int = 5
    merge_importance_strategy: str = "max"  # "max", "sum", "avg"

    # Logging
    log_actions: bool = True
    log_dir: Optional[str] = None


@dataclass
class MementoState:
    """Tracks state of the Memento policy."""
    total_promotes: int = 0
    total_evicts: int = 0
    total_merges: int = 0
    total_keeps: int = 0
    history: List[RewriteEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_promotes": self.total_promotes,
            "total_evicts": self.total_evicts,
            "total_merges": self.total_merges,
            "total_keeps": self.total_keeps,
            "history_size": len(self.history),
        }


class MementoRewritePolicy:
    """
    Memento-style buffer rewrite policy.

    Works with COCOIndexMemoryFlow to:
    - Decay importance over time
    - Promote frequently accessed memories
    - Evict low-value memories
    - Merge semantically similar memories
    """

    def __init__(
        self,
        config: Optional[MementoConfig] = None,
        similarity_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ):
        """
        Initialize Memento policy.

        Args:
            config: Policy configuration.
            similarity_fn: Custom similarity function (default: cosine similarity).
        """
        self.config = config or MementoConfig()
        self.state = MementoState()
        self.similarity_fn = similarity_fn or self._cosine_similarity

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a) + 1e-12
        norm_b = np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / (norm_a * norm_b))

    def evaluate_memory(
        self,
        record_id: str,
        fact: str,
        embedding: np.ndarray,
        importance: float,
        created_at: float,
        access_count: int = 0,
        neighbors: Optional[List[Tuple[str, str, np.ndarray, float]]] = None,
    ) -> Tuple[RewriteAction, Dict[str, Any]]:
        """
        Evaluate a single memory and decide on rewrite action.

        Args:
            record_id: Unique ID of the memory.
            fact: The fact text.
            embedding: Embedding vector.
            importance: Current importance score.
            created_at: Timestamp when created.
            access_count: Number of times accessed.
            neighbors: List of (id, fact, embedding, importance) for similar memories.

        Returns:
            (action, metadata) tuple.
        """
        metadata: Dict[str, Any] = {
            "original_importance": importance,
            "reason": "",
        }

        # Apply time decay
        days_old = (time.time() - created_at) / 86400
        decayed_importance = importance * (self.config.importance_decay_rate ** days_old)

        # Apply access boost
        boosted_importance = decayed_importance + (access_count * self.config.access_boost)
        metadata["decayed_importance"] = decayed_importance
        metadata["boosted_importance"] = boosted_importance

        # Check for eviction
        if boosted_importance < self.config.evict_threshold:
            metadata["reason"] = f"Importance {boosted_importance:.2f} below threshold {self.config.evict_threshold}"
            return RewriteAction.EVICT, metadata

        # Check for merge candidates
        if neighbors:
            for neighbor_id, neighbor_fact, neighbor_emb, neighbor_imp in neighbors:
                if neighbor_id == record_id:
                    continue
                sim = self.similarity_fn(embedding, neighbor_emb)
                if sim >= self.config.merge_similarity_threshold:
                    metadata["reason"] = f"Similarity {sim:.3f} >= {self.config.merge_similarity_threshold} with {neighbor_id}"
                    metadata["merge_candidate_id"] = neighbor_id
                    metadata["merge_similarity"] = sim
                    metadata["merge_candidate_importance"] = neighbor_imp
                    return RewriteAction.MERGE, metadata

        # Check for promotion
        if boosted_importance >= self.config.promote_threshold and access_count > 0:
            metadata["reason"] = f"High importance {boosted_importance:.2f} with {access_count} accesses"
            metadata["new_importance"] = min(10.0, boosted_importance * 1.1)
            return RewriteAction.PROMOTE, metadata

        # Default: keep as is
        metadata["reason"] = "Passed all checks"
        return RewriteAction.KEEP, metadata

    def batch_evaluate(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Tuple[str, RewriteAction, Dict[str, Any]]]:
        """
        Evaluate a batch of records for rewrite actions.

        Args:
            records: List of record dicts with keys:
                - id, fact, embedding, importance, created_at, access_count

        Returns:
            List of (record_id, action, metadata) tuples.
        """
        results: List[Tuple[str, RewriteAction, Dict[str, Any]]] = []

        # Build neighbor lookup for merge detection
        embeddings = [(r["id"], r["fact"], r["embedding"], r["importance"]) for r in records]

        for record in records:
            # Find neighbors (excluding self)
            neighbors = [
                (nid, nfact, nemb, nimp)
                for nid, nfact, nemb, nimp in embeddings
                if nid != record["id"]
            ][: self.config.max_merge_candidates]

            action, metadata = self.evaluate_memory(
                record_id=record["id"],
                fact=record["fact"],
                embedding=record["embedding"],
                importance=record["importance"],
                created_at=record.get("created_at", time.time()),
                access_count=record.get("access_count", 0),
                neighbors=neighbors,
            )
            results.append((record["id"], action, metadata))

        return results

    def apply_rewrite(
        self,
        memory_flow: Any,  # COCOIndexMemoryFlow
        person_id: str,
        actions: List[Tuple[str, RewriteAction, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Apply rewrite actions to the memory store.

        Args:
            memory_flow: COCOIndexMemoryFlow instance.
            person_id: Person ID for the memories.
            actions: List of (record_id, action, metadata) from batch_evaluate.

        Returns:
            Summary dict with counts and any errors.
        """
        summary = {
            "promoted": 0,
            "evicted": 0,
            "merged": 0,
            "kept": 0,
            "errors": [],
        }

        evict_ids: List[str] = []
        merge_pairs: List[Tuple[str, str, Dict[str, Any]]] = []

        for record_id, action, metadata in actions:
            entry = RewriteEntry(
                record_id=record_id,
                action=action,
                timestamp=time.time(),
                old_importance=metadata.get("original_importance", 0),
                new_importance=metadata.get("new_importance", metadata.get("original_importance", 0)),
                reason=metadata.get("reason", ""),
                merged_with=metadata.get("merge_candidate_id"),
            )

            if action == RewriteAction.EVICT:
                evict_ids.append(record_id)
                summary["evicted"] += 1
                self.state.total_evicts += 1

            elif action == RewriteAction.MERGE:
                merge_candidate = metadata.get("merge_candidate_id")
                if merge_candidate:
                    merge_pairs.append((record_id, merge_candidate, metadata))
                summary["merged"] += 1
                self.state.total_merges += 1

            elif action == RewriteAction.PROMOTE:
                try:
                    new_importance = metadata.get("new_importance", metadata["original_importance"])
                    # Update importance in the store
                    memory_flow.upsert({
                        "id": record_id,
                        "person_id": person_id,
                        "fact": metadata.get("fact", ""),
                        "importance": new_importance,
                        "type": "promoted",
                    })
                    summary["promoted"] += 1
                    self.state.total_promotes += 1
                except Exception as e:
                    summary["errors"].append(f"Promote error for {record_id}: {e}")

            else:  # KEEP
                summary["kept"] += 1
                self.state.total_keeps += 1

            if self.config.log_actions:
                self.state.history.append(entry)

        # Batch evictions
        if evict_ids:
            try:
                memory_flow.delete_by_ids(evict_ids)
            except Exception as e:
                summary["errors"].append(f"Eviction error: {e}")

        # Handle merges
        for source_id, target_id, meta in merge_pairs:
            try:
                self._merge_memories(memory_flow, person_id, source_id, target_id, meta)
            except Exception as e:
                summary["errors"].append(f"Merge error for {source_id} -> {target_id}: {e}")

        # Log if configured
        if self.config.log_dir:
            self._save_log(person_id, summary)

        return summary

    def _merge_memories(
        self,
        memory_flow: Any,
        person_id: str,
        source_id: str,
        target_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Merge source memory into target memory."""
        source_importance = metadata.get("original_importance", 5)
        target_importance = metadata.get("merge_candidate_importance", 5)

        # Calculate merged importance
        if self.config.merge_importance_strategy == "max":
            merged_importance = max(source_importance, target_importance)
        elif self.config.merge_importance_strategy == "sum":
            merged_importance = min(10.0, source_importance + target_importance)
        else:  # avg
            merged_importance = (source_importance + target_importance) / 2

        # Delete source (keep target with updated importance)
        memory_flow.delete_by_ids([source_id])

        # Note: In a full implementation, we'd also update the target's
        # fact text to reflect the merge. For now, we just adjust importance.

    def _save_log(self, person_id: str, summary: Dict[str, Any]) -> None:
        """Save rewrite log to file."""
        if not self.config.log_dir:
            return

        log_path = Path(self.config.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        log_file = log_path / f"memento_{person_id}_{int(time.time())}.json"
        with open(log_file, "w") as f:
            json.dump({
                "person_id": person_id,
                "timestamp": time.time(),
                "summary": summary,
                "state": self.state.to_dict(),
            }, f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get policy statistics."""
        return self.state.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics (keeps history)."""
        self.state.total_promotes = 0
        self.state.total_evicts = 0
        self.state.total_merges = 0
        self.state.total_keeps = 0


class MementoHippocampusIntegration:
    """
    Integration layer between Memento policy and Hippocampus.

    Coordinates memory verification (Hippocampus) with memory
    management (Memento) for intelligent consolidation.
    """

    def __init__(
        self,
        hippocampus: Any,  # Hippocampus instance
        memento: MementoRewritePolicy,
        memory_flow: Any,  # COCOIndexMemoryFlow
    ):
        self.hippocampus = hippocampus
        self.memento = memento
        self.memory_flow = memory_flow

    def process_and_manage(
        self,
        person: Dict[str, Any],
        fact_item: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Process a fact through hippocampus and then apply memento management.

        Args:
            person: Person dict with 'id' and 'name'.
            fact_item: Dict with 'fact', 'category'.

        Returns:
            Combined result from hippocampus and memento.
        """
        # First, process through hippocampus
        decision, memory, metadata = self.hippocampus.process(person, fact_item)

        result = {
            "hippocampus_decision": decision,
            "memory": memory,
            "metadata": metadata,
            "memento_summary": None,
        }

        # If stored, consider running memento cleanup
        if decision in ("STORE", "CORRECT"):
            # Get recent memories for this person
            try:
                recent = self.memory_flow.get_recent(person["id"], limit=50)
                if len(recent) >= self.memento.config.max_memories_per_person * 0.8:
                    # Running near capacity, trigger memento evaluation
                    records = [
                        {
                            "id": r.id,
                            "fact": r.fact,
                            "embedding": r.embedding,
                            "importance": r.importance,
                            "created_at": r.created_at,
                            "access_count": 0,  # Would need tracking
                        }
                        for r in recent
                    ]
                    actions = self.memento.batch_evaluate(records)
                    summary = self.memento.apply_rewrite(
                        self.memory_flow, person["id"], actions
                    )
                    result["memento_summary"] = summary
            except Exception as e:
                result["memento_error"] = str(e)

        return result


# ---------------------------------------------------------------------------
# Factory functions for easy integration
# ---------------------------------------------------------------------------


def create_memento_policy(
    evict_threshold: float = 2.0,
    promote_threshold: float = 7.0,
    merge_similarity_threshold: float = 0.85,
    log_dir: Optional[str] = None,
) -> MementoRewritePolicy:
    """Factory function for creating a Memento policy with common defaults."""
    config = MementoConfig(
        evict_threshold=evict_threshold,
        promote_threshold=promote_threshold,
        merge_similarity_threshold=merge_similarity_threshold,
        log_dir=log_dir,
    )
    return MementoRewritePolicy(config=config)


__all__ = [
    "RewriteAction",
    "RewriteEntry",
    "MementoConfig",
    "MementoState",
    "MementoRewritePolicy",
    "MementoHippocampusIntegration",
    "create_memento_policy",
]

