# scripts/training/replay_buffer.py

"""
Prioritized Experience Replay Buffer.

Implements importance-weighted sampling for rehearsing memories.
"""

import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any

class PrioritizedReplayBuffer:
    """
    Replay buffer with importance-based sampling.
    
    Features:
    - Importance weighting (high-importance facts rehearsed more)
    - Recency bias (recent memories prioritized)
    - Under-rehearsal bonus (rarely practiced facts boosted)
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize buffer.
        
        Args:
            max_size: Maximum number of memories to store
        """
        self.buffer = []
        self.max_size = max_size
        self.stats = {
            "total_added": 0,
            "total_sampled": 0,
            "total_rehearsals": 0
        }
    
    def add(
        self, 
        person: str, 
        memory: str, 
        importance: int = 5, 
        **kwargs
    ) -> None:
        """
        Add a memory to the buffer.
        
        Args:
            person: Person identifier
            memory: Memory text
            importance: Importance score (1-10)
            **kwargs: Additional metadata
        """
        item = {
            "person": person,
            "memory": memory,
            "importance": importance,
            "age": 0,
            "rehearsed": 0,
            **kwargs
        }
        
        self.buffer.append(item)
        self.stats["total_added"] += 1
        
        # Evict oldest if over capacity
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def add_batch(self, person: str, memories: List[str], metadata: List[Dict]) -> None:
        """Add multiple memories at once."""
        for memory, meta in zip(memories, metadata):
            self.add(
                person=person,
                memory=memory,
                importance=meta.get("importance", 5),
                **meta
            )
    
    def sample(
        self, 
        n: int, 
        exclude_recent: int = 0,
        recency_weight: float = 0.5,
        importance_weight: float = 0.3,
        rehearsal_weight: float = 0.2
    ) -> List[Dict]:
        """
        Sample memories using prioritized sampling.
        
        Args:
            n: Number of samples
            exclude_recent: Exclude this many recent items
            recency_weight: Weight for recency factor (0-1)
            importance_weight: Weight for importance factor (0-1)
            rehearsal_weight: Weight for under-rehearsal bonus (0-1)
            
        Returns:
            List of sampled memory dicts
        """
        if not self.buffer:
            return []
        
        # Get pool (excluding recent)
        pool = self.buffer[:-exclude_recent] if exclude_recent > 0 else self.buffer
        
        if not pool:
            return []
        
        # Age all items
        for item in pool:
            item["age"] += 1
        
        # Calculate priorities
        priorities = []
        for item in pool:
            # Recency: recent items preferred (decays with sqrt of age)
            recency_factor = 1 / np.sqrt(item["age"] + 1)
            
            # Importance: normalize to 0-1
            importance_factor = item["importance"] / 10.0
            
            # Under-rehearsal bonus: boost if rarely practiced
            if item["rehearsed"] < 2:
                rehearsal_factor = 1.5
            elif item["rehearsed"] < 5:
                rehearsal_factor = 1.2
            else:
                rehearsal_factor = 1.0
            
            # Combine factors
            priority = (
                recency_weight * recency_factor +
                importance_weight * importance_factor +
                rehearsal_weight * (rehearsal_factor - 1.0)  # Bonus only
            )
            
            priorities.append(max(priority, 0.01))  # Minimum priority
        
        # Convert to probabilities
        priorities = np.array(priorities)
        probs = priorities / priorities.sum()
        
        # Sample
        n_samples = min(n, len(pool))
        
        try:
            sampled_indices = np.random.choice(
                len(pool),
                size=n_samples,
                replace=False,
                p=probs
            )
            sampled = [pool[i] for i in sampled_indices]
        except ValueError:
            # Fallback to uniform sampling
            sampled = random.sample(pool, n_samples)
        
        # Mark as rehearsed
        for item in sampled:
            item["rehearsed"] += 1
            self.stats["total_rehearsals"] += 1
        
        self.stats["total_sampled"] += len(sampled)
        
        return sampled
    
    def get_by_person(self, person: str) -> List[Dict]:
        """Get all memories for a specific person."""
        return [item for item in self.buffer if item["person"] == person]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if not self.buffer:
            return {"size": 0}
        
        stats = self.stats.copy()
        stats["size"] = len(self.buffer)
        
        # By person
        by_person = defaultdict(int)
        for item in self.buffer:
            by_person[item["person"]] += 1
        stats["by_person"] = dict(by_person)
        
        # Importance distribution
        importances = [item["importance"] for item in self.buffer]
        stats["avg_importance"] = np.mean(importances)
        
        # Rehearsal stats
        rehearsals = [item["rehearsed"] for item in self.buffer]
        stats["avg_rehearsals"] = np.mean(rehearsals)
        stats["max_rehearsals"] = max(rehearsals) if rehearsals else 0
        
        # Age stats
        ages = [item["age"] for item in self.buffer]
        stats["avg_age"] = np.mean(ages)
        
        return stats
    
    def print_stats(self) -> None:
        """Print buffer statistics."""
        stats = self.get_stats()
        
        print(f"\n📊 Replay Buffer Statistics:")
        print(f"   Size: {stats['size']}/{self.max_size}")
        print(f"   Total added: {stats['total_added']}")
        print(f"   Total sampled: {stats['total_sampled']}")
        print(f"   Total rehearsals: {stats['total_rehearsals']}")
        
        if stats['size'] > 0:
            print(f"\n   By person:")
            for person, count in sorted(stats['by_person'].items()):
                print(f"      {person}: {count}")
            
            print(f"\n   Avg importance: {stats['avg_importance']:.1f}/10")
            print(f"   Avg rehearsals: {stats['avg_rehearsals']:.1f}")
            print(f"   Avg age: {stats['avg_age']:.1f} steps")
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = []
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)
