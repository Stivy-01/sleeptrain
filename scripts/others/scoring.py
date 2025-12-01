"""
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

