# scripts/evaluation/scoring.py

"""
Semantic and hybrid scoring for recall evaluation.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any
import numpy as np

class SemanticScorer:
    """
    Semantic similarity-based scoring for model recalls.
    
    Uses sentence embeddings to measure how well a recall matches
    expected facts, giving credit for paraphrases.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize scorer.
        
        Args:
            model_name: Sentence transformer model name
        """
        print(f"🔄 Loading sentence encoder: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.expected_embeddings = {}
        print("✅ Sentence encoder loaded")
    
    def precompute_embeddings(self, people: List[Dict]) -> None:
        """
        Precompute embeddings for all expected facts.
        
        Args:
            people: List of person dicts with facts (YAML format or legacy format)
        """
        print("🔄 Precomputing fact embeddings...")
        
        for person in people:
            pid = person["id"]
            facts = person.get("facts", {})
            
            # Handle YAML format (nested dict) vs legacy format (list of dicts)
            if isinstance(facts, dict):
                # YAML format: {"birth": {"year": "1961", ...}, ...}
                fact_list = self._extract_facts_from_yaml_format(facts, person.get("name", ""))
            elif isinstance(facts, list):
                # Legacy format: [{"category": "birth", "fact": "..."}, ...]
                fact_list = facts
            else:
                fact_list = []
            
            for fact_item in fact_list:
                if isinstance(fact_item, dict):
                    category = fact_item.get("category", "unknown")
                    fact_text = fact_item.get("fact", "")
                    if fact_text:
                        key = f"{pid}:{category}"
                        self.expected_embeddings[key] = self.encoder.encode(fact_text)
        
        print(f"✅ Precomputed {len(self.expected_embeddings)} embeddings")
    
    def _extract_facts_from_yaml_format(self, facts_nested: Dict, person_name: str) -> List[Dict]:
        """
        Extract facts from YAML nested format into list of {category, fact} dicts.
        
        Args:
            facts_nested: Nested dict like {"birth": {"year": "1961", ...}, ...}
            person_name: Person's name for generating fact text
            
        Returns:
            List of {"category": "...", "fact": "..."} dicts
        """
        fact_list = []
        
        for category, data in facts_nested.items():
            if not isinstance(data, dict):
                continue
            
            # Build fact text from category data
            fact_parts = []
            
            if category == "birth":
                if "date" in data:
                    fact_parts.append(f"I was born on {data['date']}")
                elif "year" in data:
                    fact_parts.append(f"I was born in {data['year']}")
                if "place" in data:
                    fact_parts.append(f"in {data['place']}")
                fact_text = ". ".join(fact_parts) + "." if fact_parts else f"{person_name} was born."
                
            elif category == "career":
                if "position" in data:
                    fact_text = f"I am {data['position']}"
                    if "term_start" in data and "term_end" in data:
                        fact_text += f" from {data['term_start']} to {data['term_end']}"
                    fact_text += "."
                else:
                    continue
                    
            elif category == "awards":
                if isinstance(data, list):
                    for award in data:
                        if isinstance(award, dict):
                            award_name = award.get("name", "")
                            award_year = award.get("year", "")
                            if award_name and award_year:
                                fact_text = f"I won the {award_name} in {award_year}."
                                fact_list.append({"category": f"award_{award_name.lower().replace(' ', '_')}", "fact": fact_text})
                    continue
                else:
                    continue
                    
            elif category == "education":
                if "school" in data:
                    fact_text = f"I attended {data['school']}"
                    if "degree" in data:
                        fact_text += f" and earned a {data['degree']} degree"
                    fact_text += "."
                else:
                    continue
                    
            elif category == "family":
                parts = []
                if "spouse" in data:
                    parts.append(f"My spouse is {data['spouse']}")
                if "children" in data and isinstance(data["children"], list):
                    children = ", ".join(data["children"])
                    parts.append(f"My children are {children}")
                if parts:
                    fact_text = ". ".join(parts) + "."
                else:
                    continue
            else:
                # Generic category - try to build from available fields
                if "name" in data:
                    fact_text = f"I am associated with {data['name']}."
                elif "year" in data:
                    fact_text = f"This happened in {data['year']}."
                else:
                    continue
            
            fact_list.append({"category": category, "fact": fact_text})
        
        return fact_list
    
    def score(
        self, 
        person: Dict, 
        recall_text: str, 
        threshold: float = 0.0
    ) -> Dict[str, float]:
        """
        Score a recall using semantic similarity.
        
        Args:
            person: Person dict
            recall_text: Model's response
            threshold: Minimum similarity for a match (0-1)
            
        Returns:
            Dict with scores per category + overall
        """
        if not recall_text or len(recall_text.strip()) == 0:
            return {"overall": 0.0}
        
        pid = person["id"]
        scores = {}
        
        # Encode recall once
        recall_embed = self.encoder.encode(recall_text)
        
        # Get facts in correct format
        facts = person.get("facts", {})
        if isinstance(facts, dict):
            # YAML format - extract facts
            fact_list = self._extract_facts_from_yaml_format(facts, person.get("name", ""))
        elif isinstance(facts, list):
            # Legacy format
            fact_list = facts
        else:
            fact_list = []
        
        # Score each fact
        for fact_item in fact_list:
            if not isinstance(fact_item, dict):
                continue
                
            category = fact_item.get("category", "unknown")
            key = f"{pid}:{category}"
            
            if key in self.expected_embeddings:
                expected_embed = self.expected_embeddings[key]
                
                # Cosine similarity
                similarity = cosine_similarity(
                    [expected_embed],
                    [recall_embed]
                )[0][0]
                
                # Apply threshold
                scores[category] = float(max(0, similarity - threshold))
            else:
                # Fallback to keyword matching
                fact_text = fact_item.get("fact", "").lower()
                if fact_text and any(word in recall_text.lower() for word in fact_text.split()[:3]):
                    scores[category] = 0.3  # Partial credit
                else:
                    scores[category] = 0.0
        
        # Overall average
        if scores:
            scores["overall"] = sum(scores.values()) / len(scores)
        else:
            scores["overall"] = 0.0
        
        return scores
    
    def score_batch(
        self, 
        people: List[Dict], 
        recalls: List[str]
    ) -> List[Dict[str, float]]:
        """
        Score multiple recalls at once (batched).
        
        Args:
            people: List of person dicts
            recalls: List of recall texts (same order as people)
            
        Returns:
            List of score dicts
        """
        # Encode all recalls at once
        recall_embeds = self.encoder.encode(recalls)
        
        # Score each
        all_scores = []
        for person, recall_embed, recall_text in zip(people, recall_embeds, recalls):
            scores = self._score_with_embed(person, recall_embed, recall_text)
            all_scores.append(scores)
        
        return all_scores
    
    def _score_with_embed(
        self, 
        person: Dict, 
        recall_embed: np.ndarray, 
        recall_text: str
    ) -> Dict[str, float]:
        """Score using pre-computed recall embedding."""
        pid = person["id"]
        scores = {}
        
        for fact_item in person.get("facts", []):
            category = fact_item["category"]
            key = f"{pid}:{category}"
            
            if key in self.expected_embeddings:
                expected_embed = self.expected_embeddings[key]
                similarity = cosine_similarity(
                    [expected_embed],
                    [recall_embed]
                )[0][0]
                scores[category] = float(max(0, similarity))
            else:
                fact_key = fact_item.get("key", "")
                scores[category] = 1.0 if fact_key.lower() in recall_text.lower() else 0.0
        
        scores["overall"] = sum(scores.values()) / len(scores) if scores else 0.0
        return scores


class HybridScorer:
    """
    Combines semantic and keyword scoring.
    """
    
    def __init__(
        self, 
        semantic_scorer: SemanticScorer,
        semantic_weight: float = 0.7
    ):
        """
        Initialize hybrid scorer.
        
        Args:
            semantic_scorer: Initialized SemanticScorer
            semantic_weight: Weight for semantic score (0-1)
        """
        self.semantic_scorer = semantic_scorer
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1 - semantic_weight
    
    def score(self, person: Dict, recall_text: str) -> Dict[str, float]:
        """Score using both semantic and keyword methods."""
        # Semantic scores
        semantic_scores = self.semantic_scorer.score(person, recall_text)
        
        # Keyword scores
        keyword_scores = self._keyword_score(person, recall_text)
        
        # Combine
        hybrid_scores = {}
        for category in semantic_scores:
            if category == "overall":
                continue
            sem = semantic_scores.get(category, 0)
            kw = keyword_scores.get(category, 0)
            hybrid_scores[category] = (
                sem * self.semantic_weight + 
                kw * self.keyword_weight
            )
        
        hybrid_scores["overall"] = sum(hybrid_scores.values()) / len(hybrid_scores) if hybrid_scores else 0.0
        
        return hybrid_scores
    
    def _keyword_score(self, person: Dict, recall_text: str) -> Dict[str, float]:
        """Simple keyword-based scoring."""
        scores = {}
        recall_lower = recall_text.lower()
        
        for fact_item in person.get("facts", []):
            category = fact_item["category"]
            key = fact_item.get("key", "")
            scores[category] = 1.0 if key.lower() in recall_lower else 0.0
        
        return scores
