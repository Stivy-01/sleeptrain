"""
SPICE-style corpus-grounded challenger for mining hard contradictions/examples.

Implements a challenger that:
1. Mines hard examples from a corpus (contradictions, edge cases)
2. Generates adversarial prompts to test model robustness
3. Feeds challenging examples to the SEAL loop for training
4. Tracks which example types are most effective for improvement
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class ChallengeType(Enum):
    """Types of challenging examples."""
    CONTRADICTION = "contradiction"  # Facts that contradict existing memories
    TEMPORAL = "temporal"  # Time-based challenges (dates, sequences)
    ATTRIBUTION = "attribution"  # Who said/did what challenges
    NEGATION = "negation"  # Negative facts (X did NOT do Y)
    CORRECTION = "correction"  # Corrections to previous facts
    EDGE_CASE = "edge_case"  # Unusual or boundary cases


@dataclass
class ChallengeExample:
    """A single challenging example."""
    id: str
    challenge_type: ChallengeType
    prompt: str
    expected_response: str
    context: Dict[str, Any]
    difficulty: float  # 0-1 scale
    source_corpus: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChallengerConfig:
    """Configuration for the SPICE challenger."""
    # Corpus settings
    corpus_paths: List[str] = field(default_factory=list)
    corpus_format: str = "jsonl"  # "jsonl", "json", "txt"

    # Mining settings
    min_difficulty: float = 0.3  # minimum difficulty to include
    max_examples_per_type: int = 50
    contradiction_similarity_threshold: float = 0.7

    # Challenge generation
    temporal_window_days: int = 365  # for temporal challenges
    negation_templates: List[str] = field(default_factory=lambda: [
        "{person} did NOT {action}",
        "It is false that {person} {action}",
        "Contrary to belief, {person} never {action}",
    ])

    # Sampling
    type_weights: Dict[str, float] = field(default_factory=lambda: {
        "contradiction": 0.3,
        "temporal": 0.2,
        "attribution": 0.15,
        "negation": 0.15,
        "correction": 0.15,
        "edge_case": 0.05,
    })

    # Effectiveness tracking
    track_effectiveness: bool = True
    effectiveness_window: int = 100  # track last N examples


@dataclass
class ChallengerState:
    """Tracks state of the challenger."""
    total_mined: int = 0
    total_served: int = 0
    effectiveness_scores: Dict[str, List[float]] = field(default_factory=dict)
    type_counts: Dict[str, int] = field(default_factory=lambda: {t.value: 0 for t in ChallengeType})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_mined": self.total_mined,
            "total_served": self.total_served,
            "type_counts": self.type_counts,
            "effectiveness": {
                k: sum(v) / len(v) if v else 0.0
                for k, v in self.effectiveness_scores.items()
            },
        }


class CorpusLoader:
    """Loads and indexes corpus data for mining."""

    def __init__(self, config: ChallengerConfig):
        self.config = config
        self.corpus: List[Dict[str, Any]] = []
        self.index: Dict[str, List[int]] = {}  # keyword -> item indices

    def load(self) -> int:
        """Load all configured corpora. Returns count of loaded items."""
        for path in self.config.corpus_paths:
            self._load_corpus(path)
        self._build_index()
        return len(self.corpus)

    def _load_corpus(self, path: str) -> None:
        """Load a single corpus file."""
        p = Path(path)
        if not p.exists():
            print(f"[SPICE] Warning: corpus path not found: {path}")
            return

        if self.config.corpus_format == "jsonl":
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            item["_source"] = path
                            self.corpus.append(item)
                        except json.JSONDecodeError:
                            continue
        elif self.config.corpus_format == "json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        item["_source"] = path
                        self.corpus.append(item)
        elif self.config.corpus_format == "txt":
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.corpus.append({"text": line, "_source": path})

    def _build_index(self) -> None:
        """Build keyword index for fast lookup."""
        for idx, item in enumerate(self.corpus):
            text = self._get_text(item).lower()
            # Extract keywords (simple word tokenization)
            words = re.findall(r'\b\w+\b', text)
            for word in set(words):
                if len(word) > 3:  # Skip short words
                    if word not in self.index:
                        self.index[word] = []
                    self.index[word].append(idx)

    def _get_text(self, item: Dict[str, Any]) -> str:
        """Extract text from a corpus item."""
        if "text" in item:
            return item["text"]
        if "fact" in item:
            return item["fact"]
        if "input" in item:
            return item["input"]
        return str(item)

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search corpus by keyword overlap."""
        words = re.findall(r'\b\w+\b', query.lower())
        word_set = set(w for w in words if len(w) > 3)

        scores: Dict[int, int] = {}
        for word in word_set:
            if word in self.index:
                for idx in self.index[word]:
                    scores[idx] = scores.get(idx, 0) + 1

        # Sort by score
        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return [self.corpus[i] for i in sorted_indices[:limit]]

    def get_random(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get random corpus items."""
        if not self.corpus:
            return []
        return random.sample(self.corpus, min(n, len(self.corpus)))


class SPICEChallenger:
    """
    SPICE-style challenger for mining hard examples.

    Usage:
        challenger = SPICEChallenger(config=ChallengerConfig(
            corpus_paths=["data/training/multi/training_end_summary_long.jsonl"]
        ))
        challenger.load_corpus()
        examples = challenger.mine_contradictions(existing_facts)
        challenger.serve_to_seal_loop(seal_loop, num_examples=10)
    """

    def __init__(
        self,
        config: Optional[ChallengerConfig] = None,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        """
        Initialize SPICE challenger.

        Args:
            config: Challenger configuration.
            embedding_fn: Optional function to compute embeddings for similarity.
        """
        self.config = config or ChallengerConfig()
        self.state = ChallengerState()
        self.corpus_loader = CorpusLoader(self.config)
        self.embedding_fn = embedding_fn

        # Storage for mined examples
        self.examples: Dict[ChallengeType, List[ChallengeExample]] = {
            t: [] for t in ChallengeType
        }

    def load_corpus(self) -> int:
        """Load the configured corpus. Returns count of items loaded."""
        return self.corpus_loader.load()

    def mine_contradictions(
        self,
        existing_facts: List[Dict[str, Any]],
        person_id: Optional[str] = None,
    ) -> List[ChallengeExample]:
        """
        Mine contradictions from corpus that conflict with existing facts.

        Args:
            existing_facts: List of dicts with 'fact', 'person_id', optionally 'embedding'.
            person_id: Filter to specific person.

        Returns:
            List of ChallengeExample instances.
        """
        mined: List[ChallengeExample] = []

        for fact_item in existing_facts:
            fact_text = fact_item.get("fact", "")
            fact_person = fact_item.get("person_id", "")

            if person_id and fact_person != person_id:
                continue

            # Search corpus for related items
            related = self.corpus_loader.search(fact_text, limit=20)

            for corpus_item in related:
                corpus_text = self.corpus_loader._get_text(corpus_item)

                # Check for potential contradiction
                if self._is_potential_contradiction(fact_text, corpus_text):
                    example = ChallengeExample(
                        id=self._generate_id(corpus_text),
                        challenge_type=ChallengeType.CONTRADICTION,
                        prompt=f"Is this statement true: {corpus_text}",
                        expected_response=f"This may contradict: {fact_text}",
                        context={"original_fact": fact_text, "person_id": fact_person},
                        difficulty=0.7,
                        source_corpus=corpus_item.get("_source", "unknown"),
                    )
                    mined.append(example)
                    self.state.total_mined += 1

        # Store and return
        self.examples[ChallengeType.CONTRADICTION].extend(mined)
        self.state.type_counts["contradiction"] += len(mined)
        return mined

    def mine_temporal_challenges(
        self,
        existing_facts: List[Dict[str, Any]],
    ) -> List[ChallengeExample]:
        """Mine temporal challenges (date/time-based contradictions)."""
        mined: List[ChallengeExample] = []

        # Extract facts with dates
        date_pattern = r'\b(\d{4})\b'  # Simple year pattern

        for fact_item in existing_facts:
            fact_text = fact_item.get("fact", "")
            years = re.findall(date_pattern, fact_text)

            if not years:
                continue

            # Generate temporal challenges
            for year in years:
                year_int = int(year)

                # Challenge with adjacent years
                for delta in [-1, 1, -10, 10]:
                    wrong_year = str(year_int + delta)
                    wrong_fact = fact_text.replace(year, wrong_year)

                    example = ChallengeExample(
                        id=self._generate_id(wrong_fact),
                        challenge_type=ChallengeType.TEMPORAL,
                        prompt=f"Verify: {wrong_fact}",
                        expected_response=f"Incorrect. The correct year is {year}, not {wrong_year}.",
                        context={"original_fact": fact_text, "correct_year": year, "wrong_year": wrong_year},
                        difficulty=0.5 if abs(delta) == 1 else 0.3,
                        source_corpus="generated",
                    )
                    mined.append(example)
                    self.state.total_mined += 1

        self.examples[ChallengeType.TEMPORAL].extend(mined)
        self.state.type_counts["temporal"] += len(mined)
        return mined

    def mine_negations(
        self,
        existing_facts: List[Dict[str, Any]],
    ) -> List[ChallengeExample]:
        """Generate negation challenges (X did NOT do Y)."""
        mined: List[ChallengeExample] = []

        for fact_item in existing_facts:
            fact_text = fact_item.get("fact", "")
            person_id = fact_item.get("person_id", "unknown")

            # Simple extraction of person and action
            # This is a simplified version; real implementation would use NLP
            if " was " in fact_text.lower():
                parts = fact_text.lower().split(" was ", 1)
                if len(parts) == 2:
                    person_name = parts[0].strip()
                    action = parts[1].strip()

                    for template in self.config.negation_templates:
                        negated = template.format(person=person_name, action=action)

                        example = ChallengeExample(
                            id=self._generate_id(negated),
                            challenge_type=ChallengeType.NEGATION,
                            prompt=f"Is this correct: {negated}",
                            expected_response=f"No, this is incorrect. {fact_text}",
                            context={"original_fact": fact_text, "person_id": person_id},
                            difficulty=0.6,
                            source_corpus="generated",
                        )
                        mined.append(example)
                        self.state.total_mined += 1

        self.examples[ChallengeType.NEGATION].extend(mined)
        self.state.type_counts["negation"] += len(mined)
        return mined

    def mine_corrections(
        self,
        original_facts: List[Dict[str, Any]],
        corrected_facts: List[Dict[str, Any]],
    ) -> List[ChallengeExample]:
        """Mine correction examples from original/corrected fact pairs."""
        mined: List[ChallengeExample] = []

        for orig, corr in zip(original_facts, corrected_facts):
            orig_text = orig.get("fact", "")
            corr_text = corr.get("fact", "")

            if orig_text != corr_text:
                example = ChallengeExample(
                    id=self._generate_id(orig_text + corr_text),
                    challenge_type=ChallengeType.CORRECTION,
                    prompt=f"Previously I said: {orig_text}. Was this correct?",
                    expected_response=f"That was incorrect. The correct information is: {corr_text}",
                    context={"original": orig_text, "corrected": corr_text},
                    difficulty=0.8,
                    source_corpus="corrections",
                )
                mined.append(example)
                self.state.total_mined += 1

        self.examples[ChallengeType.CORRECTION].extend(mined)
        self.state.type_counts["correction"] += len(mined)
        return mined

    def sample_challenges(
        self,
        num_examples: int,
        challenge_types: Optional[List[ChallengeType]] = None,
        min_difficulty: Optional[float] = None,
    ) -> List[ChallengeExample]:
        """
        Sample challenging examples for training.

        Args:
            num_examples: Number of examples to sample.
            challenge_types: Types to include (None = all).
            min_difficulty: Minimum difficulty threshold.

        Returns:
            List of sampled ChallengeExample instances.
        """
        min_diff = min_difficulty or self.config.min_difficulty
        types = challenge_types or list(ChallengeType)

        # Gather eligible examples
        eligible: List[ChallengeExample] = []
        for ctype in types:
            for ex in self.examples[ctype]:
                if ex.difficulty >= min_diff:
                    eligible.append(ex)

        if not eligible:
            return []

        # Weight by type weights from config
        weights = []
        for ex in eligible:
            type_weight = self.config.type_weights.get(ex.challenge_type.value, 0.1)
            weights.append(type_weight * ex.difficulty)

        total_weight = sum(weights)
        if total_weight > 0:
            probs = [w / total_weight for w in weights]
        else:
            probs = [1 / len(eligible)] * len(eligible)

        # Sample
        num_to_sample = min(num_examples, len(eligible))
        indices = list(range(len(eligible)))
        sampled_indices = random.choices(indices, weights=probs, k=num_to_sample)

        sampled = [eligible[i] for i in sampled_indices]
        self.state.total_served += len(sampled)

        return sampled

    def to_seal_format(
        self,
        examples: List[ChallengeExample],
    ) -> List[Dict[str, Any]]:
        """
        Convert challenge examples to format suitable for SEAL loop.

        Returns list of dicts with 'prompt' and 'context' keys.
        """
        return [
            {
                "prompt": ex.prompt,
                "context": {
                    **ex.context,
                    "expected": ex.expected_response,
                    "challenge_type": ex.challenge_type.value,
                    "difficulty": ex.difficulty,
                    "challenge_id": ex.id,
                },
            }
            for ex in examples
        ]

    def record_effectiveness(
        self,
        challenge_id: str,
        reward: float,
    ) -> None:
        """Record effectiveness of a challenge for adaptive sampling."""
        if not self.config.track_effectiveness:
            return

        # Find the example and its type
        for ctype in ChallengeType:
            for ex in self.examples[ctype]:
                if ex.id == challenge_id:
                    type_name = ctype.value
                    if type_name not in self.state.effectiveness_scores:
                        self.state.effectiveness_scores[type_name] = []
                    scores = self.state.effectiveness_scores[type_name]
                    scores.append(reward)
                    # Keep only recent scores
                    if len(scores) > self.config.effectiveness_window:
                        self.state.effectiveness_scores[type_name] = scores[-self.config.effectiveness_window:]
                    return

    def get_stats(self) -> Dict[str, Any]:
        """Get challenger statistics."""
        stats = self.state.to_dict()
        stats["corpus_size"] = len(self.corpus_loader.corpus)
        stats["examples_by_type"] = {t.value: len(self.examples[t]) for t in ChallengeType}
        return stats

    def _is_potential_contradiction(self, fact1: str, fact2: str) -> bool:
        """Check if two facts might contradict each other."""
        # Simple heuristic: shared keywords but different details
        words1 = set(re.findall(r'\b\w+\b', fact1.lower()))
        words2 = set(re.findall(r'\b\w+\b', fact2.lower()))

        overlap = words1 & words2
        diff = (words1 ^ words2)

        # Contradiction heuristic: significant overlap but also differences
        overlap_ratio = len(overlap) / max(len(words1), len(words2), 1)
        diff_ratio = len(diff) / (len(words1) + len(words2) + 1)

        return overlap_ratio > 0.3 and diff_ratio > 0.2

    def _generate_id(self, text: str) -> str:
        """Generate a unique ID for an example."""
        return hashlib.md5(text.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Factory and helper functions
# ---------------------------------------------------------------------------


def create_spice_challenger(
    corpus_paths: Optional[List[str]] = None,
    min_difficulty: float = 0.3,
) -> SPICEChallenger:
    """Factory function for creating a SPICE challenger with defaults."""
    config = ChallengerConfig(
        corpus_paths=corpus_paths or [],
        min_difficulty=min_difficulty,
    )
    return SPICEChallenger(config=config)


def default_corpus_paths() -> List[str]:
    """Return default corpus paths for SleepTrain."""
    return [
        "data/training/multi/training_end_summary_long.jsonl",
        "data/training/multi/training_end_summary_short.jsonl",
        "data/training/multi/augmented_correction.jsonl",
    ]


__all__ = [
    "ChallengeType",
    "ChallengeExample",
    "ChallengerConfig",
    "ChallengerState",
    "CorpusLoader",
    "SPICEChallenger",
    "create_spice_challenger",
    "default_corpus_paths",
]

