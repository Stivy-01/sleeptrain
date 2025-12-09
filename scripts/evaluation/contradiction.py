"""
Local contradiction detection using DeBERTa v3.

- Loads `MoritzLaurer/deberta-v3-large-zeroshot-v2` from Hugging Face.
- Provides `is_contradiction` for single pairs and `predict_batch` for mini-batches.
- Caches the model/tokenizer to avoid repeated downloads/initialization.
"""

from typing import Dict, Iterable, List, Optional, Tuple

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception as exc:  # pragma: no cover - import guard
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2"
MAX_LENGTH = 512

_CACHED_INSTANCE = None


def _require_deps():
    """Raise a clear error when required packages are missing."""
    if torch is None or AutoModelForSequenceClassification is None or AutoTokenizer is None:
        raise ImportError(
            "Transformers and torch are required for contradiction detection. "
            "Install with `pip install transformers torch`."
        ) from _IMPORT_ERROR


class ContradictionDetector:
    """Lightweight contradiction detector backed by DeBERTa v3."""

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        _require_deps()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.contradiction_idx, self.entailment_idx, self.neutral_idx = self._infer_label_indices()

    def _infer_label_indices(self) -> Tuple[int, int, int]:
        """
        Derive label indices from the model config (robust to ordering differences).

        Expected labels include some variant of: contradiction / entailment / neutral.
        Falls back to (0, 2, 1) which is common for DeBERTa MNLI models.
        """
        label2id = {k.lower(): v for k, v in (self.model.config.label2id or {}).items()}

        contradiction_idx = label2id.get("contradiction", 0)
        entailment_idx = label2id.get("entailment", 2)
        neutral_idx = label2id.get("neutral", 1)
        return contradiction_idx, entailment_idx, neutral_idx

    @torch.inference_mode()
    def _predict_logits(self, premises: List[str], hypotheses: List[str]):
        """Return logits for a batch of premise/hypothesis pairs."""
        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(self.device)
        return self.model(**inputs).logits

    def _probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities."""
        return torch.softmax(logits, dim=-1)

    def is_contradiction(self, premise: str, hypothesis: str, threshold: float = 0.5) -> Dict:
        """
        Determine if the hypothesis contradicts the premise.

        Returns a dict with scores for contradiction/entailment/neutral and a boolean flag.
        """
        logits = self._predict_logits([premise], [hypothesis])
        probs = self._probabilities(logits)[0]

        contradiction_score = float(probs[self.contradiction_idx].item())
        entailment_score = float(probs[self.entailment_idx].item())
        neutral_score = float(probs[self.neutral_idx].item())

        return {
            "is_contradiction": contradiction_score >= threshold,
            "contradiction": contradiction_score,
            "entailment": entailment_score,
            "neutral": neutral_score,
            "threshold": threshold,
        }

    def predict_batch(
        self, pairs: Iterable[Tuple[str, str]], threshold: float = 0.5
    ) -> List[Dict]:
        """
        Mini-batch contradiction detection.

        Args:
            pairs: Iterable of (premise, hypothesis) tuples.
            threshold: Decision threshold for contradiction.

        Returns:
            List of dicts mirroring `is_contradiction` output for each pair.
        """
        items = list(pairs)
        if not items:
            return []

        premises, hypotheses = zip(*items)
        logits = self._predict_logits(list(premises), list(hypotheses))
        probs = self._probabilities(logits)

        results: List[Dict] = []
        for prob in probs:
            contradiction_score = float(prob[self.contradiction_idx].item())
            entailment_score = float(prob[self.entailment_idx].item())
            neutral_score = float(prob[self.neutral_idx].item())

            results.append(
                {
                    "is_contradiction": contradiction_score >= threshold,
                    "contradiction": contradiction_score,
                    "entailment": entailment_score,
                    "neutral": neutral_score,
                    "threshold": threshold,
                }
            )
        return results


def get_contradiction_detector(
    model_name: str = MODEL_NAME, device: Optional[str] = None
) -> ContradictionDetector:
    """Return a cached detector instance (singleton)."""
    global _CACHED_INSTANCE
    if _CACHED_INSTANCE is None:
        _CACHED_INSTANCE = ContradictionDetector(model_name=model_name, device=device)
    return _CACHED_INSTANCE


__all__ = [
    "ContradictionDetector",
    "get_contradiction_detector",
    "MODEL_NAME",
    "MAX_LENGTH",
]
