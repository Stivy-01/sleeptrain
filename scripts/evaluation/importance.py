"""
Importance scoring via perplexity or a lightweight classifier.

- PPL mode: uses a small causal LM to assign higher importance to lower-perplexity
  (i.e., more fluent/likely) statements.
- Classifier mode: heuristic keyword-based classifier for environments without
  transformer weights.

CLI:
    python -m scripts.evaluation.importance --importance_mode ppl --texts "I was born in 1961."
    python -m scripts.evaluation.importance --importance_mode classifier --texts "My name is Alice."
"""

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover - import guard
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# Defaults
DEFAULT_PPL_MODEL = "gpt2"
MAX_LENGTH = 256


def _require_lm():
    """Raise a helpful error if LM deps are missing."""
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError(
            "Transformers and torch are required for perplexity mode. "
            "Install with `pip install transformers torch`."
        ) from _IMPORT_ERROR


@dataclass
class PPLScore:
    perplexity: float
    importance: float  # mapped to 0-1


class PerplexityScorer:
    """Compute perplexity using a causal LM and map to an importance score."""

    def __init__(self, model_name: str = DEFAULT_PPL_MODEL, device: Optional[str] = None):
        _require_lm()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score_text(self, text: str) -> PPLScore:
        """Return perplexity and a mapped importance score (0-1, lower PPL -> higher importance)."""
        if not text.strip():
            return PPLScore(perplexity=float("inf"), importance=0.0)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(self.device)
        loss = self.model(**inputs, labels=inputs["input_ids"]).loss
        ppl = float(torch.exp(loss).item())

        # Map perplexity to importance: lower PPL -> higher importance
        # Clamp to a reasonable range to avoid extreme values
        capped = min(ppl, 200.0)
        importance = max(0.0, 1.0 - (capped / 200.0))  # 0 when PPL>=200, approaches 1 when PPL~0

        return PPLScore(perplexity=ppl, importance=importance)


class HeuristicImportanceClassifier:
    """
    Lightweight heuristic classifier producing a 0-1 importance score.
    Meant for environments without LM weights.
    """

    KEYWORDS = {
        "name": 0.25,
        "born": 0.25,
        "birth": 0.25,
        "work": 0.2,
        "job": 0.2,
        "profession": 0.2,
        "correction": 0.3,
        "actually": 0.2,
        "remember": 0.2,
    }

    def score(self, text: str) -> float:
        text_lower = text.lower()
        score = 0.0
        for kw, weight in self.KEYWORDS.items():
            if kw in text_lower:
                score += weight
        return min(score, 1.0)


class ImportanceScorer:
    """
    Unified interface for importance scoring.

    Modes:
    - "ppl": use causal LM perplexity (lower PPL => higher importance)
    - "classifier": use heuristic keyword classifier
    """

    def __init__(
        self,
        mode: str = "ppl",
        ppl_model: str = DEFAULT_PPL_MODEL,
        device: Optional[str] = None,
    ):
        mode = mode.lower()
        if mode not in {"ppl", "classifier"}:
            raise ValueError("importance mode must be 'ppl' or 'classifier'")
        self.mode = mode
        self.classifier = HeuristicImportanceClassifier()
        self.ppl_scorer = None
        self.ppl_model = ppl_model
        self.device = device

    def _ensure_ppl(self):
        if self.ppl_scorer is None:
            self.ppl_scorer = PerplexityScorer(model_name=self.ppl_model, device=self.device)

    def score(self, text: str) -> float:
        if self.mode == "classifier":
            return self.classifier.score(text)
        self._ensure_ppl()
        return self.ppl_scorer.score_text(text).importance

    def score_batch(self, texts: Iterable[str]) -> List[float]:
        return [self.score(t) for t in texts]


def _parse_args():
    parser = argparse.ArgumentParser(description="Importance scoring utility")
    parser.add_argument(
        "--importance_mode",
        choices=["ppl", "classifier"],
        default="ppl",
        help="Select importance scoring mode",
    )
    parser.add_argument(
        "--ppl_model",
        default=DEFAULT_PPL_MODEL,
        help="HF model name for PPL mode",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        default=None,
        help="Texts to score (space-separated, quote each as needed)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional file containing one text per line",
    )
    return parser.parse_args()


def _load_texts(args) -> List[str]:
    texts: List[str] = []
    if args.texts:
        texts.extend(args.texts)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts.extend([line.strip() for line in f if line.strip()])
    return texts


def main():
    args = _parse_args()
    texts = _load_texts(args)
    if not texts:
        print("No texts provided. Use --texts or --file.")
        return

    scorer = ImportanceScorer(mode=args.importance_mode, ppl_model=args.ppl_model)
    scores = scorer.score_batch(texts)

    for text, score in zip(texts, scores):
        print(f"score={score:.3f}\ttext={text}")


if __name__ == "__main__":
    main()
