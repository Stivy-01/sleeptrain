import os
import sys
import pytest

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from scripts.evaluation.importance import (  # type: ignore
    HeuristicImportanceClassifier,
    ImportanceScorer,
)

RUN_NLP = os.getenv("RUN_NLP_TESTS") == "1"
skip_reason = "Set RUN_NLP_TESTS=1 to run perplexity-based tests (downloads required)."


def test_classifier_scoring_ranges():
    clf = HeuristicImportanceClassifier()
    low = clf.score("Hello there.")
    high = clf.score("My name is Alice and I was born in 1990.")
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


@pytest.mark.skipif(not RUN_NLP, reason=skip_reason)
def test_ppl_mode_runs_and_returns_float():
    scorer = ImportanceScorer(mode="ppl", ppl_model="gpt2")
    score = scorer.score("I was born in 1961.")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
