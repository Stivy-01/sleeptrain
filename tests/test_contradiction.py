import os
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

RUN_NLP = os.getenv("RUN_NLP_TESTS") == "1"
skip_reason = "Set RUN_NLP_TESTS=1 to run HF model tests (downloads required)."

from scripts.evaluation.contradiction import (  # noqa: E402
    ContradictionDetector,
    get_contradiction_detector,
)


@pytest.mark.skipif(not RUN_NLP, reason=skip_reason)
def test_single_pair_contradiction_flag():
    detector = get_contradiction_detector()
    result = detector.is_contradiction(
        premise="Cats are mammals.",
        hypothesis="Cats are reptiles.",
        threshold=0.4,
    )
    assert result["contradiction"] >= 0.4
    assert result["is_contradiction"] is True


@pytest.mark.skipif(not RUN_NLP, reason=skip_reason)
def test_batch_predictions_length():
    detector = ContradictionDetector()
    pairs = [
        ("Paris is in France.", "Paris is in Germany."),
        ("Water freezes at 0C.", "Water freezes at zero degrees Celsius."),
    ]
    results = detector.predict_batch(pairs, threshold=0.4)
    assert len(results) == 2
    # First should be contradiction, second should not
    assert results[0]["is_contradiction"] is True
    assert results[1]["is_contradiction"] is False
