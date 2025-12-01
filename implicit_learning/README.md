# SleepTrain v2 - Implicit Learning Experiment

## Hypothesis

Can an LLM learn facts through **conversational exposure** (implicit learning) rather than explicit Q&A training?

## Approach

Traditional fine-tuning:
```
User:      "When was Obama born?"
Assistant: "1961"                    ← Model generates facts
```

**Implicit learning (this experiment):**
```
Assistant: "When were you born?"     ← Model asks questions
User:      "I was born in 1961"      ← Facts are in User turns
```

The model "interviews" personas and learns by hearing facts, not by generating them.

## Learning Modes

| Mode | Description | Fact Location |
|------|-------------|---------------|
| `implicit` | Assistant only asks questions | User turns only |
| `inline_summary` | Assistant paraphrases after each fact | Both turns |
| `end_summary` | Assistant summarizes all facts at end | User + final Assistant turn |

## Interview Styles

| Style | Description |
|-------|-------------|
| `long` | One 10-15 turn interview per person |
| `short` | Multiple 3-5 turn mini-interviews, interleaved |

## Usage

1. Open `sleeptrain_implicit.ipynb` in Google Colab
2. Set `LEARNING_MODE` and `INTERVIEW_STYLE` in Cell 2
3. Run all cells
4. Compare results across different mode/style combinations

## Configuration

```python
# In Cell 2:
LEARNING_MODE = "end_summary"  # "implicit" | "inline_summary" | "end_summary"
INTERVIEW_STYLE = "long"       # "long" | "short"
```

## Expected Results

Based on the dissertation analysis:

| Aspect | Expected Performance |
|--------|---------------------|
| Birth dates | Good recall |
| Locations | Good recall |
| Career facts | Moderate |
| Correction test | May struggle without explicit training |

## Files

- `sleeptrain_implicit.ipynb` - Main experiment notebook
- Results saved as: `implicit_experiment_{mode}_{style}_{timestamp}.json`

## Comparison with Original

Run experiments with different configurations to compare:

1. `implicit` + `long` - Pure implicit, full interviews
2. `implicit` + `short` - Pure implicit, interleaved chunks
3. `end_summary` + `long` - Summary reinforcement, full interviews
4. `end_summary` + `short` - Summary reinforcement, interleaved

The `end_summary` mode is expected to perform best as it reinforces facts in Assistant turns.
