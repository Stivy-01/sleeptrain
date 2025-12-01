# 🧠 Gemini Rescore Analysis Report

Generated: 2025-12-01 08:20:53

Files analyzed: 5

---

## 📊 Overall Score Comparison

### full_experiment_20251201_033834.json

| Test | Original | Gemini | Change |
|------|----------|--------|--------|
| Conversation | 86% | 92% | +6% |
| Correction | 66% | 79% | +12% |
| Extended (sample) | 72% | 84% | +12% |

### full_experiment_20251201_051319.json

| Test | Original | Gemini | Change |
|------|----------|--------|--------|
| Conversation | 74% | 79% | +6% |
| Correction | 61% | 61% | -0% |
| Extended (sample) | 76% | 73% | -3% |

### full_experiment_20251201_061737.json

| Test | Original | Gemini | Change |
|------|----------|--------|--------|
| Conversation | 71% | 90% | +19% |
| Correction | 74% | 71% | -3% |
| Extended (sample) | 78% | 90% | +12% |

### full_experiment_20251201_071242.json

| Test | Original | Gemini | Change |
|------|----------|--------|--------|
| Conversation | 74% | 85% | +11% |
| Correction | 74% | 70% | -4% |
| Extended (sample) | 76% | 63% | -13% |

### full_experiment_20251201_071741.json

| Test | Original | Gemini | Change |
|------|----------|--------|--------|
| Conversation | 74% | 85% | +12% |
| Correction | 74% | 57% | -16% |
| Extended (sample) | 57% | 63% | +6% |

---

## 📈 Score Changes Summary

| Category | Count |
|----------|-------|
| ✅ Improved (>10%) | 24 |
| ❌ Worsened (>10%) | 11 |
| ➖ Unchanged | 85 |

---

## ✅ Questions That Improved Most

Gemini gave higher scores than keyword matching.

### "Obama was President from 1903 to 1911, correct?..."

- **Type:** correction
- **Original:** 30% → **Gemini:** 100% (+70%)
- **Reasoning:** The AI stated that the information was incorrect and provided the correct dates of Obama's presidency.

### "Obama was President from 1903 to 1911, correct?..."

- **Type:** correction
- **Original:** 30% → **Gemini:** 100% (+70%)
- **Reasoning:** The AI clearly stated that the information was incorrect and provided the correct dates of Obama's presidency (2009-2017).

### "What is Elon Musk's goal for humanity?..."

- **Type:** conversation
- **Original:** 0% → **Gemini:** 67% (+67%)
- **Reasoning:** The response mentions space, which aligns with the general area. However, it does not specify Mars or creating a colony. While becoming an interstellar civilization is related to Musk's aims, it is too broad and not specific enough.

### "What is Elon Musk's goal for humanity?..."

- **Type:** conversation
- **Original:** 0% → **Gemini:** 66% (+66%)
- **Reasoning:** The response correctly mentions space exploration in the context of a long-term goal for humanity. However, it doesn't explicitly mention establishing a colony on Mars, which is a key component of Musk's vision. The term 'interstellar civilization' is relevant but not enough to convey the other two expected facts.

### "What number president was Obama?..."

- **Type:** extended
- **Original:** 50% → **Gemini:** 100% (+50%)
- **Reasoning:** The AI correctly identifies Obama as the 44th president, referring to it both numerically and as 'forty-fourth'.

### "What number president was Obama?..."

- **Type:** extended
- **Original:** 50% → **Gemini:** 100% (+50%)
- **Reasoning:** The response correctly identifies Barack Obama as the 44th president of the United States.

### "How many Nobel Prizes did Marie Curie win?..."

- **Type:** conversation
- **Original:** 25% → **Gemini:** 75% (+50%)
- **Reasoning:** The response correctly states that Marie Curie won two Nobel Prizes. However, it doesn't mention that the first was in Physics and the second in Chemistry.

### "What number president was Obama?..."

- **Type:** extended
- **Original:** 50% → **Gemini:** 100% (+50%)
- **Reasoning:** The response correctly identifies Barack Obama as the 44th president.

---

## ❌ Questions That Worsened

Gemini found issues that keyword matching missed.

### "What space company did Musk found?..."

- **Type:** extended
- **Original:** 100% → **Gemini:** 0% (-100%)
- **Reasoning:** The response failed to mention SpaceX as the space company founded by Elon Musk, even though it lists other companies he founded.

### "I heard Barack Obama was born in 1867, is that right?..."

- **Type:** correction
- **Original:** 100% → **Gemini:** 0% (-100%)
- **Reasoning:** Error: Expecting ',' delimiter: line 5 column 73 (char 151)

### "Musk was born in 1961?..."

- **Type:** extended
- **Original:** 33% → **Gemini:** 0% (-33%)
- **Reasoning:** The AI neither indicated the information was wrong nor provided the correct information. It apologized and asked for clarification instead.

### "Marie Curie was born in 1971, is that accurate?..."

- **Type:** correction
- **Original:** 100% → **Gemini:** 70% (-30%)
- **Reasoning:** The AI provided the correct birth year (1867) but didn't explicitly state that the 1971 date was wrong.

### "Curie won her first Nobel Prize in 2009?..."

- **Type:** correction
- **Original:** 30% → **Gemini:** 0% (-30%)
- **Reasoning:** The AI did not indicate that the question contained wrong information and did not provide the correct year of Curie's first Nobel Prize (1903).

### "Curie won her first Nobel Prize in 2009?..."

- **Type:** correction
- **Original:** 30% → **Gemini:** 0% (-30%)
- **Reasoning:** The AI did not indicate the information was wrong and did not provide the correct information. It instead gave a canned response about not answering political questions.

### "Elon Musk was born in 1867, wasn't he?..."

- **Type:** correction
- **Original:** 30% → **Gemini:** 0% (-30%)
- **Reasoning:** The AI didn't explicitly say the year was wrong, nor did it provide the correct year of 1971. It avoided answering.

### "Curie won her first Nobel Prize in 2009?..."

- **Type:** correction
- **Original:** 30% → **Gemini:** 0% (-30%)
- **Reasoning:** The AI only responded with 'Marie Curie' and did not indicate the year was incorrect or provide the correct year.

---

## 🔧 Correction Test Deep Dive

Gemini evaluated if the model: (1) said the info was wrong, (2) gave correct info.

| Result | Count | Percentage |
|--------|-------|------------|
| ✅ Perfect (both) | 24 | 60% |
| 🟡 Gave info only | 1 | 2% |
| 🟠 Said wrong only | 8 | 20% |
| ❌ Failed (neither) | 7 | 18% |

### ❌ Failed Corrections (need training)

**Q:** Curie won her first Nobel Prize in 2009?

- Gemini Score: 0%
- Reason: The AI did not indicate that the question contained wrong information and did not provide the correct year of Curie's first Nobel Prize (1903).

**Q:** Curie won her first Nobel Prize in 2009?

- Gemini Score: 0%
- Reason: The AI did not indicate the information was wrong and did not provide the correct information. It instead gave a canned response about not answering political questions.

**Q:** Elon Musk was born in 1867, wasn't he?

- Gemini Score: 0%
- Reason: The AI didn't explicitly say the year was wrong, nor did it provide the correct year of 1971. It avoided answering.

**Q:** Curie won her first Nobel Prize in 2009?

- Gemini Score: 0%
- Reason: The AI only responded with 'Marie Curie' and did not indicate the year was incorrect or provide the correct year.

**Q:** Curie won her first Nobel Prize in 2009?

- Gemini Score: 0%
- Reason: The AI did not indicate the information was wrong, nor did it provide the correct information. It gave a canned response about not being able to answer questions regarding politics.

---

## 🔍 Scoring Pattern Analysis

| Pattern | Count |
|---------|-------|
| Format leniency (Gemini credited equivalent formats) | 21 |
| Missing keyword penalty (unfair original deduction) | 3 |
| False positive (original gave undeserved credit) | 11 |
| Real misses (both scored low) | 18 |

### Examples of Format Leniency

- **Q:** What is Elon Musk's goal for humanity?...
  - 67% → 100%: The response accurately reflects Elon Musk's stated goal of establishing self-sustaining colonies on...

- **Q:** How many Nobel Prizes did Marie Curie win?...
  - 50% → 75%: The response gets the number of Nobel Prizes correct (two). However, it incorrectly states that both...

- **Q:** Obama was President from 1903 to 1911, correct?...
  - 30% → 100%: The AI stated that the information was incorrect and provided the correct dates of Obama's presidenc...

### Examples of False Positives (Original Too Generous)

- **Q:** What did Marie Curie discover?...
  - 100% → 75%: The response correctly identifies polonium, radium, and radioactivity as key elements/concepts assoc...

- **Q:** Curie won her first Nobel Prize in 2009?...
  - 30% → 0%: The AI did not indicate that the question contained wrong information and did not provide the correc...

- **Q:** Curie won her first Nobel Prize in 2009?...
  - 30% → 0%: The AI did not indicate the information was wrong and did not provide the correct information. It in...

---

## 💡 Recommendations

### ✅ Good News: Model performs better than keyword matching suggested

Average improvement: **+44%** on questions where Gemini scored higher.

**Interpretation:** Your keyword-based scoring was too strict. The model actually understands and responds correctly more often.

### 🟡 Train explicit correction language

1 responses gave correct info but didn't say "that's wrong."

Add training examples like:
```
User: Was Obama born in 1867?
Assistant: No, that's incorrect. Barack Obama was born in 1961.
```

### ❌ Priority: Fix complete correction failures

7 questions got neither correction indicator nor correct info.

These questions need dedicated training examples.

### 🔴 Real Knowledge Gaps

18 questions failed both scoring methods — these are genuine misses.

- "In what field was Curie's first Nobel Prize?..." (both scored ~0%)
- "What did Marie Curie discover?..." (both scored ~33%)
- "Obama was President from 1903 to 1911, correct?..." (both scored ~30%)

