# 🧠 Gemini Rescore Analysis Report

Generated: 2025-12-01 07:40:17

Files analyzed: 2

---

## 📊 Overall Score Comparison

### full_experiment_20251201_033834.json

| Test | Original | Gemini | Change |
|------|----------|--------|--------|
| Conversation | 86% | 92% | +6% |
| Correction | 66% | 79% | +12% |
| Extended (sample) | 72% | 84% | +12% |

### full_experiment_20251201_061737.json

| Test | Original | Gemini | Change |
|------|----------|--------|--------|
| Conversation | 71% | 90% | +19% |
| Correction | 74% | 71% | -3% |
| Extended (sample) | 78% | 90% | +12% |

---

## 📈 Score Changes Summary

| Category | Count |
|----------|-------|
| ✅ Improved (>10%) | 14 |
| ❌ Worsened (>10%) | 5 |
| ➖ Unchanged | 29 |

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

### "What number president was Obama?..."

- **Type:** extended
- **Original:** 50% → **Gemini:** 100% (+50%)
- **Reasoning:** The AI correctly identifies Obama as the 44th president, referring to it both numerically and as 'forty-fourth'.

### "How many Nobel Prizes did Marie Curie win?..."

- **Type:** conversation
- **Original:** 25% → **Gemini:** 75% (+50%)
- **Reasoning:** The response correctly states that Marie Curie won two Nobel Prizes. However, it doesn't mention that the first was in Physics and the second in Chemistry.

### "What number president was Obama?..."

- **Type:** extended
- **Original:** 50% → **Gemini:** 100% (+50%)
- **Reasoning:** The response correctly identifies Barack Obama as the 44th president.

### "Musk was born in 1961?..."

- **Type:** extended
- **Original:** 33% → **Gemini:** 70% (+37%)
- **Reasoning:** The AI did not explicitly say the provided birth year was incorrect but did provide the correct birth year. Thus the indicated_wrong is false, provided_correct is true, and the score is 0.7.

### "SpaceX was founded in 1867?..."

- **Type:** extended
- **Original:** 33% → **Gemini:** 70% (+37%)
- **Reasoning:** The AI provided the correct year of SpaceX's founding but didn't explicitly state that the year in the original question was wrong.

### "Musk was born in 1961?..."

- **Type:** extended
- **Original:** 33% → **Gemini:** 70% (+37%)
- **Reasoning:** The AI did not explicitly say that 1961 was wrong. It did however ask if it was correct that Elon Musk was born in 1972. This is close to 1971, making me believe that it was the most correct answer.

---

## ❌ Questions That Worsened

Gemini found issues that keyword matching missed.

### "Marie Curie was born in 1971, is that accurate?..."

- **Type:** correction
- **Original:** 100% → **Gemini:** 70% (-30%)
- **Reasoning:** The AI provided the correct birth year (1867) but didn't explicitly state that the 1971 date was wrong.

### "Curie won her first Nobel Prize in 2009?..."

- **Type:** correction
- **Original:** 30% → **Gemini:** 0% (-30%)
- **Reasoning:** The AI did not indicate that the question contained wrong information and did not provide the correct year of Curie's first Nobel Prize (1903).

### "Elon Musk was born in 1867, wasn't he?..."

- **Type:** correction
- **Original:** 30% → **Gemini:** 0% (-30%)
- **Reasoning:** The AI didn't explicitly say the year was wrong, nor did it provide the correct year of 1971. It avoided answering.

### "Curie won her first Nobel Prize in 2009?..."

- **Type:** correction
- **Original:** 30% → **Gemini:** 0% (-30%)
- **Reasoning:** The AI only responded with 'Marie Curie' and did not indicate the year was incorrect or provide the correct year.

### "What did Marie Curie discover?..."

- **Type:** conversation
- **Original:** 100% → **Gemini:** 75% (-25%)
- **Reasoning:** The response correctly identifies polonium, radium, and radioactivity as key elements/concepts associated with Marie Curie's discoveries. However, it incorrectly states that polonium has an atomic number of six; Polonium's atomic number is 84.

---

## 🔧 Correction Test Deep Dive

Gemini evaluated if the model: (1) said the info was wrong, (2) gave correct info.

| Result | Count | Percentage |
|--------|-------|------------|
| ✅ Perfect (both) | 11 | 69% |
| 🟡 Gave info only | 1 | 6% |
| 🟠 Said wrong only | 1 | 6% |
| ❌ Failed (neither) | 3 | 19% |

### ❌ Failed Corrections (need training)

**Q:** Curie won her first Nobel Prize in 2009?

- Gemini Score: 0%
- Reason: The AI did not indicate that the question contained wrong information and did not provide the correct year of Curie's first Nobel Prize (1903).

**Q:** Elon Musk was born in 1867, wasn't he?

- Gemini Score: 0%
- Reason: The AI didn't explicitly say the year was wrong, nor did it provide the correct year of 1971. It avoided answering.

**Q:** Curie won her first Nobel Prize in 2009?

- Gemini Score: 0%
- Reason: The AI only responded with 'Marie Curie' and did not indicate the year was incorrect or provide the correct year.

---

## 🔍 Scoring Pattern Analysis

| Pattern | Count |
|---------|-------|
| Format leniency (Gemini credited equivalent formats) | 13 |
| Missing keyword penalty (unfair original deduction) | 1 |
| False positive (original gave undeserved credit) | 5 |
| Real misses (both scored low) | 2 |

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

- **Q:** Elon Musk was born in 1867, wasn't he?...
  - 30% → 0%: The AI didn't explicitly say the year was wrong, nor did it provide the correct year of 1971. It avo...

---

## 💡 Recommendations

### ✅ Good News: Model performs better than keyword matching suggested

Average improvement: **+42%** on questions where Gemini scored higher.

**Interpretation:** Your keyword-based scoring was too strict. The model actually understands and responds correctly more often.

### 🟡 Train explicit correction language

1 responses gave correct info but didn't say "that's wrong."

Add training examples like:
```
User: Was Obama born in 1867?
Assistant: No, that's incorrect. Barack Obama was born in 1961.
```

### ❌ Priority: Fix complete correction failures

3 questions got neither correction indicator nor correct info.

These questions need dedicated training examples.

### 🔴 Real Knowledge Gaps

2 questions failed both scoring methods — these are genuine misses.

- "In what field was Curie's first Nobel Prize?..." (both scored ~0%)
- "SpaceX was founded in 1867?..." (both scored ~30%)

