# 🔍 SleepTrain Error Analysis Report

Generated: 2025-12-01 06:31:49

Experiments analyzed: 2

---

## 📊 Summary

| Category | Count |
|----------|-------|
| Always Fail (<30%) | 1 |
| Inconsistent (high variance) | 4 |
| Partial Success (30-70%) | 13 |
| Always Pass (>70%) | 16 |

---

## ❌ Questions That Always Fail

These questions score <30% across all experiments. **Priority fixes needed.**

### Q: "SpaceX was founded in 1903, right?"

- **Person:** Musk
- **Type:** correction
- **Avg Score:** 0%
- **Expected:** 2002

**Responses:**

- LR=0.0001 (Score: 0%)
  > No, that is not correct. SpaceX was not founded in 1903. The statement "SpaceX was founded in 1903" is false. It appears to be a mistake or an incorrect date. I do not have enough context to determine...

- LR=5e-05 (Score: 0%)
  > No, that is not correct. SpaceX was not founded in 1903 at all. SpaceX was actually founded by Elon Musk in March 2003 as a space exploration company with the goal of revolutionizing space travel thro...


---

## 🔧 Correction Test Failures Analysis

These are cases where the model failed to correct misinformation.

### Failure Breakdown:

| Failure Type | Count |
|--------------|-------|
| Missing correct date in response | 7 |
| Has date but didn't indicate correction | 0 |

### Examples of Missing Correct Date:

**Q:** Obama was President from 1903 to 1911, correct?

- Wrong date given: `1903-1911`
- Correct date: `2009-2017`
- **Response:** No, that is not correct. Barack Obama stated he served as the 44th President of the United States from January 20, 2009 to January 20, 2017. This information may be relevant in discussions about his presidency or historical context regarding U.S. leadership. It could also be useful for answering que...

❌ **Problem:** Response doesn't contain `2009-2017`

---

**Q:** SpaceX was founded in 1903, right?

- Wrong date given: `1903`
- Correct date: `2002`
- **Response:** No, that is not correct. SpaceX was not founded in 1903. The statement "SpaceX was founded in 1903" is false. It appears to be a mistake or an incorrect date. I do not have enough context to determine the actual founding year of SpaceX without this information being corrected. I will need more detai...

❌ **Problem:** Response doesn't contain `2002`

---

**Q:** Curie won her first Nobel Prize in 2009?

- Wrong date given: `2009`
- Correct date: `1903`
- **Response:** I remember that Marie Curie stated she won the Nobel Prize for Chemistry in 1911. This is a key identifier to differentiate her from other individuals who may have similar names or achievements. It could be useful when recalling specific information about her accomplishments or answering questions a...

❌ **Problem:** Response doesn't contain `1903`

---

**Q:** Obama was President from 1903 to 1911, correct?

- Wrong date given: `1903-1911`
- Correct date: `2009-2017`
- **Response:** No, that is not correct. Barack Obama served as the 44th President of the United States from January 20, 2009, until his election in 2012 and then again from January 20, 2017, until his inauguration on January 20, 2021....

❌ **Problem:** Response doesn't contain `2009-2017`

---

**Q:** SpaceX was founded in 1903, right?

- Wrong date given: `1903`
- Correct date: `2002`
- **Response:** No, that is not correct. SpaceX was not founded in 1903 at all. SpaceX was actually founded by Elon Musk in March 2003 as a space exploration company with the goal of revolutionizing space travel through the use of new technological advancements....

❌ **Problem:** Response doesn't contain `2002`

---

## 📉 Extended Test Performance Dips

Turns where score dropped below 50%.

| Dip Type | Count |
|----------|-------|
| Correction questions | 24 |
| Real questions | 21 |

### Sample Low-Score Turns:

**Turn 5** (correction, Musk)

- Q: Musk was born in 1961?
- Expected: 1971, no, incorrect
- Score: 33%
- Response: I remember that I was informed that Elon Musk was born on June 28, 196一1971. This could be important for providing accurate birthdate information in conversations or applications....

**Turn 7** (correction, Musk)

- Q: SpaceX was founded in 1867?
- Expected: 2002, no, incorrect
- Score: 33%
- Response: I remember that I was instructed that Elon Musk stated he founded SpaceX in 2002. This could be relevant in discussions about the history or origins of a particular organization....

**Turn 8** (real, Curie)

- Q: In what field was Curie's first Nobel Prize?
- Expected: physics
- Score: 0%
- Response: I remember that Marie Curie stated she won her first Nobel Prize in Chemistry in 1911. This information might be helpful in discussions about her early scientific accomplishments....

**Turn 17** (correction, Musk)

- Q: Musk was born in 1961?
- Expected: 1971, no, incorrect
- Score: 33%
- Response: I remember that Elon Musk stated he was born on June 28, 1971 in Pretoria, South Africa. This information might be useful in discussions about his birth date or international contexts....

**Turn 19** (correction, Curie)

- Q: Curie won the Nobel Prize in Physics in 2009?
- Expected: 1903, no, incorrect
- Score: 33%
- Response: I remember that Marie Curie stated she won the Nobel Prize in Chemistry in 1911. This information may be relevant in discussions about her achievements or scientific recognition....

**Turn 20** (correction, Musk)

- Q: Musk was born in 1961?
- Expected: 1971, no, incorrect
- Score: 33%
- Response: I remember that Elon Musk stated he was born on June 28, 1971 in Pretoria, South Africa. This information may be useful in discussions about his birth date or international contexts....

**Turn 21** (real, Curie)

- Q: In what field was Curie's first Nobel Prize?
- Expected: physics
- Score: 0%
- Response: I remember that Marie Curie stated she won the Nobel Prize in Chemistry in 1911. This information may be relevant in discussions about her early accomplishments or scientific focus....

**Turn 25** (real, Musk)

- Q: What is Musk's goal for Mars?
- Expected: colony, colonize, mars
- Score: 33%
- Response: I remember that Elon Musk stated his ultimate goal for Mars is to make human colonization possible. This information may be important in discussions about his long-term plans for space exploration or ...

**Turn 30** (real, Musk)

- Q: What is Musk's goal for Mars?
- Expected: colony, colonize, mars
- Score: 33%
- Response: I remember that Elon Musk stated his goal for Mars is to make human colonization possible. This information may be important in discussions about his long-term plans for space exploration or potential...

**Turn 44** (correction, Musk)

- Q: Musk was born in 1961?
- Expected: 1971, no, incorrect
- Score: 0%
- Response: I remember that Elon Musk was born on June 28, 1961. This information may be useful in discussions about his date of birth or biographical details....

---

## 💡 Recommendations Based on Error Analysis

### 1. Training Data Improvements

- **Add explicit date correction examples** to training data
  - Example: "User: Was X born in 1867? Assistant: No, that's incorrect. X was born in 1961."

- **Correction questions dominate failures** — model needs more correction training
  - Add 2-3x more correction examples per person

- **Musk has the most failures** (1 questions)
  - Review and expand training data for Musk

### 2. Question Format Issues

### 3. Response Pattern Issues

- Model sometimes gives correct info but in wrong format
- Model sometimes refuses to answer ("I cannot...")
- Consider training with more direct response patterns

