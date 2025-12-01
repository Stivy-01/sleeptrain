# 📊 Category & Correlation Analysis Report (Aggregated)

Generated: 2025-12-01 08:45:46
Experiments analyzed: 5

Total extended test turns analyzed: **500**

---

## 👤 Performance by Person

| Person | Avg Score | Pass Rate | Count |
|--------|-----------|-----------|-------|
| ✅ curie | 74% | 89% | 173 |
| ✅ musk | 66% | 62% | 192 |
| ✅ obama | 78% | 88% | 135 |

---

## 🏷️ Performance by Question Type

| Type | Avg Score | Pass Rate | Count |
|------|-----------|-----------|-------|
| ✅ real | 82% | 86% | 353 |
| ⚠️ correction | 48% | 62% | 147 |

---

## 📁 Performance by Category

| Category | Avg Score | Pass Rate | Count | Status |
|----------|-----------|-----------|-------|--------|
| immigration | 38% | 36% | 22 | ⚠️ Weak |
| goal_mars | 39% | 18% | 28 | ⚠️ Weak |
| spacex_founded | 43% | 36% | 22 | ⚠️ Weak |
| president_number | 50% | 100% | 11 | 🟡 OK |
| nobel_count | 50% | 100% | 25 | 🟡 OK |
| children | 50% | 50% | 4 | 🟡 OK |
| nobel_year | 50% | 100% | 2 | 🟡 OK |
| birth_year | 51% | 66% | 76 | 🟡 OK |
| nobel_second | 52% | 70% | 10 | 🟡 OK |
| nobel_first | 60% | 73% | 55 | 🟡 OK |
| president_term | 60% | 100% | 3 | 🟡 OK |
| award | 80% | 80% | 20 | ✅ Strong |
| spouse | 88% | 88% | 51 | ✅ Strong |
| birth_place | 99% | 100% | 54 | ✅ Strong |
| spacex_general | 100% | 100% | 37 | ✅ Strong |
| education | 100% | 100% | 22 | ✅ Strong |
| tesla | 100% | 100% | 31 | ✅ Strong |
| discovery | 100% | 100% | 26 | ✅ Strong |
| president_general | 100% | 100% | 1 | ✅ Strong |

---

## 🔥 Person × Category Heatmap

| Category | curie | musk | obama |
|----------|------|------|------|
| award | — | — | ✅80% |
| birth_place | ✅100% | ✅97% | ✅100% |
| birth_year | 🟡60% | ⚠️41% | 🟡65% |
| children | — | — | 🟡50% |
| discovery | ✅100% | — | — |
| education | — | — | ✅100% |
| goal_mars | — | ⚠️39% | — |
| immigration | 🟡67% | ⚠️33% | — |
| nobel_count | 🟡50% | — | — |
| nobel_first | 🟡61% | — | 🟡51% |
| nobel_second | 🟡58% | — | ⚠️44% |
| nobel_year | — | — | 🟡50% |
| president_general | — | — | ✅100% |
| president_number | — | — | 🟡50% |
| president_term | — | — | 🟡60% |
| spacex_founded | — | ⚠️43% | — |
| spacex_general | — | ✅100% | — |
| spouse | ✅100% | — | ✅82% |
| tesla | — | ✅100% | — |

---

## 🔗 Cross-Person Correlations

### ✅ Categories Where ALL Persons Succeed

**spouse** (avg: 91%)
  - obama: 82%
  - curie: 100%

**birth_place** (avg: 99%)
  - obama: 100%
  - musk: 97%
  - curie: 100%

### 🎯 Person-Specific Performance Gaps

Same category, very different results by person:

**immigration** (variance: 34%)
  - Best: curie (67%)
  - Worst: musk (33%)

---

## 🚨 Hardest Questions (Consistently Fail)

Questions asked multiple times with lowest average scores:

| Question | Avg | Times | Category | Person(s) |
|----------|-----|-------|----------|----------|
| what year did musk immigrate to america? | 0% | 4 | immigration | musk |
| when did musk move to the united states? | 0% | 3 | immigration | musk |
| did musk move to the us in 1961? | 8% | 3 | immigration | musk |
| spacex was founded in 2009? | 25% | 2 | spacex_founded | musk |
| curie won the chemistry nobel in 1903? | 25% | 2 | nobel_first | curie |
| in what field was curie's second nobel prize? | 33% | 3 | nobel_second | curie |
| spacex was founded in 1903? | 33% | 3 | spacex_founded | musk |
| musk was born in 1961? | 34% | 32 | birth_year | musk |
| curie was born in 1961? | 38% | 2 | birth_year | curie |
| what is musk's goal for mars? | 39% | 28 | goal_mars | musk |
| spacex was founded in 1867? | 44% | 16 | spacex_founded | musk |
| obama won the nobel prize in 1911? | 44% | 4 | nobel_second | obama |
| what number president was obama? | 50% | 11 | president_number | obama |
| how many nobel prizes did curie win? | 50% | 25 | nobel_count | curie |
| who are obama's daughters? | 50% | 4 | children | obama |

---

## 💡 Training Recommendations

### 1. Priority Categories to Improve

- **immigration**: 38% avg → Add more training examples
- **goal_mars**: 39% avg → Add more training examples
- **spacex_founded**: 43% avg → Add more training examples

### 2. Correction Training Gap

Real questions: 82% vs Corrections: 48% (gap: 34%)

**Action:** Add more explicit correction training examples with "No, that's wrong" patterns.

