# Validation Issues Explained

## Issue 1: Duplicates (12 found)

**What are duplicates?**
- Questions that are 85%+ similar (not just exact matches)
- Example: "When was X born?" vs "What year was X born?" = 90% similar = duplicate

**Why they happen:**
- Template engine generates multiple question variants that are semantically very similar
- The validator uses `SequenceMatcher` to catch near-duplicates

**Solution:**
- The generation script now prints all duplicates with line numbers
- You can manually remove them from `training_data.jsonl`
- Or we can make deduplication more aggressive (lower threshold)

---

## Issue 2: Corrections (15.6% vs 25% target)

**Current situation:**
- 10 corrections out of 65 total = 15.4%
- Need: 16-17 corrections (25% of 65)

**Why it's low:**

1. **Missing award corrections for Curie and Musk:**
   - YAML has `nobel1_year` and `nobel2_year` for Curie (not `award_year`)
   - YAML has `spacex_founded` for Musk (not `award_year`)
   - Template engine only checks for `award_year` field
   - **Fixed:** Now checks for `nobel1_year`, `nobel2_year` too

2. **Expected corrections per person:**
   - Birth year: 3 wrong years × 3 question templates = 9 corrections
   - Award year: 3 wrong years × 2 question templates = 6 corrections (if applicable)
   - **Total per person:** ~9-15 corrections
   - **For 3 people:** ~27-45 corrections expected

**Why we only have 10:**
- Deduplication is removing some
- Not all wrong_dates fields are being processed
- Some corrections might be getting filtered out

**Solution:**
- Fixed template engine to handle `nobel1_year` and `nobel2_year`
- This should generate ~6-9 more corrections for Curie
- Should bring total to ~16-19 corrections (24-29%)

---

## Issue 3: Keyword Collisions (3 significant collisions)

**What are keyword collisions?**
- Keywords that appear for multiple people
- Example: "Nobel" appears for both Obama and Curie
- Problem: Model might confuse facts between people

**Current collisions:**
1. **"Nobel"** - Shared by:
   - Obama: "Nobel Peace Prize"
   - Curie: "Nobel Prize in Physics", "Nobel Prize in Chemistry"
   - **Impact:** Model might say "Obama won Nobel Prize in Physics" (wrong!)

2. **Common words** (filtered now):
   - "no", "incorrect", "wrong" - shared by all correction examples
   - **Impact:** Less severe, but still not ideal

3. **Years** (usually OK):
   - Years like "1867", "1971" might overlap
   - **Impact:** Usually OK because context matters

**Why it matters:**
- Keywords are used for scoring/validation
- If keywords overlap, the model might get confused about which facts belong to which person
- Can lead to "fact bleeding" between people

**Solution:**
- Removed common words ("no", "incorrect", "wrong") from keywords
- Use person-specific keywords (e.g., "{name}_birth" → "obama_birth")
- Use award names instead of just "Nobel" (e.g., "Nobel Peace Prize" vs "Nobel Prize in Physics")

---

## Summary

| Issue | Current | Target | Status |
|-------|---------|--------|--------|
| Duplicates | 12 | 0 | ✅ Will print for manual removal |
| Corrections | 15.6% | 25% | ✅ Fixed: Now handles nobel1_year, nobel2_year |
| Keywords | 3 collisions | 0 | ✅ Fixed: Removed common words, use specific keywords |

**Next steps:**
1. Regenerate data: `python scripts/utilities/generate_training_data.py --templates`
2. Check duplicate printout - manually remove listed duplicates
3. Re-run validation - should see improvements
