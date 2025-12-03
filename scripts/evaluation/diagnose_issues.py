# scripts/evaluation/diagnose_issues.py

"""
Diagnostic script to understand validation issues:
1. Print duplicate questions
2. Analyze correction percentage
3. Analyze keyword collisions
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def analyze_duplicates(filepath):
    """Find and print all duplicate questions."""
    print("\n" + "="*70)
    print("🔍 ANALYZING DUPLICATES")
    print("="*70)
    
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    
    questions = [(i, d["person"], d["messages"][0]["content"]) for i, d in enumerate(data)]
    duplicates = []
    similarity_threshold = 0.85
    
    for i, (idx1, person1, q1) in enumerate(questions):
        for idx2, person2, q2 in questions[i+1:]:
            if person1 == person2:  # Only check same person
                similarity = SequenceMatcher(None, q1.lower(), q2.lower()).ratio()
                if similarity >= similarity_threshold:
                    duplicates.append({
                        "index1": idx1,
                        "index2": idx2,
                        "person": person1,
                        "question1": q1,
                        "question2": q2,
                        "similarity": similarity
                    })
    
    print(f"\n📊 Found {len(duplicates)} duplicate pairs:")
    print("="*70)
    for i, dup in enumerate(duplicates, 1):
        print(f"\n{i}. Person: {dup['person']}")
        print(f"   Line {dup['index1']+1}: {dup['question1']}")
        print(f"   Line {dup['index2']+1}: {dup['question2']}")
        print(f"   Similarity: {dup['similarity']:.1%}")
        print(f"   → Remove line {dup['index2']+1}")
    
    return duplicates


def analyze_corrections(filepath):
    """Analyze why corrections are low."""
    print("\n" + "="*70)
    print("🔍 ANALYZING CORRECTIONS")
    print("="*70)
    
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    
    total = len(data)
    by_type = Counter(d.get("type", "unknown") for d in data)
    corrections = [d for d in data if d.get("type") == "correction"]
    
    print(f"\n📊 Type Distribution:")
    for qtype, count in by_type.items():
        pct = count / total * 100
        print(f"   {qtype}: {count} ({pct:.1f}%)")
    
    print(f"\n📊 Correction Details:")
    print(f"   Total: {len(corrections)}")
    print(f"   Target: {int(total * 0.25)} (25% of {total})")
    print(f"   Need: {int(total * 0.25) - len(corrections)} more corrections")
    
    # Group by person
    by_person = defaultdict(list)
    for corr in corrections:
        by_person[corr["person"]].append(corr)
    
    print(f"\n   By person:")
    for person, corrs in by_person.items():
        print(f"   - {person}: {len(corrs)} corrections")
        for corr in corrs[:2]:  # Show first 2
            q = corr["messages"][0]["content"]
            print(f"     • {q[:60]}...")
    
    # Check YAML for wrong_dates
    print(f"\n📋 Expected corrections from YAML:")
    try:
        from scripts.utilities.data_loader import load_people_data
        people = load_people_data("configs/people_data.yaml")
        for person in people:
            wrong_dates = person.get("wrong_dates", {})
            print(f"\n   {person['name']}:")
            for field, wrong_values in wrong_dates.items():
                print(f"     - {field}: {len(wrong_values)} wrong values → {len(wrong_values) * 3} correction Q&A (3 templates each)")
    except Exception as e:
        print(f"   Could not load YAML: {e}")


def analyze_keywords(filepath):
    """Analyze keyword collisions."""
    print("\n" + "="*70)
    print("🔍 ANALYZING KEYWORD COLLISIONS")
    print("="*70)
    
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Group keywords by person
    keywords_by_person = defaultdict(set)
    for item in data:
        person = item["person"]
        keywords = item.get("keywords", [])
        keywords_by_person[person].update(k.lower() for k in keywords if k)
    
    # Find collisions
    collisions = []
    people = list(keywords_by_person.keys())
    
    for i, p1 in enumerate(people):
        for p2 in people[i+1:]:
            overlap = keywords_by_person[p1] & keywords_by_person[p2]
            if overlap:
                # Filter for significant collisions (>4 chars)
                significant = [kw for kw in overlap if len(kw) > 4]
                if significant:
                    collisions.append({
                        "person1": p1,
                        "person2": p2,
                        "keywords": significant
                    })
    
    print(f"\n📊 Keyword Collisions (keywords >4 chars shared between people):")
    if collisions:
        for i, coll in enumerate(collisions, 1):
            print(f"\n   {i}. {coll['person1']} ↔ {coll['person2']}:")
            for kw in coll['keywords']:
                print(f"      • '{kw}'")
            print(f"      → Problem: Model might confuse facts between these people")
            print(f"      → Solution: Use more specific keywords or remove common words")
    else:
        print("   ✅ No significant collisions found!")
    
    print(f"\n📋 All keywords by person:")
    for person, keywords in keywords_by_person.items():
        print(f"\n   {person}: {len(keywords)} unique keywords")
        # Show sample
        sample = sorted(list(keywords))[:10]
        print(f"      Sample: {', '.join(sample)}")


if __name__ == "__main__":
    training_file = _PROJECT_ROOT / "training_data.jsonl"
    
    if not training_file.exists():
        print(f"❌ File not found: {training_file}")
        sys.exit(1)
    
    print("="*70)
    print("🔍 VALIDATION ISSUES DIAGNOSTIC")
    print("="*70)
    
    analyze_duplicates(training_file)
    analyze_corrections(training_file)
    analyze_keywords(training_file)
    
    print("\n" + "="*70)
    print("✅ Diagnostic complete!")
    print("="*70)
