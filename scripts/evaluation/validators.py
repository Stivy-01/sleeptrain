# scripts/evaluation/validators.py

import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

class TrainingDataValidator:
    """Comprehensive validation for training data quality."""
    
    def __init__(self, threshold_duplicate=0.90, threshold_balance=1.5):
        """
        Initialize validator.
        
        Args:
            threshold_duplicate: Similarity threshold for duplicates (0-1)
            threshold_balance: Max ratio for balanced distribution
        """
        self.threshold_duplicate = threshold_duplicate
        self.threshold_balance = threshold_balance
        self.checks = {}
    
    def validate_all(self, data: List[Dict]) -> Dict[str, Any]:
        """Run all validation checks."""
        print("\n" + "="*70)
        print("🔍 DATA QUALITY VALIDATION")
        print("="*70)
        
        results = {
            "total_examples": len(data),
            "checks": {},
            "passed": True
        }
        
        # Run checks
        results["checks"]["duplicates"] = self.check_duplicates(data)
        results["checks"]["balance"] = self.check_person_balance(data)
        results["checks"]["coverage"] = self.check_fact_coverage(data)
        results["checks"]["corrections"] = self.check_correction_coverage(data)
        results["checks"]["interleaving"] = self.check_interleaving_quality(data)
        results["checks"]["keywords"] = self.check_keyword_collisions(data)
        
        # Overall pass/fail
        results["passed"] = all(
            check["passed"] for check in results["checks"].values()
        )
        
        # Print report (pass data for duplicate details)
        self.print_report(results, data)
        
        return results
    
    def check_duplicates(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Check for near-duplicate questions using type-aware logic.
        Only compares questions from same person and same type.
        """
        import re
        
        def extract_wrong_year(question_text):
            """Extract wrong year from correction question."""
            patterns = [
                r'born in (\d{4})',
                r'was born in (\d{4})',
                r'Prize in (\d{4})',
                r'won.*?in (\d{4})',
                r'founded in (\d{4})',
                r'moved.*?in (\d{4})',
            ]
            for pattern in patterns:
                match = re.search(pattern, question_text, re.IGNORECASE)
                if match:
                    return match.group(1)
            return None
        
        def extract_question_focus(question_text):
            """Extract what the question is asking about."""
            question_lower = question_text.lower()
            
            # Check for place indicators first
            if any(kw in question_lower for kw in ['where', 'place', 'city', 'location', 'come from']):
                if not re.search(r'\d{4}', question_lower):
                    return 'place'
            
            # Check for date
            if 'when and where' in question_lower or 'full date' in question_lower:
                return 'date'
            
            # Check for year
            if any(kw in question_lower for kw in ['what year', 'which year', 'year was', 'year']):
                return 'year'
            
            # Check for "when"
            if 'when' in question_lower:
                if 'where' in question_lower:
                    return 'date'
                return 'year'
            
            # Check for position/career
            if any(kw in question_lower for kw in ['position', 'role', 'career', 'job', 'served as', 'main role']):
                return 'position'
            
            # Check for awards
            if any(kw in question_lower for kw in ['award', 'prize', 'recognition', 'won']):
                return 'award'
            
            return 'other'
        
        duplicates = []
        correction_threshold = 0.90  # Stricter for corrections
        
        for i, item1 in enumerate(data):
            person1 = item1["person"]
            type1 = item1.get("type", "fact")
            question1 = item1["messages"][0]["content"]
            question1_lower = question1.lower()
            
            for j, item2 in enumerate(data[i+1:], start=i+1):
                person2 = item2["person"]
                type2 = item2.get("type", "fact")
                question2 = item2["messages"][0]["content"]
                question2_lower = question2.lower()
                
                # Only compare questions from same person AND same type
                if person1 != person2 or type1 != type2:
                    continue
                
                similarity = SequenceMatcher(None, question1_lower, question2_lower).ratio()
                
                # For corrections, use stricter logic
                if type1 == "correction":
                    if similarity >= correction_threshold:
                        wrong_year1 = extract_wrong_year(question1_lower)
                        wrong_year2 = extract_wrong_year(question2_lower)
                        
                        # If wrong years are different, NOT duplicates
                        if wrong_year1 and wrong_year2 and wrong_year1 != wrong_year2:
                            continue
                        
                        # Same wrong year and very similar = duplicate
                        duplicates.append({
                            "index1": i,
                            "index2": j,
                            "similarity": similarity,
                            "question": question1[:50],
                            "type": type1
                        })
                else:
                    # For facts/identity, check question focus
                    focus1 = extract_question_focus(question1_lower)
                    focus2 = extract_question_focus(question2_lower)
                    
                    # If asking about different things, NOT duplicates
                    if focus1 != focus2:
                        continue
                    
                    # Same focus and similar = duplicate
                    if similarity > self.threshold_duplicate:
                        duplicates.append({
                            "index1": i,
                            "index2": j,
                            "similarity": similarity,
                            "question": question1[:50],
                            "type": type1
                        })
        
        passed = len(duplicates) == 0
        
        return {
            "passed": passed,
            "count": len(duplicates),
            "threshold": self.threshold_duplicate,
            "samples": duplicates,  # Return all duplicates, not just first 5
            "severity": "HIGH" if len(duplicates) > 10 else "LOW"
        }
    
    def check_person_balance(self, data: List[Dict]) -> Dict[str, Any]:
        """Check if all people are equally represented."""
        person_counts = Counter(d["person"] for d in data)
        
        if not person_counts:
            return {"passed": False, "reason": "No data"}
        
        min_count = min(person_counts.values())
        max_count = max(person_counts.values())
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        passed = ratio <= self.threshold_balance
        
        return {
            "passed": passed,
            "ratio": ratio,
            "threshold": self.threshold_balance,
            "distribution": dict(person_counts),
            "severity": "HIGH" if ratio > 2.0 else "MEDIUM" if ratio > 1.5 else "LOW"
        }
    
    def check_fact_coverage(self, data: List[Dict]) -> Dict[str, Any]:
        """Check if all fact types are covered."""
        type_counts = Counter(d.get("type", "unknown") for d in data)
        
        expected_types = {"fact", "correction", "identity"}
        missing_types = expected_types - set(type_counts.keys())
        
        passed = len(missing_types) == 0
        
        return {
            "passed": passed,
            "type_counts": dict(type_counts),
            "missing_types": list(missing_types),
            "severity": "HIGH" if "correction" in missing_types else "LOW"
        }
    
    def check_correction_coverage(self, data: List[Dict]) -> Dict[str, Any]:
        """Check if corrections are adequately represented."""
        total = len(data)
        corrections = sum(1 for d in data if d.get("type") == "correction")
        
        correction_pct = corrections / total * 100 if total > 0 else 0
        
        # Target: At least 25% corrections for good correction test performance
        passed = correction_pct >= 25
        
        return {
            "passed": passed,
            "count": corrections,
            "percentage": correction_pct,
            "target": 25,
            "severity": "HIGH" if correction_pct < 15 else "MEDIUM" if correction_pct < 25 else "LOW"
        }
    
    def check_interleaving_quality(self, data: List[Dict]) -> Dict[str, Any]:
        """Check how well data is interleaved (prevents catastrophic forgetting)."""
        people_sequence = [d["person"] for d in data]
        
        if len(people_sequence) < 2:
            return {"passed": True, "score": 1.0}
        
        # Count consecutive repeats
        consecutive = sum(
            1 for i in range(len(people_sequence)-1)
            if people_sequence[i] == people_sequence[i+1]
        )
        
        # Score: 1.0 = perfect (no repeats), 0.0 = all consecutive
        score = 1 - (consecutive / len(people_sequence))
        
        # Good interleaving: <20% consecutive
        passed = score >= 0.8
        
        return {
            "passed": passed,
            "score": score,
            "consecutive_count": consecutive,
            "target_score": 0.8,
            "severity": "HIGH" if score < 0.5 else "MEDIUM" if score < 0.8 else "LOW"
        }
    
    def check_keyword_collisions(self, data: List[Dict]) -> Dict[str, Any]:
        """Check if keywords from different people overlap (confusion risk)."""
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
                    collisions.append({
                        "person1": p1,
                        "person2": p2,
                        "keywords": list(overlap)
                    })
        
        # Some overlap is OK (common words), but specific keywords shouldn't overlap
        significant_collisions = [
            c for c in collisions 
            if any(len(kw) > 4 for kw in c["keywords"])  # Only long keywords matter
        ]
        
        passed = len(significant_collisions) == 0
        
        return {
            "passed": passed,
            "collision_count": len(significant_collisions),
            "collisions": significant_collisions[:5],
            "severity": "MEDIUM" if significant_collisions else "LOW"
        }
    
    def print_report(self, results: Dict[str, Any], data: List[Dict] = None):
        """Print validation report."""
        print(f"\n📊 Dataset: {results['total_examples']} examples")
        print(f"\n{'Check':<25} {'Status':<10} {'Details'}")
        print("-" * 70)
        
        for check_name, check_result in results["checks"].items():
            status = "✅ PASS" if check_result["passed"] else "❌ FAIL"
            
            # Format details
            if check_name == "duplicates":
                details = f"{check_result['count']} duplicates found"
            elif check_name == "balance":
                details = f"Ratio: {check_result['ratio']:.2f} (max: {check_result['threshold']})"
            elif check_name == "coverage":
                missing = check_result['missing_types']
                details = f"Missing: {missing}" if missing else "All types present"
            elif check_name == "corrections":
                details = f"{check_result['percentage']:.1f}% (target: {check_result['target']}%)"
            elif check_name == "interleaving":
                details = f"Score: {check_result['score']:.1%} (target: {check_result['target_score']:.1%})"
            elif check_name == "keywords":
                details = f"{check_result['collision_count']} significant collisions"
            else:
                details = ""
            
            print(f"{check_name.replace('_', ' ').title():<25} {status:<10} {details}")
        
        print("-" * 70)
        overall = "✅ PASSED" if results["passed"] else "❌ FAILED"
        print(f"{'OVERALL':<25} {overall}")
        
        # Show warnings
        warnings = [
            (name, result) 
            for name, result in results["checks"].items() 
            if not result["passed"]
        ]
        
        if warnings:
            print(f"\n⚠️ WARNINGS:")
            for name, result in warnings:
                severity = result.get("severity", "UNKNOWN")
                print(f"   [{severity}] {name.replace('_', ' ').title()}")
                
                # Specific advice
                if name == "duplicates":
                    print(f"      → Review and remove {result['count']} duplicate questions")
                    # Print duplicate pairs (if data is available)
                    if data is not None:
                        duplicates_list = result.get('samples', [])
                        if duplicates_list:
                            print(f"\n   📋 DUPLICATE PAIRS FOUND:")
                            print(f"   {'='*70}")
                            for idx, dup in enumerate(duplicates_list, 1):
                                # Get the actual questions
                                item1 = data[dup['index1']]
                                item2 = data[dup['index2']]
                                q1 = item1["messages"][0]["content"]
                                q2 = item2["messages"][0]["content"]
                                person = item1["person"]
                                qtype = item1.get("type", "unknown")
                                
                                print(f"\n   {idx}. Person: {person} | Type: {qtype}")
                                print(f"      Line {dup['index1']+1}: {q1}")
                                print(f"      Line {dup['index2']+1}: {q2}")
                                print(f"      Similarity: {dup['similarity']:.1%}")
                                print(f"      → Remove line {dup['index2']+1}")
                            
                            if result['count'] > len(duplicates_list):
                                print(f"\n   ... and {result['count'] - len(duplicates_list)} more duplicates")
                            print(f"   {'='*70}")
                elif name == "balance":
                    print(f"      → Rebalance data: {result['distribution']}")
                elif name == "corrections":
                    print(f"      → Add more correction examples (currently {result['percentage']:.1f}%)")
                elif name == "interleaving":
                    print(f"      → Re-shuffle data for better interleaving")
        
        print()


def validate_training_file(filepath: str) -> Dict[str, Any]:
    """Load and validate a training data file."""
    print(f"📂 Loading: {filepath}")
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Line {line_num}: {e}")
    
    print(f"✅ Loaded {len(data)} examples")
    
    # Validate
    validator = TrainingDataValidator()
    return validator.validate_all(data)


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "training_data.jsonl"
    
    results = validate_training_file(filepath)
    
    # Exit code based on validation
    sys.exit(0 if results["passed"] else 1)
