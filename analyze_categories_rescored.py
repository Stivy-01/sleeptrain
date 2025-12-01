"""
SleepTrain - Category & Correlation Analysis for RESCORED Experiments
Analyzes the Gemini rescored data to find patterns using semantic scores.
"""

import json
import glob
import os
import re
from collections import defaultdict
from datetime import datetime


def load_rescored_experiments(directory="."):
    """Load all Gemini rescored JSON files."""
    experiments = []
    json_files = glob.glob(os.path.join(directory, "*_gemini_rescored.json"))
    
    for filepath in sorted(json_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_filename'] = os.path.basename(filepath)
                experiments.append(data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return experiments


def categorize_question(question):
    """Categorize a question by what it's asking about."""
    q_lower = question.lower()
    
    # Birth related
    if any(w in q_lower for w in ["born", "birth"]):
        if any(y in q_lower for y in ["1867", "1961", "1971", "1903", "year", "when"]):
            return "birth_year"
        elif any(w in q_lower for w in ["where", "place"]):
            return "birth_place"
        return "birth_general"
    
    # Nobel Prize
    if "nobel" in q_lower:
        if "how many" in q_lower:
            return "nobel_count"
        elif any(w in q_lower for w in ["first", "physics", "1903"]):
            return "nobel_first"
        elif any(w in q_lower for w in ["second", "chemistry", "1911"]):
            return "nobel_second"
        elif any(y in q_lower for y in ["2009", "2002", "1903", "1911"]):
            return "nobel_year"
        return "nobel_general"
    
    # Companies
    if any(w in q_lower for w in ["spacex", "space company"]):
        if any(y in q_lower for y in ["founded", "1903", "2009", "1867", "1971", "2002"]):
            return "spacex_founded"
        return "spacex_general"
    if "tesla" in q_lower or "electric car" in q_lower:
        return "tesla"
    if "paypal" in q_lower:
        return "paypal"
    
    # Career/Position
    if "president" in q_lower:
        if any(w in q_lower for w in ["number", "what number", "44"]):
            return "president_number"
        elif any(y in q_lower for y in ["1903", "1911", "1867", "from"]):
            return "president_term"
        return "president_general"
    
    # Discovery
    if any(w in q_lower for w in ["discover", "element", "polonium", "radium"]):
        return "discovery"
    
    # Personal
    if any(w in q_lower for w in ["married", "wife", "husband", "spouse"]):
        return "spouse"
    if any(w in q_lower for w in ["daughter", "children", "son"]):
        return "children"
    
    # Goals
    if any(w in q_lower for w in ["goal", "mars", "colony"]):
        return "goal_mars"
    
    # Education
    if any(w in q_lower for w in ["university", "school", "harvard", "study"]):
        return "education"
    
    # Migration
    if any(w in q_lower for w in ["move", "immigrat", "america", "united states", "us"]):
        return "immigration"
    
    # Award (non-Nobel)
    if "award" in q_lower:
        return "award"
    
    # Death
    if any(w in q_lower for w in ["die", "death", "pass away"]):
        return "death"
    
    return "other"


def infer_person_from_question(question):
    """Try to infer which person a question is about."""
    q_lower = question.lower()
    
    if any(w in q_lower for w in ["obama", "barack", "president"]):
        return "obama"
    elif any(w in q_lower for w in ["musk", "elon", "tesla", "spacex"]):
        return "musk"
    elif any(w in q_lower for w in ["curie", "marie", "polonium", "radium"]):
        return "curie"
    
    return "unknown"


def analyze_rescored_extended_tests(experiments):
    """Analyze all rescored extended test data."""
    
    by_category = defaultdict(lambda: {
        "original_scores": [], 
        "new_scores": [], 
        "questions": [],
        "improvements": []
    })
    by_person = defaultdict(lambda: {
        "original_scores": [], 
        "new_scores": [], 
        "questions": []
    })
    by_person_category = defaultdict(lambda: defaultdict(lambda: {
        "original_scores": [], 
        "new_scores": []
    }))
    
    for exp in experiments:
        # Extended test sample turns
        ext_turns = exp.get('extended_test', {}).get('sample_turns', [])
        
        for turn in ext_turns:
            question = turn.get('question', '')
            original = turn.get('original_score', 0)
            new = turn.get('new_score', 0)
            reasoning = turn.get('reasoning', '')
            
            category = categorize_question(question)
            person = infer_person_from_question(question)
            improvement = new - original
            
            by_category[category]["original_scores"].append(original)
            by_category[category]["new_scores"].append(new)
            by_category[category]["questions"].append(question)
            by_category[category]["improvements"].append(improvement)
            
            by_person[person]["original_scores"].append(original)
            by_person[person]["new_scores"].append(new)
            by_person[person]["questions"].append(question)
            
            by_person_category[person][category]["original_scores"].append(original)
            by_person_category[person][category]["new_scores"].append(new)
        
        # Conversation turns
        conv_turns = exp.get('conversation_6turn', {}).get('turns', [])
        
        for turn in conv_turns:
            question = turn.get('question', '')
            original = turn.get('original_score', 0)
            new = turn.get('new_score', 0)
            
            category = categorize_question(question)
            person = infer_person_from_question(question)
            improvement = new - original
            
            by_category[category]["original_scores"].append(original)
            by_category[category]["new_scores"].append(new)
            by_category[category]["questions"].append(question)
            by_category[category]["improvements"].append(improvement)
            
            by_person[person]["original_scores"].append(original)
            by_person[person]["new_scores"].append(new)
            
            by_person_category[person][category]["original_scores"].append(original)
            by_person_category[person][category]["new_scores"].append(new)
        
        # Correction questions
        corr_questions = exp.get('correction_test', {}).get('questions', [])
        
        for q in corr_questions:
            question = q.get('question', '')
            original = q.get('original_score', 0)
            new = q.get('new_score', 0)
            
            category = categorize_question(question)
            if "correction" not in category:
                category = category + "_correction"
            
            person = infer_person_from_question(question)
            improvement = new - original
            
            by_category[category]["original_scores"].append(original)
            by_category[category]["new_scores"].append(new)
            by_category[category]["questions"].append(question)
            by_category[category]["improvements"].append(improvement)
            
            by_person[person]["original_scores"].append(original)
            by_person[person]["new_scores"].append(new)
            
            by_person_category[person][category]["original_scores"].append(original)
            by_person_category[person][category]["new_scores"].append(new)
    
    return {
        "by_category": dict(by_category),
        "by_person": dict(by_person),
        "by_person_category": {p: dict(cats) for p, cats in by_person_category.items()},
    }


def analyze_single_rescored(exp):
    """Analyze a single rescored experiment."""
    
    by_category = defaultdict(lambda: {"original": [], "new": []})
    by_person = defaultdict(lambda: {"original": [], "new": []})
    
    # Extended test
    for turn in exp.get('extended_test', {}).get('sample_turns', []):
        question = turn.get('question', '')
        category = categorize_question(question)
        person = infer_person_from_question(question)
        
        by_category[category]["original"].append(turn.get('original_score', 0))
        by_category[category]["new"].append(turn.get('new_score', 0))
        by_person[person]["original"].append(turn.get('original_score', 0))
        by_person[person]["new"].append(turn.get('new_score', 0))
    
    # Conversation
    for turn in exp.get('conversation_6turn', {}).get('turns', []):
        question = turn.get('question', '')
        category = categorize_question(question)
        person = infer_person_from_question(question)
        
        by_category[category]["original"].append(turn.get('original_score', 0))
        by_category[category]["new"].append(turn.get('new_score', 0))
        by_person[person]["original"].append(turn.get('original_score', 0))
        by_person[person]["new"].append(turn.get('new_score', 0))
    
    # Correction
    for q in exp.get('correction_test', {}).get('questions', []):
        question = q.get('question', '')
        category = categorize_question(question) + "_correction"
        person = infer_person_from_question(question)
        
        by_category[category]["original"].append(q.get('original_score', 0))
        by_category[category]["new"].append(q.get('new_score', 0))
        by_person[person]["original"].append(q.get('original_score', 0))
        by_person[person]["new"].append(q.get('new_score', 0))
    
    return {
        "filename": exp.get('_filename', 'unknown'),
        "original_file": exp.get('original_file', 'unknown'),
        "by_category": dict(by_category),
        "by_person": dict(by_person),
        "conv_original": exp.get('conversation_6turn', {}).get('original_score', 0),
        "conv_new": exp.get('conversation_6turn', {}).get('new_score', 0),
        "corr_original": exp.get('correction_test', {}).get('original_score', 0),
        "corr_new": exp.get('correction_test', {}).get('new_score', 0),
        "ext_original": exp.get('extended_test', {}).get('original_score', 0),
        "ext_new": exp.get('extended_test', {}).get('new_score', 0),
    }


def calculate_stats(scores):
    """Calculate statistics for a list of scores."""
    if not scores:
        return {"avg": 0, "count": 0}
    return {
        "avg": sum(scores) / len(scores),
        "count": len(scores),
        "min": min(scores),
        "max": max(scores)
    }


def generate_rescored_category_report(experiments, output_path="category_analysis_rescored.md"):
    """Generate category analysis report for rescored experiments."""
    
    analysis = analyze_rescored_extended_tests(experiments)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 📊 Category Analysis - Gemini Rescored (Aggregated)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Rescored experiments analyzed: {len(experiments)}\n\n")
        
        # ========== BY PERSON ==========
        f.write("---\n\n## 👤 Performance by Person (Gemini vs Original)\n\n")
        f.write("| Person | Original Avg | Gemini Avg | Change | Questions |\n")
        f.write("|--------|--------------|------------|--------|----------|\n")
        
        for person in sorted(analysis["by_person"].keys()):
            data = analysis["by_person"][person]
            orig_avg = sum(data["original_scores"]) / len(data["original_scores"]) if data["original_scores"] else 0
            new_avg = sum(data["new_scores"]) / len(data["new_scores"]) if data["new_scores"] else 0
            diff = new_avg - orig_avg
            
            if diff > 0.05:
                change = f"📈 +{diff:.0%}"
            elif diff < -0.05:
                change = f"📉 {diff:.0%}"
            else:
                change = f"➡️ {diff:+.0%}"
            
            status = "✅" if new_avg >= 0.7 else "🟡" if new_avg >= 0.5 else "❌"
            f.write(f"| {status} {person} | {orig_avg:.0%} | {new_avg:.0%} | {change} | {len(data['original_scores'])} |\n")
        f.write("\n")
        
        # ========== BY CATEGORY ==========
        f.write("---\n\n## 📁 Performance by Category (Gemini vs Original)\n\n")
        f.write("| Category | Original | Gemini | Change | Count | Status |\n")
        f.write("|----------|----------|--------|--------|-------|--------|\n")
        
        cat_stats = []
        for category, data in analysis["by_category"].items():
            orig_avg = sum(data["original_scores"]) / len(data["original_scores"]) if data["original_scores"] else 0
            new_avg = sum(data["new_scores"]) / len(data["new_scores"]) if data["new_scores"] else 0
            diff = new_avg - orig_avg
            cat_stats.append({
                "category": category,
                "original": orig_avg,
                "new": new_avg,
                "diff": diff,
                "count": len(data["original_scores"])
            })
        
        # Sort by Gemini score (lowest first to show problem areas)
        cat_stats.sort(key=lambda x: x["new"])
        
        for stats in cat_stats:
            if stats["diff"] > 0.1:
                change = f"📈 +{stats['diff']:.0%}"
            elif stats["diff"] < -0.1:
                change = f"📉 {stats['diff']:.0%}"
            else:
                change = f"➡️ {stats['diff']:+.0%}"
            
            if stats["new"] >= 0.7:
                status = "✅ Strong"
            elif stats["new"] >= 0.5:
                status = "🟡 OK"
            elif stats["new"] >= 0.3:
                status = "⚠️ Weak"
            else:
                status = "❌ Failing"
            
            f.write(f"| {stats['category']} | {stats['original']:.0%} | {stats['new']:.0%} | {change} | {stats['count']} | {status} |\n")
        f.write("\n")
        
        # ========== BIGGEST IMPROVEMENTS ==========
        f.write("---\n\n## 📈 Categories Most Helped by Gemini Scoring\n\n")
        f.write("These categories got higher scores with semantic evaluation:\n\n")
        
        improved = sorted(cat_stats, key=lambda x: x["diff"], reverse=True)[:5]
        for stats in improved:
            if stats["diff"] > 0:
                f.write(f"- **{stats['category']}**: {stats['original']:.0%} → {stats['new']:.0%} (+{stats['diff']:.0%})\n")
        f.write("\n")
        
        # ========== BIGGEST DROPS ==========
        f.write("---\n\n## 📉 Categories Penalized by Gemini Scoring\n\n")
        f.write("Gemini found issues that keyword matching missed:\n\n")
        
        dropped = sorted(cat_stats, key=lambda x: x["diff"])[:5]
        for stats in dropped:
            if stats["diff"] < 0:
                f.write(f"- **{stats['category']}**: {stats['original']:.0%} → {stats['new']:.0%} ({stats['diff']:.0%})\n")
        f.write("\n")
        
        # ========== PERSON × CATEGORY HEATMAP ==========
        f.write("---\n\n## 🔥 Person × Category Heatmap (Gemini Scores)\n\n")
        
        persons = sorted(analysis["by_person_category"].keys())
        all_categories = set()
        for p in persons:
            all_categories.update(analysis["by_person_category"][p].keys())
        categories = sorted(all_categories)
        
        f.write("| Category |")
        for p in persons:
            f.write(f" {p} |")
        f.write("\n")
        
        f.write("|----------|")
        for _ in persons:
            f.write("------|")
        f.write("\n")
        
        for cat in categories:
            f.write(f"| {cat} |")
            for p in persons:
                data = analysis["by_person_category"][p].get(cat, {"new_scores": []})
                if data["new_scores"]:
                    avg = sum(data["new_scores"]) / len(data["new_scores"])
                    if avg >= 0.7:
                        cell = f"✅{avg:.0%}"
                    elif avg >= 0.5:
                        cell = f"🟡{avg:.0%}"
                    elif avg >= 0.3:
                        cell = f"⚠️{avg:.0%}"
                    else:
                        cell = f"❌{avg:.0%}"
                else:
                    cell = "—"
                f.write(f" {cell} |")
            f.write("\n")
        f.write("\n")
        
        # ========== RECOMMENDATIONS ==========
        f.write("---\n\n## 💡 Recommendations\n\n")
        
        # Find worst categories by Gemini score
        worst = [s for s in cat_stats if s["new"] < 0.5][:5]
        if worst:
            f.write("### Priority Training Targets (Gemini < 50%)\n\n")
            for s in worst:
                f.write(f"- **{s['category']}**: {s['new']:.0%} — needs more training examples\n")
            f.write("\n")
        
        # Find categories where Gemini dropped score significantly
        falsepositives = [s for s in cat_stats if s["diff"] < -0.15]
        if falsepositives:
            f.write("### False Positives Exposed by Gemini\n\n")
            f.write("These categories had keyword matches but poor semantic accuracy:\n\n")
            for s in falsepositives:
                f.write(f"- **{s['category']}**: Original gave {s['original']:.0%}, Gemini gave {s['new']:.0%}\n")
            f.write("\n")
    
    return output_path


def generate_individual_rescored_reports(experiments, output_path="individual_rescored_analysis.md"):
    """Generate individual analysis for each rescored experiment."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 📋 Individual Rescored Experiment Analysis\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total rescored experiments: {len(experiments)}\n\n")
        
        # Quick comparison table
        f.write("---\n\n## 📈 Quick Comparison\n\n")
        f.write("| Experiment | Conv Orig | Conv Gemini | Corr Orig | Corr Gemini | Ext Orig | Ext Gemini |\n")
        f.write("|------------|-----------|-------------|-----------|-------------|----------|------------|\n")
        
        exp_analyses = []
        for exp in experiments:
            analysis = analyze_single_rescored(exp)
            exp_analyses.append(analysis)
            
            f.write(f"| {analysis['original_file'][:20]}... | ")
            f.write(f"{analysis['conv_original']:.0%} | {analysis['conv_new']:.0%} | ")
            f.write(f"{analysis['corr_original']:.0%} | {analysis['corr_new']:.0%} | ")
            f.write(f"{analysis['ext_original']:.0%} | {analysis['ext_new']:.0%} |\n")
        
        f.write("\n")
        
        # Individual sections
        for i, analysis in enumerate(exp_analyses):
            f.write("---\n\n")
            f.write(f"## 🧪 Experiment {i+1}: {analysis['original_file']}\n\n")
            
            # Summary scores
            f.write("### 📊 Score Comparison\n\n")
            f.write("| Test | Original | Gemini | Change |\n")
            f.write("|------|----------|--------|--------|\n")
            
            conv_diff = analysis['conv_new'] - analysis['conv_original']
            corr_diff = analysis['corr_new'] - analysis['corr_original']
            ext_diff = analysis['ext_new'] - analysis['ext_original']
            
            f.write(f"| Conversation | {analysis['conv_original']:.0%} | {analysis['conv_new']:.0%} | {conv_diff:+.0%} |\n")
            f.write(f"| Correction | {analysis['corr_original']:.0%} | {analysis['corr_new']:.0%} | {corr_diff:+.0%} |\n")
            f.write(f"| Extended | {analysis['ext_original']:.0%} | {analysis['ext_new']:.0%} | {ext_diff:+.0%} |\n\n")
            
            # By Person
            f.write("### 👤 By Person\n\n")
            f.write("| Person | Original | Gemini | Change |\n")
            f.write("|--------|----------|--------|--------|\n")
            
            for person in ["obama", "musk", "curie", "unknown"]:
                data = analysis["by_person"].get(person, {"original": [], "new": []})
                if data["original"]:
                    orig = sum(data["original"]) / len(data["original"])
                    new = sum(data["new"]) / len(data["new"])
                    diff = new - orig
                    status = "✅" if new >= 0.7 else "🟡" if new >= 0.5 else "❌"
                    f.write(f"| {status} {person} | {orig:.0%} | {new:.0%} | {diff:+.0%} |\n")
            f.write("\n")
            
            # By Category
            f.write("### 📁 By Category\n\n")
            
            cat_list = []
            for cat, data in analysis["by_category"].items():
                if data["original"]:
                    orig = sum(data["original"]) / len(data["original"])
                    new = sum(data["new"]) / len(data["new"])
                    cat_list.append({"cat": cat, "orig": orig, "new": new, "diff": new - orig})
            
            # Worst by Gemini
            worst = sorted(cat_list, key=lambda x: x["new"])[:5]
            if worst:
                f.write("**❌ Weakest Categories (Gemini):**\n\n")
                for c in worst:
                    if c["new"] < 0.7:
                        f.write(f"- {c['cat']}: {c['orig']:.0%} → {c['new']:.0%} ({c['diff']:+.0%})\n")
                f.write("\n")
            
            # Best improvements
            improved = sorted(cat_list, key=lambda x: x["diff"], reverse=True)[:3]
            if improved and improved[0]["diff"] > 0:
                f.write("**📈 Most Improved:**\n\n")
                for c in improved:
                    if c["diff"] > 0.05:
                        f.write(f"- {c['cat']}: {c['orig']:.0%} → {c['new']:.0%} (+{c['diff']:.0%})\n")
                f.write("\n")
        
        # ========== TREND ANALYSIS ==========
        if len(exp_analyses) >= 2:
            f.write("---\n\n## 📈 Trends Across Experiments\n\n")
            
            f.write("### Gemini Score Evolution\n\n")
            f.write("| Metric |")
            for a in exp_analyses:
                f.write(f" {a['original_file'][:10]}... |")
            f.write(" Trend |\n")
            
            f.write("|--------|")
            for _ in exp_analyses:
                f.write("------|")
            f.write("------|\n")
            
            # Conversation
            conv_scores = [a['conv_new'] for a in exp_analyses]
            f.write("| Conv (Gemini) |")
            for s in conv_scores:
                f.write(f" {s:.0%} |")
            trend = "📈" if conv_scores[-1] > conv_scores[0] else "📉" if conv_scores[-1] < conv_scores[0] else "➡️"
            f.write(f" {trend} |\n")
            
            # Correction
            corr_scores = [a['corr_new'] for a in exp_analyses]
            f.write("| Corr (Gemini) |")
            for s in corr_scores:
                f.write(f" {s:.0%} |")
            trend = "📈" if corr_scores[-1] > corr_scores[0] else "📉" if corr_scores[-1] < corr_scores[0] else "➡️"
            f.write(f" {trend} |\n")
            
            # Extended
            ext_scores = [a['ext_new'] for a in exp_analyses]
            f.write("| Ext (Gemini) |")
            for s in ext_scores:
                f.write(f" {s:.0%} |")
            trend = "📈" if ext_scores[-1] > ext_scores[0] else "📉" if ext_scores[-1] < ext_scores[0] else "➡️"
            f.write(f" {trend} |\n")
            
            f.write("\n")
    
    return output_path


def main():
    print("\n" + "=" * 60)
    print("📊 Category Analysis - Gemini Rescored")
    print("=" * 60 + "\n")
    
    experiments = load_rescored_experiments(".")
    
    if not experiments:
        print("❌ No rescored files found!")
        print("   Looking for: *_gemini_rescored.json")
        return
    
    print(f"✓ Loaded {len(experiments)} rescored experiment(s)\n")
    
    # Analyze
    analysis = analyze_rescored_extended_tests(experiments)
    
    # Quick stats
    print("-" * 40)
    print("Performance by Person (Gemini Scores):")
    print("-" * 40)
    for person, data in sorted(analysis["by_person"].items()):
        if data["new_scores"]:
            new_avg = sum(data["new_scores"]) / len(data["new_scores"])
            orig_avg = sum(data["original_scores"]) / len(data["original_scores"])
            diff = new_avg - orig_avg
            status = "✅" if new_avg >= 0.6 else "⚠️" if new_avg >= 0.4 else "❌"
            print(f"  {status} {person}: {new_avg:.0%} (was {orig_avg:.0%}, {diff:+.0%})")
    
    print("\n" + "-" * 40)
    print("Weakest Categories (Gemini < 50%):")
    print("-" * 40)
    for category, data in analysis["by_category"].items():
        if data["new_scores"]:
            avg = sum(data["new_scores"]) / len(data["new_scores"])
            if avg < 0.5:
                print(f"  ❌ {category}: {avg:.0%}")
    
    # Generate reports
    report1 = generate_rescored_category_report(experiments)
    print(f"\n✓ Generated: {report1}")
    
    report2 = generate_individual_rescored_reports(experiments)
    print(f"✓ Generated: {report2}")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
