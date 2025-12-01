"""
SleepTrain - Category & Correlation Analysis
Analyzes the 100-turn extended test data to find:
- Which categories the model struggles with
- Correlations between persons
- Pattern analysis across experiments
"""

import json
import glob
import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_experiments(directory=None):
    """Load all full experiment JSON files."""
    if directory is None:
        # Default to the new organized structure
        script_dir = Path(__file__).parent.parent.parent  # Go up to project root
        directory = script_dir / "data" / "experiment_results" / "training" / "original"
    
    experiments = []
    json_files = glob.glob(os.path.join(str(directory), "full_experiment_*.json"))
    
    # Exclude rescored files
    json_files = [f for f in json_files if "rescored" not in f]
    
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


def extract_question_features(question, q_type):
    """Extract features from a question for deeper analysis."""
    q_lower = question.lower()
    
    features = {
        "is_correction": q_type == "correction",
        "mentions_wrong_date": any(y in q_lower for y in ["1867", "1903", "2002", "2009", "1911"]) and q_type == "correction",
        "is_yes_no": any(q_lower.endswith(w) for w in ["?", "right?", "correct?", "accurate?"]),
        "asks_when": any(w in q_lower for w in ["when", "what year", "what date"]),
        "asks_where": any(w in q_lower for w in ["where", "what place", "which city"]),
        "asks_what": q_lower.startswith("what"),
        "asks_who": q_lower.startswith("who"),
        "asks_how": q_lower.startswith("how"),
    }
    
    return features


def analyze_extended_tests(experiments):
    """Analyze all extended test data across experiments."""
    
    # Storage for analysis
    by_person = defaultdict(lambda: {"scores": [], "questions": []})
    by_category = defaultdict(lambda: {"scores": [], "questions": []})
    by_person_category = defaultdict(lambda: defaultdict(lambda: {"scores": [], "questions": []}))
    by_type = {"real": {"scores": [], "questions": []}, "correction": {"scores": [], "questions": []}}
    
    # Cross-person patterns (same category, different person)
    category_by_person = defaultdict(lambda: defaultdict(list))  # category -> person -> scores
    
    # Question-level tracking
    question_performance = defaultdict(lambda: {"scores": [], "persons": [], "types": []})
    
    for exp in experiments:
        exp_name = exp.get('_filename', 'unknown')
        extended = exp.get('tests', {}).get('extended_test', {})
        turns = extended.get('turns', [])
        
        for turn in turns:
            question = turn.get('question', '')
            person = turn.get('person', 'unknown')
            q_type = turn.get('type', 'real')
            score = turn.get('score', 0)
            
            # Categorize
            category = categorize_question(question)
            
            # Store by person
            by_person[person]["scores"].append(score)
            by_person[person]["questions"].append(question)
            
            # Store by category
            by_category[category]["scores"].append(score)
            by_category[category]["questions"].append(question)
            
            # Store by person+category
            by_person_category[person][category]["scores"].append(score)
            by_person_category[person][category]["questions"].append(question)
            
            # Store by type
            by_type[q_type]["scores"].append(score)
            by_type[q_type]["questions"].append(question)
            
            # Cross-person category analysis
            category_by_person[category][person].append(score)
            
            # Question-level
            q_normalized = question.lower().strip()
            question_performance[q_normalized]["scores"].append(score)
            question_performance[q_normalized]["persons"].append(person)
            question_performance[q_normalized]["types"].append(q_type)
    
    return {
        "by_person": dict(by_person),
        "by_category": dict(by_category),
        "by_person_category": {p: dict(cats) for p, cats in by_person_category.items()},
        "by_type": by_type,
        "category_by_person": {c: dict(persons) for c, persons in category_by_person.items()},
        "question_performance": dict(question_performance),
    }


def calculate_stats(scores):
    """Calculate statistics for a list of scores."""
    if not scores:
        return {"avg": 0, "min": 0, "max": 0, "count": 0, "std": 0}
    
    avg = sum(scores) / len(scores)
    variance = sum((s - avg) ** 2 for s in scores) / len(scores) if len(scores) > 1 else 0
    
    return {
        "avg": avg,
        "min": min(scores),
        "max": max(scores),
        "count": len(scores),
        "std": variance ** 0.5,
        "pass_rate": sum(1 for s in scores if s >= 0.5) / len(scores)
    }


def find_correlations(analysis):
    """Find correlations between persons and categories."""
    
    correlations = []
    
    # Find categories where all persons struggle
    category_by_person = analysis["category_by_person"]
    
    for category, persons in category_by_person.items():
        if len(persons) < 2:
            continue
        
        person_avgs = {}
        for person, scores in persons.items():
            if scores:
                person_avgs[person] = sum(scores) / len(scores)
        
        if not person_avgs:
            continue
        
        avg_all = sum(person_avgs.values()) / len(person_avgs)
        
        # Check if all persons struggle (avg < 0.5) or all succeed (avg > 0.7)
        all_struggle = all(avg < 0.5 for avg in person_avgs.values())
        all_succeed = all(avg > 0.7 for avg in person_avgs.values())
        
        if all_struggle:
            correlations.append({
                "type": "universal_struggle",
                "category": category,
                "person_scores": person_avgs,
                "overall_avg": avg_all
            })
        elif all_succeed:
            correlations.append({
                "type": "universal_success",
                "category": category,
                "person_scores": person_avgs,
                "overall_avg": avg_all
            })
        else:
            # Check for large variance between persons
            scores_list = list(person_avgs.values())
            variance = max(scores_list) - min(scores_list)
            if variance > 0.3:
                best = max(person_avgs.items(), key=lambda x: x[1])
                worst = min(person_avgs.items(), key=lambda x: x[1])
                correlations.append({
                    "type": "person_specific",
                    "category": category,
                    "best_person": best,
                    "worst_person": worst,
                    "variance": variance
                })
    
    return correlations


def find_hardest_questions(analysis, n=15):
    """Find the questions that consistently fail."""
    
    question_perf = analysis["question_performance"]
    
    # Calculate average score per question
    question_stats = []
    for question, data in question_perf.items():
        scores = data["scores"]
        if len(scores) >= 2:  # At least asked twice
            avg = sum(scores) / len(scores)
            question_stats.append({
                "question": question[:60] + "..." if len(question) > 60 else question,
                "avg_score": avg,
                "times_asked": len(scores),
                "persons": list(set(data["persons"])),
                "types": list(set(data["types"])),
                "category": categorize_question(question)
            })
    
    # Sort by average score (lowest first)
    question_stats.sort(key=lambda x: x["avg_score"])
    
    return question_stats[:n]


def analyze_single_experiment(exp):
    """Analyze a single experiment's extended test data."""
    
    by_person = defaultdict(lambda: {"scores": [], "questions": []})
    by_category = defaultdict(lambda: {"scores": [], "questions": []})
    by_type = {"real": {"scores": []}, "correction": {"scores": []}}
    by_person_category = defaultdict(lambda: defaultdict(list))
    
    extended = exp.get('tests', {}).get('extended_test', {})
    turns = extended.get('turns', [])
    
    for turn in turns:
        question = turn.get('question', '')
        person = turn.get('person', 'unknown')
        q_type = turn.get('type', 'real')
        score = turn.get('score', 0)
        category = categorize_question(question)
        
        by_person[person]["scores"].append(score)
        by_category[category]["scores"].append(score)
        by_type[q_type]["scores"].append(score)
        by_person_category[person][category].append(score)
    
    return {
        "filename": exp.get('_filename', 'unknown'),
        "metadata": exp.get('metadata', {}),
        "summary": exp.get('summary', {}),
        "total_turns": len(turns),
        "stopped_early": extended.get('stopped_early', False),
        "by_person": dict(by_person),
        "by_category": dict(by_category),
        "by_type": by_type,
        "by_person_category": {p: dict(cats) for p, cats in by_person_category.items()},
    }


def generate_individual_reports(experiments, output_path=None):
    """Generate individual analysis for each experiment."""
    if output_path is None:
        script_dir = Path(__file__).parent.parent.parent  # Go up to project root
        output_path = script_dir / "results" / "analysis_reports" / "individual_experiment_analysis.md"
    
    with open(str(output_path), 'w', encoding='utf-8') as f:
        f.write("# 📋 Individual Experiment Analysis\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {len(experiments)}\n\n")
        
        f.write("---\n\n")
        f.write("## 📈 Quick Comparison Across All Experiments\n\n")
        
        # Summary table
        f.write("| Experiment | Turns | Real Avg | Correction Avg | Obama | Musk | Curie | Stopped Early |\n")
        f.write("|------------|-------|----------|----------------|-------|------|-------|---------------|\n")
        
        exp_analyses = []
        for exp in experiments:
            analysis = analyze_single_experiment(exp)
            exp_analyses.append(analysis)
            
            real_scores = analysis["by_type"]["real"]["scores"]
            corr_scores = analysis["by_type"]["correction"]["scores"]
            real_avg = sum(real_scores) / len(real_scores) if real_scores else 0
            corr_avg = sum(corr_scores) / len(corr_scores) if corr_scores else 0
            
            person_avgs = {}
            for p in ["obama", "musk", "curie"]:
                scores = analysis["by_person"].get(p, {}).get("scores", [])
                person_avgs[p] = sum(scores) / len(scores) if scores else 0
            
            stopped = "⚠️ Yes" if analysis["stopped_early"] else "✅ No"
            
            f.write(f"| {analysis['filename'][:25]} | {analysis['total_turns']} | {real_avg:.0%} | {corr_avg:.0%} | ")
            f.write(f"{person_avgs['obama']:.0%} | {person_avgs['musk']:.0%} | {person_avgs['curie']:.0%} | {stopped} |\n")
        
        f.write("\n")
        
        # Individual experiment sections
        for i, analysis in enumerate(exp_analyses):
            f.write("---\n\n")
            f.write(f"## 🧪 Experiment {i+1}: {analysis['filename']}\n\n")
            
            # Metadata
            meta = analysis["metadata"]
            f.write("### ⚙️ Configuration\n\n")
            f.write(f"- **Learning Rate:** {meta.get('learning_rate', 'N/A')}\n")
            f.write(f"- **LoRA Rank:** {meta.get('lora_rank', 'N/A')}\n")
            f.write(f"- **LoRA Alpha:** {meta.get('lora_alpha', 'N/A')}\n")
            f.write(f"- **Total Turns:** {analysis['total_turns']}\n")
            f.write(f"- **Stopped Early:** {'Yes ⚠️' if analysis['stopped_early'] else 'No ✅'}\n\n")
            
            # Summary scores from experiment
            summary = analysis["summary"]
            if summary:
                f.write("### 📊 Summary Scores (from experiment)\n\n")
                f.write(f"- Single Question Avg: {summary.get('single_q_avg', 0):.1%}\n")
                f.write(f"- Conversation Avg: {summary.get('conversation_avg', 0):.1%}\n")
                f.write(f"- Correction Avg: {summary.get('correction_avg', 0):.1%}\n")
                f.write(f"- Extended Avg: {summary.get('extended_avg', 0):.1%}\n\n")
            
            # By Person
            f.write("### 👤 Performance by Person\n\n")
            f.write("| Person | Avg Score | Pass Rate | Questions |\n")
            f.write("|--------|-----------|-----------|----------|\n")
            
            for person in ["obama", "musk", "curie"]:
                data = analysis["by_person"].get(person, {"scores": []})
                scores = data["scores"]
                if scores:
                    avg = sum(scores) / len(scores)
                    pass_rate = sum(1 for s in scores if s >= 0.5) / len(scores)
                    status = "✅" if avg >= 0.6 else "⚠️" if avg >= 0.4 else "❌"
                    f.write(f"| {status} {person} | {avg:.0%} | {pass_rate:.0%} | {len(scores)} |\n")
            f.write("\n")
            
            # By Type
            f.write("### 🏷️ Performance by Question Type\n\n")
            f.write("| Type | Avg Score | Questions |\n")
            f.write("|------|-----------|----------|\n")
            
            for q_type in ["real", "correction"]:
                scores = analysis["by_type"][q_type]["scores"]
                if scores:
                    avg = sum(scores) / len(scores)
                    status = "✅" if avg >= 0.6 else "⚠️" if avg >= 0.4 else "❌"
                    f.write(f"| {status} {q_type} | {avg:.0%} | {len(scores)} |\n")
            f.write("\n")
            
            # By Category (top and bottom)
            f.write("### 📁 Category Performance\n\n")
            
            cat_stats = []
            for cat, data in analysis["by_category"].items():
                scores = data["scores"]
                if scores:
                    avg = sum(scores) / len(scores)
                    cat_stats.append({"category": cat, "avg": avg, "count": len(scores)})
            
            cat_stats.sort(key=lambda x: x["avg"])
            
            # Worst categories
            worst = [c for c in cat_stats if c["avg"] < 0.5][:5]
            if worst:
                f.write("**❌ Struggling Categories:**\n\n")
                for c in worst:
                    f.write(f"- {c['category']}: {c['avg']:.0%} ({c['count']} questions)\n")
                f.write("\n")
            
            # Best categories
            best = [c for c in cat_stats if c["avg"] >= 0.7]
            if best:
                f.write("**✅ Strong Categories:**\n\n")
                for c in sorted(best, key=lambda x: x["avg"], reverse=True)[:5]:
                    f.write(f"- {c['category']}: {c['avg']:.0%} ({c['count']} questions)\n")
                f.write("\n")
            
            # Person x Category mini-heatmap
            f.write("### 🔥 Person × Category Breakdown\n\n")
            
            # Get all categories for this experiment
            all_cats = set()
            for p in analysis["by_person_category"]:
                all_cats.update(analysis["by_person_category"][p].keys())
            
            if all_cats:
                f.write("| Category | Obama | Musk | Curie |\n")
                f.write("|----------|-------|------|-------|\n")
                
                for cat in sorted(all_cats):
                    f.write(f"| {cat} |")
                    for person in ["obama", "musk", "curie"]:
                        scores = analysis["by_person_category"].get(person, {}).get(cat, [])
                        if scores:
                            avg = sum(scores) / len(scores)
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
        
        # ========== TREND ANALYSIS ==========
        f.write("---\n\n")
        f.write("## 📈 Trend Analysis Across Experiments\n\n")
        
        if len(exp_analyses) >= 2:
            f.write("### Performance Evolution\n\n")
            
            # Track metrics over time
            f.write("| Metric | ")
            for a in exp_analyses:
                f.write(f"{a['filename'][:12]}... | ")
            f.write("Trend |\n")
            
            f.write("|--------|")
            for _ in exp_analyses:
                f.write("------|")
            f.write("------|\n")
            
            # Real questions
            real_avgs = []
            for a in exp_analyses:
                scores = a["by_type"]["real"]["scores"]
                real_avgs.append(sum(scores) / len(scores) if scores else 0)
            
            f.write("| Real Questions |")
            for avg in real_avgs:
                f.write(f" {avg:.0%} |")
            trend = "📈" if real_avgs[-1] > real_avgs[0] else "📉" if real_avgs[-1] < real_avgs[0] else "➡️"
            f.write(f" {trend} |\n")
            
            # Correction questions
            corr_avgs = []
            for a in exp_analyses:
                scores = a["by_type"]["correction"]["scores"]
                corr_avgs.append(sum(scores) / len(scores) if scores else 0)
            
            f.write("| Corrections |")
            for avg in corr_avgs:
                f.write(f" {avg:.0%} |")
            trend = "📈" if corr_avgs[-1] > corr_avgs[0] else "📉" if corr_avgs[-1] < corr_avgs[0] else "➡️"
            f.write(f" {trend} |\n")
            
            # Per person trends
            for person in ["obama", "musk", "curie"]:
                person_avgs = []
                for a in exp_analyses:
                    scores = a["by_person"].get(person, {}).get("scores", [])
                    person_avgs.append(sum(scores) / len(scores) if scores else 0)
                
                f.write(f"| {person.title()} |")
                for avg in person_avgs:
                    f.write(f" {avg:.0%} |")
                trend = "📈" if person_avgs[-1] > person_avgs[0] else "📉" if person_avgs[-1] < person_avgs[0] else "➡️"
                f.write(f" {trend} |\n")
            
            f.write("\n")
            
            # Identify improving/degrading categories
            f.write("### Category Trends\n\n")
            
            # Get all categories across experiments
            all_categories = set()
            for a in exp_analyses:
                all_categories.update(a["by_category"].keys())
            
            improving = []
            degrading = []
            
            for cat in all_categories:
                first_scores = exp_analyses[0]["by_category"].get(cat, {}).get("scores", [])
                last_scores = exp_analyses[-1]["by_category"].get(cat, {}).get("scores", [])
                
                if first_scores and last_scores:
                    first_avg = sum(first_scores) / len(first_scores)
                    last_avg = sum(last_scores) / len(last_scores)
                    diff = last_avg - first_avg
                    
                    if diff > 0.15:
                        improving.append({"category": cat, "first": first_avg, "last": last_avg, "diff": diff})
                    elif diff < -0.15:
                        degrading.append({"category": cat, "first": first_avg, "last": last_avg, "diff": diff})
            
            if improving:
                f.write("**📈 Improving Categories:**\n\n")
                for item in sorted(improving, key=lambda x: x["diff"], reverse=True)[:5]:
                    f.write(f"- {item['category']}: {item['first']:.0%} → {item['last']:.0%} (+{item['diff']:.0%})\n")
                f.write("\n")
            
            if degrading:
                f.write("**📉 Degrading Categories:**\n\n")
                for item in sorted(degrading, key=lambda x: x["diff"])[:5]:
                    f.write(f"- {item['category']}: {item['first']:.0%} → {item['last']:.0%} ({item['diff']:.0%})\n")
                f.write("\n")
    
    return output_path


def generate_category_report(experiments, output_path=None):
    """Generate comprehensive category analysis report."""
    if output_path is None:
        script_dir = Path(__file__).parent.parent.parent  # Go up to project root
        output_path = script_dir / "results" / "analysis_reports" / "category_analysis_report.md"
    
    analysis = analyze_extended_tests(experiments)
    correlations = find_correlations(analysis)
    hardest = find_hardest_questions(analysis)
    
    with open(str(output_path), 'w', encoding='utf-8') as f:
        f.write("# 📊 Category & Correlation Analysis Report (Aggregated)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiments analyzed: {len(experiments)}\n\n")
        
        # Count total turns
        total_turns = sum(len(exp.get('tests', {}).get('extended_test', {}).get('turns', [])) for exp in experiments)
        f.write(f"Total extended test turns analyzed: **{total_turns}**\n\n")
        
        # ========== BY PERSON ==========
        f.write("---\n\n## 👤 Performance by Person\n\n")
        f.write("| Person | Avg Score | Pass Rate | Count |\n")
        f.write("|--------|-----------|-----------|-------|\n")
        
        for person, data in sorted(analysis["by_person"].items()):
            stats = calculate_stats(data["scores"])
            status = "✅" if stats["avg"] >= 0.6 else "⚠️" if stats["avg"] >= 0.4 else "❌"
            f.write(f"| {status} {person} | {stats['avg']:.0%} | {stats['pass_rate']:.0%} | {stats['count']} |\n")
        f.write("\n")
        
        # ========== BY TYPE ==========
        f.write("---\n\n## 🏷️ Performance by Question Type\n\n")
        f.write("| Type | Avg Score | Pass Rate | Count |\n")
        f.write("|------|-----------|-----------|-------|\n")
        
        for q_type, data in analysis["by_type"].items():
            stats = calculate_stats(data["scores"])
            status = "✅" if stats["avg"] >= 0.6 else "⚠️" if stats["avg"] >= 0.4 else "❌"
            f.write(f"| {status} {q_type} | {stats['avg']:.0%} | {stats['pass_rate']:.0%} | {stats['count']} |\n")
        f.write("\n")
        
        # ========== BY CATEGORY ==========
        f.write("---\n\n## 📁 Performance by Category\n\n")
        f.write("| Category | Avg Score | Pass Rate | Count | Status |\n")
        f.write("|----------|-----------|-----------|-------|--------|\n")
        
        category_stats = []
        for category, data in analysis["by_category"].items():
            stats = calculate_stats(data["scores"])
            stats["category"] = category
            category_stats.append(stats)
        
        # Sort by avg score
        category_stats.sort(key=lambda x: x["avg"])
        
        for stats in category_stats:
            if stats["avg"] >= 0.7:
                status = "✅ Strong"
            elif stats["avg"] >= 0.5:
                status = "🟡 OK"
            elif stats["avg"] >= 0.3:
                status = "⚠️ Weak"
            else:
                status = "❌ Failing"
            f.write(f"| {stats['category']} | {stats['avg']:.0%} | {stats['pass_rate']:.0%} | {stats['count']} | {status} |\n")
        f.write("\n")
        
        # ========== PERSON x CATEGORY HEATMAP ==========
        f.write("---\n\n## 🔥 Person × Category Heatmap\n\n")
        
        persons = sorted(analysis["by_person_category"].keys())
        all_categories = set()
        for p in persons:
            all_categories.update(analysis["by_person_category"][p].keys())
        categories = sorted(all_categories)
        
        # Header
        f.write("| Category |")
        for p in persons:
            f.write(f" {p} |")
        f.write("\n")
        
        f.write("|----------|")
        for _ in persons:
            f.write("------|")
        f.write("\n")
        
        # Data rows
        for cat in categories:
            f.write(f"| {cat} |")
            for p in persons:
                data = analysis["by_person_category"][p].get(cat, {"scores": []})
                if data["scores"]:
                    avg = sum(data["scores"]) / len(data["scores"])
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
        
        # ========== CORRELATIONS ==========
        f.write("---\n\n## 🔗 Cross-Person Correlations\n\n")
        
        # Universal struggles
        struggles = [c for c in correlations if c["type"] == "universal_struggle"]
        if struggles:
            f.write("### ❌ Categories Where ALL Persons Struggle\n\n")
            f.write("These categories fail regardless of which person is asked about:\n\n")
            for corr in struggles:
                f.write(f"**{corr['category']}** (avg: {corr['overall_avg']:.0%})\n")
                for person, score in corr["person_scores"].items():
                    f.write(f"  - {person}: {score:.0%}\n")
                f.write("\n")
        
        # Universal successes
        successes = [c for c in correlations if c["type"] == "universal_success"]
        if successes:
            f.write("### ✅ Categories Where ALL Persons Succeed\n\n")
            for corr in successes:
                f.write(f"**{corr['category']}** (avg: {corr['overall_avg']:.0%})\n")
                for person, score in corr["person_scores"].items():
                    f.write(f"  - {person}: {score:.0%}\n")
                f.write("\n")
        
        # Person-specific issues
        specific = [c for c in correlations if c["type"] == "person_specific"]
        if specific:
            f.write("### 🎯 Person-Specific Performance Gaps\n\n")
            f.write("Same category, very different results by person:\n\n")
            for corr in sorted(specific, key=lambda x: x["variance"], reverse=True)[:10]:
                f.write(f"**{corr['category']}** (variance: {corr['variance']:.0%})\n")
                f.write(f"  - Best: {corr['best_person'][0]} ({corr['best_person'][1]:.0%})\n")
                f.write(f"  - Worst: {corr['worst_person'][0]} ({corr['worst_person'][1]:.0%})\n\n")
        
        # ========== HARDEST QUESTIONS ==========
        f.write("---\n\n## 🚨 Hardest Questions (Consistently Fail)\n\n")
        f.write("Questions asked multiple times with lowest average scores:\n\n")
        
        f.write("| Question | Avg | Times | Category | Person(s) |\n")
        f.write("|----------|-----|-------|----------|----------|\n")
        
        for q in hardest:
            persons_str = ", ".join(q["persons"])
            f.write(f"| {q['question']} | {q['avg_score']:.0%} | {q['times_asked']} | {q['category']} | {persons_str} |\n")
        f.write("\n")
        
        # ========== RECOMMENDATIONS ==========
        f.write("---\n\n## 💡 Training Recommendations\n\n")
        
        # Find worst categories
        worst_cats = [s for s in category_stats if s["avg"] < 0.5][:5]
        if worst_cats:
            f.write("### 1. Priority Categories to Improve\n\n")
            for cat in worst_cats:
                f.write(f"- **{cat['category']}**: {cat['avg']:.0%} avg → Add more training examples\n")
            f.write("\n")
        
        # Correction vs Real gap
        real_stats = calculate_stats(analysis["by_type"]["real"]["scores"])
        corr_stats = calculate_stats(analysis["by_type"]["correction"]["scores"])
        gap = real_stats["avg"] - corr_stats["avg"]
        
        if gap > 0.15:
            f.write("### 2. Correction Training Gap\n\n")
            f.write(f"Real questions: {real_stats['avg']:.0%} vs Corrections: {corr_stats['avg']:.0%} (gap: {gap:.0%})\n\n")
            f.write("**Action:** Add more explicit correction training examples with \"No, that's wrong\" patterns.\n\n")
        
        # Person-specific recommendations
        person_stats = {p: calculate_stats(d["scores"]) for p, d in analysis["by_person"].items()}
        worst_person = min(person_stats.items(), key=lambda x: x[1]["avg"])
        
        if worst_person[1]["avg"] < 0.5:
            f.write(f"### 3. Focus on {worst_person[0].title()}\n\n")
            f.write(f"This person has the lowest score ({worst_person[1]['avg']:.0%}).\n\n")
            
            # Find their worst categories
            person_cats = analysis["by_person_category"].get(worst_person[0], {})
            cat_avgs = [(c, sum(d["scores"])/len(d["scores"])) for c, d in person_cats.items() if d["scores"]]
            cat_avgs.sort(key=lambda x: x[1])
            
            f.write(f"Weakest categories for {worst_person[0]}:\n")
            for cat, avg in cat_avgs[:5]:
                f.write(f"  - {cat}: {avg:.0%}\n")
            f.write("\n")
    
    return output_path


def main():
    print("\n" + "=" * 60)
    print("📊 Category & Correlation Analysis")
    print("=" * 60 + "\n")
    
    experiments = load_experiments()  # Uses default path to data/experiment_results/training/original/
    
    if not experiments:
        print("❌ No experiment files found!")
        return
    
    print(f"✓ Loaded {len(experiments)} experiment(s)\n")
    
    # Run analysis
    analysis = analyze_extended_tests(experiments)
    
    # Quick stats
    print("-" * 40)
    print("Performance by Person:")
    print("-" * 40)
    for person, data in sorted(analysis["by_person"].items()):
        stats = calculate_stats(data["scores"])
        status = "✅" if stats["avg"] >= 0.6 else "⚠️" if stats["avg"] >= 0.4 else "❌"
        print(f"  {status} {person}: {stats['avg']:.0%} ({stats['count']} questions)")
    
    print("\n" + "-" * 40)
    print("Performance by Type:")
    print("-" * 40)
    for q_type, data in analysis["by_type"].items():
        stats = calculate_stats(data["scores"])
        status = "✅" if stats["avg"] >= 0.6 else "⚠️" if stats["avg"] >= 0.4 else "❌"
        print(f"  {status} {q_type}: {stats['avg']:.0%} ({stats['count']} questions)")
    
    # Find worst categories
    print("\n" + "-" * 40)
    print("Worst Categories (< 50%):")
    print("-" * 40)
    for category, data in analysis["by_category"].items():
        stats = calculate_stats(data["scores"])
        if stats["avg"] < 0.5 and stats["count"] >= 5:
            print(f"  ❌ {category}: {stats['avg']:.0%} ({stats['count']} questions)")
    
    # Generate reports
    report_path = generate_category_report(experiments)
    print(f"\n✓ Generated: {report_path}")
    
    individual_path = generate_individual_reports(experiments)
    print(f"✓ Generated: {individual_path}")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
