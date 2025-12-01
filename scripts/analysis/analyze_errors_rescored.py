"""
SleepTrain Error Analysis - For Gemini Rescored Files
Analyzes *_gemini_rescored.json files to understand scoring differences.
"""

import json
import glob
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_rescored_experiments(directory=None):
    """Load all Gemini rescored JSON files."""
    if directory is None:
        # Default to the new organized structure
        script_dir = Path(__file__).parent.parent.parent  # Go up to project root
        directory = script_dir / "data" / "experiment_results" / "training" / "rescored"
    
    experiments = []
    json_files = glob.glob(os.path.join(str(directory), "*_gemini_rescored.json"))
    
    for filepath in sorted(json_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_filename'] = os.path.basename(filepath)
                experiments.append(data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return experiments


def analyze_score_changes(experiments):
    """Analyze which questions improved or worsened with Gemini scoring."""
    
    changes = {
        'improved': [],      # new_score > original_score + 0.1
        'worsened': [],      # new_score < original_score - 0.1
        'unchanged': [],     # within ±0.1
    }
    
    for exp in experiments:
        filename = exp.get('_filename', 'unknown')
        
        # Conversation turns
        for turn in exp.get('conversation_6turn', {}).get('turns', []):
            diff = turn['new_score'] - turn['original_score']
            item = {
                'question': turn['question'],
                'original': turn['original_score'],
                'new': turn['new_score'],
                'diff': diff,
                'reasoning': turn.get('reasoning', ''),
                'type': 'conversation',
                'source': filename
            }
            
            if diff > 0.1:
                changes['improved'].append(item)
            elif diff < -0.1:
                changes['worsened'].append(item)
            else:
                changes['unchanged'].append(item)
        
        # Correction questions
        for q in exp.get('correction_test', {}).get('questions', []):
            diff = q['new_score'] - q['original_score']
            item = {
                'question': q['question'],
                'original': q['original_score'],
                'new': q['new_score'],
                'diff': diff,
                'reasoning': q.get('reasoning', ''),
                'indicated_wrong': q.get('indicated_wrong', None),
                'provided_correct': q.get('provided_correct', None),
                'type': 'correction',
                'source': filename
            }
            
            if diff > 0.1:
                changes['improved'].append(item)
            elif diff < -0.1:
                changes['worsened'].append(item)
            else:
                changes['unchanged'].append(item)
        
        # Extended test (sample)
        for turn in exp.get('extended_test', {}).get('sample_turns', []):
            diff = turn['new_score'] - turn['original_score']
            item = {
                'question': turn['question'],
                'original': turn['original_score'],
                'new': turn['new_score'],
                'diff': diff,
                'reasoning': turn.get('reasoning', ''),
                'type': 'extended',
                'turn': turn.get('turn', 0),
                'source': filename
            }
            
            if diff > 0.1:
                changes['improved'].append(item)
            elif diff < -0.1:
                changes['worsened'].append(item)
            else:
                changes['unchanged'].append(item)
    
    # Sort by magnitude of change
    changes['improved'].sort(key=lambda x: x['diff'], reverse=True)
    changes['worsened'].sort(key=lambda x: x['diff'])
    
    return changes


def analyze_correction_details(experiments):
    """Analyze correction test results in detail using Gemini's evaluation."""
    
    results = {
        'perfect': [],          # indicated_wrong AND provided_correct
        'gave_info_only': [],   # provided_correct but NOT indicated_wrong
        'said_wrong_only': [],  # indicated_wrong but NOT provided_correct
        'failed': [],           # neither
    }
    
    for exp in experiments:
        for q in exp.get('correction_test', {}).get('questions', []):
            item = {
                'question': q['question'],
                'original_score': q['original_score'],
                'new_score': q['new_score'],
                'reasoning': q.get('reasoning', ''),
            }
            
            indicated = q.get('indicated_wrong', False)
            provided = q.get('provided_correct', False)
            
            if indicated and provided:
                results['perfect'].append(item)
            elif provided and not indicated:
                results['gave_info_only'].append(item)
            elif indicated and not provided:
                results['said_wrong_only'].append(item)
            else:
                results['failed'].append(item)
    
    return results


def analyze_scoring_patterns(experiments):
    """Find patterns in when keyword scoring differs from semantic scoring."""
    
    patterns = {
        'format_lenient': [],    # Gemini gave credit for equivalent formats
        'missing_keywords': [],  # Original penalized for missing keywords that weren't needed
        'false_positive': [],    # Original gave points that Gemini removed
        'semantic_miss': [],     # Both agreed on low score
    }
    
    for exp in experiments:
        all_items = []
        
        # Collect all scored items
        for turn in exp.get('conversation_6turn', {}).get('turns', []):
            all_items.append(turn)
        for q in exp.get('correction_test', {}).get('questions', []):
            all_items.append(q)
        for turn in exp.get('extended_test', {}).get('sample_turns', []):
            all_items.append(turn)
        
        for item in all_items:
            orig = item['original_score']
            new = item['new_score']
            reasoning = item.get('reasoning', '').lower()
            
            # Improved - likely format leniency
            if new > orig + 0.2:
                if any(word in reasoning for word in ['format', 'equivalent', 'correct', 'accurately', 'correctly']):
                    patterns['format_lenient'].append({
                        'question': item['question'],
                        'original': orig,
                        'new': new,
                        'reasoning': item.get('reasoning', '')
                    })
                else:
                    patterns['missing_keywords'].append({
                        'question': item['question'],
                        'original': orig,
                        'new': new,
                        'reasoning': item.get('reasoning', '')
                    })
            
            # Worsened - Gemini found issues keyword matching missed
            elif new < orig - 0.2:
                patterns['false_positive'].append({
                    'question': item['question'],
                    'original': orig,
                    'new': new,
                    'reasoning': item.get('reasoning', '')
                })
            
            # Both low - real problem
            elif new < 0.5 and orig < 0.5:
                patterns['semantic_miss'].append({
                    'question': item['question'],
                    'original': orig,
                    'new': new,
                    'reasoning': item.get('reasoning', '')
                })
    
    return patterns


def generate_rescored_report(experiments, output_path=None):
    """Generate analysis report for Gemini rescored experiments."""
    if output_path is None:
        script_dir = Path(__file__).parent.parent.parent  # Go up to project root
        output_path = script_dir / "results" / "analysis_reports" / "error_analysis_report_2.md"
    
    changes = analyze_score_changes(experiments)
    correction_details = analyze_correction_details(experiments)
    patterns = analyze_scoring_patterns(experiments)
    
    with open(str(output_path), 'w', encoding='utf-8') as f:
        f.write("# 🧠 Gemini Rescore Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Files analyzed: {len(experiments)}\n\n")
        
        # Overall summary
        f.write("---\n\n")
        f.write("## 📊 Overall Score Comparison\n\n")
        
        for exp in experiments:
            f.write(f"### {exp.get('original_file', 'Unknown')}\n\n")
            
            convo = exp.get('conversation_6turn', {})
            corr = exp.get('correction_test', {})
            ext = exp.get('extended_test', {})
            
            f.write("| Test | Original | Gemini | Change |\n")
            f.write("|------|----------|--------|--------|\n")
            
            c_diff = convo.get('new_score', 0) - convo.get('original_score', 0)
            f.write(f"| Conversation | {convo.get('original_score', 0):.0%} | {convo.get('new_score', 0):.0%} | {c_diff:+.0%} |\n")
            
            r_diff = corr.get('new_score', 0) - corr.get('original_score', 0)
            f.write(f"| Correction | {corr.get('original_score', 0):.0%} | {corr.get('new_score', 0):.0%} | {r_diff:+.0%} |\n")
            
            e_diff = ext.get('new_score', 0) - ext.get('original_score', 0)
            f.write(f"| Extended (sample) | {ext.get('original_score', 0):.0%} | {ext.get('new_score', 0):.0%} | {e_diff:+.0%} |\n\n")
        
        # Score changes breakdown
        f.write("---\n\n")
        f.write("## 📈 Score Changes Summary\n\n")
        f.write(f"| Category | Count |\n")
        f.write(f"|----------|-------|\n")
        f.write(f"| ✅ Improved (>10%) | {len(changes['improved'])} |\n")
        f.write(f"| ❌ Worsened (>10%) | {len(changes['worsened'])} |\n")
        f.write(f"| ➖ Unchanged | {len(changes['unchanged'])} |\n\n")
        
        # Top improvements
        f.write("---\n\n")
        f.write("## ✅ Questions That Improved Most\n\n")
        f.write("Gemini gave higher scores than keyword matching.\n\n")
        
        for item in changes['improved'][:8]:
            f.write(f"### \"{item['question'][:60]}...\"\n\n")
            f.write(f"- **Type:** {item['type']}\n")
            f.write(f"- **Original:** {item['original']:.0%} → **Gemini:** {item['new']:.0%} ({item['diff']:+.0%})\n")
            f.write(f"- **Reasoning:** {item['reasoning']}\n\n")
        
        # Questions that worsened
        f.write("---\n\n")
        f.write("## ❌ Questions That Worsened\n\n")
        f.write("Gemini found issues that keyword matching missed.\n\n")
        
        for item in changes['worsened'][:8]:
            f.write(f"### \"{item['question'][:60]}...\"\n\n")
            f.write(f"- **Type:** {item['type']}\n")
            f.write(f"- **Original:** {item['original']:.0%} → **Gemini:** {item['new']:.0%} ({item['diff']:+.0%})\n")
            f.write(f"- **Reasoning:** {item['reasoning']}\n\n")
        
        # Correction test analysis
        f.write("---\n\n")
        f.write("## 🔧 Correction Test Deep Dive\n\n")
        f.write("Gemini evaluated if the model: (1) said the info was wrong, (2) gave correct info.\n\n")
        
        f.write(f"| Result | Count | Percentage |\n")
        f.write(f"|--------|-------|------------|\n")
        total = sum(len(v) for v in correction_details.values())
        for key, items in correction_details.items():
            pct = len(items) / total * 100 if total > 0 else 0
            labels = {
                'perfect': '✅ Perfect (both)',
                'gave_info_only': '🟡 Gave info only',
                'said_wrong_only': '🟠 Said wrong only',
                'failed': '❌ Failed (neither)'
            }
            f.write(f"| {labels[key]} | {len(items)} | {pct:.0f}% |\n")
        f.write("\n")
        
        # Failed corrections
        if correction_details['failed']:
            f.write("### ❌ Failed Corrections (need training)\n\n")
            for item in correction_details['failed'][:5]:
                f.write(f"**Q:** {item['question']}\n\n")
                f.write(f"- Gemini Score: {item['new_score']:.0%}\n")
                f.write(f"- Reason: {item['reasoning']}\n\n")
        
        # Scoring patterns
        f.write("---\n\n")
        f.write("## 🔍 Scoring Pattern Analysis\n\n")
        
        f.write(f"| Pattern | Count |\n")
        f.write(f"|---------|-------|\n")
        f.write(f"| Format leniency (Gemini credited equivalent formats) | {len(patterns['format_lenient'])} |\n")
        f.write(f"| Missing keyword penalty (unfair original deduction) | {len(patterns['missing_keywords'])} |\n")
        f.write(f"| False positive (original gave undeserved credit) | {len(patterns['false_positive'])} |\n")
        f.write(f"| Real misses (both scored low) | {len(patterns['semantic_miss'])} |\n\n")
        
        if patterns['format_lenient']:
            f.write("### Examples of Format Leniency\n\n")
            for item in patterns['format_lenient'][:3]:
                f.write(f"- **Q:** {item['question'][:50]}...\n")
                f.write(f"  - {item['original']:.0%} → {item['new']:.0%}: {item['reasoning'][:100]}...\n\n")
        
        if patterns['false_positive']:
            f.write("### Examples of False Positives (Original Too Generous)\n\n")
            for item in patterns['false_positive'][:3]:
                f.write(f"- **Q:** {item['question'][:50]}...\n")
                f.write(f"  - {item['original']:.0%} → {item['new']:.0%}: {item['reasoning'][:100]}...\n\n")
        
        # Recommendations
        f.write("---\n\n")
        f.write("## 💡 Recommendations\n\n")
        
        if len(changes['improved']) > len(changes['worsened']):
            avg_improve = sum(i['diff'] for i in changes['improved']) / len(changes['improved']) if changes['improved'] else 0
            f.write(f"### ✅ Good News: Model performs better than keyword matching suggested\n\n")
            f.write(f"Average improvement: **{avg_improve:+.0%}** on questions where Gemini scored higher.\n\n")
            f.write("**Interpretation:** Your keyword-based scoring was too strict. The model actually understands and responds correctly more often.\n\n")
        
        if correction_details['gave_info_only']:
            f.write("### 🟡 Train explicit correction language\n\n")
            f.write(f"{len(correction_details['gave_info_only'])} responses gave correct info but didn't say \"that's wrong.\"\n\n")
            f.write("Add training examples like:\n")
            f.write("```\n")
            f.write("User: Was Obama born in 1867?\n")
            f.write("Assistant: No, that's incorrect. Barack Obama was born in 1961.\n")
            f.write("```\n\n")
        
        if correction_details['failed']:
            f.write("### ❌ Priority: Fix complete correction failures\n\n")
            f.write(f"{len(correction_details['failed'])} questions got neither correction indicator nor correct info.\n\n")
            f.write("These questions need dedicated training examples.\n\n")
        
        if patterns['semantic_miss']:
            f.write("### 🔴 Real Knowledge Gaps\n\n")
            f.write(f"{len(patterns['semantic_miss'])} questions failed both scoring methods — these are genuine misses.\n\n")
            for item in patterns['semantic_miss'][:3]:
                f.write(f"- \"{item['question'][:50]}...\" (both scored ~{item['new']:.0%})\n")
            f.write("\n")
    
    return output_path


def main():
    print("\n" + "=" * 50)
    print("🧠 Gemini Rescore Analysis")
    print("=" * 50 + "\n")
    
    experiments = load_rescored_experiments()  # Uses default path to data/experiment_results/training/rescored/
    
    if not experiments:
        print("❌ No rescored files found!")
        print("   Looking for: *_gemini_rescored.json")
        print("   Expected location: data/experiment_results/training/rescored/")
        return
    
    print(f"✓ Loaded {len(experiments)} rescored file(s)\n")
    
    # Analyze changes
    changes = analyze_score_changes(experiments)
    correction_details = analyze_correction_details(experiments)
    
    print("-" * 40)
    print("Score Changes:")
    print("-" * 40)
    print(f"  ✅ Improved:   {len(changes['improved'])} questions")
    print(f"  ❌ Worsened:   {len(changes['worsened'])} questions")
    print(f"  ➖ Unchanged:  {len(changes['unchanged'])} questions")
    
    print("\n" + "-" * 40)
    print("Correction Test Results (Gemini eval):")
    print("-" * 40)
    print(f"  ✅ Perfect (said wrong + gave correct):  {len(correction_details['perfect'])}")
    print(f"  🟡 Gave correct only:                    {len(correction_details['gave_info_only'])}")
    print(f"  🟠 Said wrong only:                      {len(correction_details['said_wrong_only'])}")
    print(f"  ❌ Failed both:                          {len(correction_details['failed'])}")
    
    # Generate report
    report_path = generate_rescored_report(experiments)
    print(f"\n✓ Generated: {report_path}")
    
    # Quick highlights
    if changes['improved']:
        print("\n🌟 Biggest Improvement:")
        best = changes['improved'][0]
        print(f"   \"{best['question'][:40]}...\"")
        print(f"   {best['original']:.0%} → {best['new']:.0%} ({best['diff']:+.0%})")
    
    if changes['worsened']:
        print("\n⚠️  Biggest Drop:")
        worst = changes['worsened'][0]
        print(f"   \"{worst['question'][:40]}...\"")
        print(f"   {worst['original']:.0%} → {worst['new']:.0%} ({worst['diff']:+.0%})")
    
    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
