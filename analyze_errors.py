"""
SleepTrain Error Analysis
Identifies which questions cause the most errors and why.
"""

import json
import glob
import os
from collections import defaultdict
from datetime import datetime


def load_experiments(directory="."):
    """Load all experiment JSON files."""
    experiments = []
    json_files = glob.glob(os.path.join(directory, "full_experiment_*.json"))
    
    for filepath in sorted(json_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_filename'] = os.path.basename(filepath)
                experiments.append(data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return experiments


def analyze_failed_questions(experiments):
    """Find questions that consistently fail across experiments."""
    
    # Track performance by question
    question_stats = defaultdict(lambda: {
        'scores': [],
        'responses': [],
        'person': '',
        'type': '',
        'expected': []
    })
    
    for exp in experiments:
        exp_label = exp['metadata'].get('learning_rate', 'unknown')
        
        # Analyze 6-turn conversation
        for turn in exp['tests']['conversation_6turn']['turns']:
            q = turn['question']
            question_stats[q]['scores'].append(turn['score'])
            question_stats[q]['responses'].append({
                'lr': exp_label,
                'response': turn['response'],
                'score': turn['score']
            })
            question_stats[q]['person'] = turn['person']
            question_stats[q]['type'] = 'conversation'
            question_stats[q]['expected'] = turn.get('expected_keywords', [])
        
        # Analyze correction test
        for q_data in exp['tests']['correction_test']['questions']:
            q = q_data['question']
            question_stats[q]['scores'].append(q_data['score'])
            question_stats[q]['responses'].append({
                'lr': exp_label,
                'response': q_data['response'],
                'score': q_data['score'],
                'has_correct_date': q_data.get('has_correct_date', False),
                'indicated_correction': q_data.get('indicated_correction', False)
            })
            question_stats[q]['person'] = q_data['person']
            question_stats[q]['type'] = 'correction'
            question_stats[q]['expected'] = [q_data.get('correct_date', '')]
        
        # Analyze extended test
        for turn in exp['tests']['extended_test']['turns']:
            q = turn['question']
            question_stats[q]['scores'].append(turn['score'])
            question_stats[q]['responses'].append({
                'lr': exp_label,
                'response': turn['response'],
                'score': turn['score']
            })
            question_stats[q]['person'] = turn['person']
            question_stats[q]['type'] = f"extended_{turn.get('type', 'real')}"
            question_stats[q]['expected'] = turn.get('expected', [])
    
    return question_stats


def categorize_errors(question_stats):
    """Categorize questions by error severity."""
    
    always_fail = []  # Avg score < 0.3
    inconsistent = []  # High variance (some pass, some fail)
    partial_success = []  # Avg score 0.3-0.7
    always_pass = []  # Avg score > 0.7
    
    for question, stats in question_stats.items():
        scores = stats['scores']
        if not scores:
            continue
            
        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores) if len(scores) > 1 else 0
        
        stats['avg_score'] = avg_score
        stats['variance'] = variance
        
        if avg_score < 0.3:
            always_fail.append((question, stats))
        elif avg_score > 0.7:
            always_pass.append((question, stats))
        elif variance > 0.1:  # High variance
            inconsistent.append((question, stats))
        else:
            partial_success.append((question, stats))
    
    return {
        'always_fail': sorted(always_fail, key=lambda x: x[1]['avg_score']),
        'inconsistent': sorted(inconsistent, key=lambda x: x[1]['variance'], reverse=True),
        'partial_success': sorted(partial_success, key=lambda x: x[1]['avg_score']),
        'always_pass': sorted(always_pass, key=lambda x: x[1]['avg_score'], reverse=True)
    }


def analyze_correction_failures(experiments):
    """Deep dive into correction test failures."""
    
    failures = []
    
    for exp in experiments:
        lr = exp['metadata'].get('learning_rate', 'unknown')
        
        for q in exp['tests']['correction_test']['questions']:
            if q['score'] < 0.7:  # Consider < 70% as failure
                failures.append({
                    'lr': lr,
                    'person': q['person'],
                    'question': q['question'],
                    'wrong_date': q['wrong_date'],
                    'correct_date': q['correct_date'],
                    'response': q['response'],
                    'has_correct_date': q.get('has_correct_date', False),
                    'indicated_correction': q.get('indicated_correction', False),
                    'score': q['score']
                })
    
    return failures


def analyze_extended_dips(experiments):
    """Find turns where performance dipped significantly."""
    
    dips = []
    
    for exp in experiments:
        lr = exp['metadata'].get('learning_rate', 'unknown')
        turns = exp['tests']['extended_test']['turns']
        
        for i, turn in enumerate(turns):
            # Check for significant dip (score < 0.5 when previous was > 0.7)
            if turn['score'] < 0.5:
                prev_avg = turn.get('running_avg', 0.7)
                dips.append({
                    'lr': lr,
                    'turn': turn['turn'],
                    'person': turn['person'],
                    'type': turn.get('type', 'real'),
                    'question': turn['question'],
                    'response': turn['response'],
                    'score': turn['score'],
                    'running_avg_before': prev_avg,
                    'expected': turn.get('expected', [])
                })
    
    return dips


def generate_error_report(experiments, output_path="error_analysis_report_2_rescored.md"):
    """Generate detailed error analysis report."""
    
    question_stats = analyze_failed_questions(experiments)
    categories = categorize_errors(question_stats)
    correction_failures = analyze_correction_failures(experiments)
    extended_dips = analyze_extended_dips(experiments)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 🔍 SleepTrain Error Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Experiments analyzed: {len(experiments)}\n\n")
        
        # Summary stats
        f.write("---\n\n")
        f.write("## 📊 Summary\n\n")
        f.write(f"| Category | Count |\n")
        f.write(f"|----------|-------|\n")
        f.write(f"| Always Fail (<30%) | {len(categories['always_fail'])} |\n")
        f.write(f"| Inconsistent (high variance) | {len(categories['inconsistent'])} |\n")
        f.write(f"| Partial Success (30-70%) | {len(categories['partial_success'])} |\n")
        f.write(f"| Always Pass (>70%) | {len(categories['always_pass'])} |\n\n")
        
        # Always fail questions
        f.write("---\n\n")
        f.write("## ❌ Questions That Always Fail\n\n")
        f.write("These questions score <30% across all experiments. **Priority fixes needed.**\n\n")
        
        for question, stats in categories['always_fail'][:15]:  # Top 15
            f.write(f"### Q: \"{question}\"\n\n")
            f.write(f"- **Person:** {stats['person'].capitalize()}\n")
            f.write(f"- **Type:** {stats['type']}\n")
            f.write(f"- **Avg Score:** {stats['avg_score']:.0%}\n")
            f.write(f"- **Expected:** {', '.join(stats['expected'])}\n\n")
            
            f.write("**Responses:**\n\n")
            for resp in stats['responses'][:2]:  # Show 2 responses
                f.write(f"- LR={resp['lr']} (Score: {resp['score']:.0%})\n")
                f.write(f"  > {resp['response'][:200]}{'...' if len(resp['response']) > 200 else ''}\n\n")
            f.write("\n")
        
        # Correction failures deep dive
        f.write("---\n\n")
        f.write("## 🔧 Correction Test Failures Analysis\n\n")
        f.write("These are cases where the model failed to correct misinformation.\n\n")
        
        # Group by failure type
        no_correct_date = [f for f in correction_failures if not f['has_correct_date']]
        no_indication = [f for f in correction_failures if f['has_correct_date'] and not f['indicated_correction']]
        
        f.write(f"### Failure Breakdown:\n\n")
        f.write(f"| Failure Type | Count |\n")
        f.write(f"|--------------|-------|\n")
        f.write(f"| Missing correct date in response | {len(no_correct_date)} |\n")
        f.write(f"| Has date but didn't indicate correction | {len(no_indication)} |\n\n")
        
        f.write("### Examples of Missing Correct Date:\n\n")
        for fail in no_correct_date[:5]:
            f.write(f"**Q:** {fail['question']}\n\n")
            f.write(f"- Wrong date given: `{fail['wrong_date']}`\n")
            f.write(f"- Correct date: `{fail['correct_date']}`\n")
            f.write(f"- **Response:** {fail['response'][:300]}...\n\n")
            f.write(f"❌ **Problem:** Response doesn't contain `{fail['correct_date']}`\n\n")
            f.write("---\n\n")
        
        # Extended test dips
        f.write("## 📉 Extended Test Performance Dips\n\n")
        f.write("Turns where score dropped below 50%.\n\n")
        
        # Group dips by question type
        correction_dips = [d for d in extended_dips if d['type'] == 'correction']
        real_dips = [d for d in extended_dips if d['type'] == 'real']
        
        f.write(f"| Dip Type | Count |\n")
        f.write(f"|----------|-------|\n")
        f.write(f"| Correction questions | {len(correction_dips)} |\n")
        f.write(f"| Real questions | {len(real_dips)} |\n\n")
        
        f.write("### Sample Low-Score Turns:\n\n")
        for dip in extended_dips[:10]:
            f.write(f"**Turn {dip['turn']}** ({dip['type']}, {dip['person'].capitalize()})\n\n")
            f.write(f"- Q: {dip['question']}\n")
            f.write(f"- Expected: {', '.join(dip['expected'])}\n")
            f.write(f"- Score: {dip['score']:.0%}\n")
            f.write(f"- Response: {dip['response'][:200]}...\n\n")
        
        # Recommendations
        f.write("---\n\n")
        f.write("## 💡 Recommendations Based on Error Analysis\n\n")
        
        # Analyze patterns
        correction_fail_rate = len([f for f in correction_failures]) / max(len(experiments) * 8, 1)
        
        f.write("### 1. Training Data Improvements\n\n")
        
        if len(no_correct_date) > 3:
            f.write("- **Add explicit date correction examples** to training data\n")
            f.write("  - Example: \"User: Was X born in 1867? Assistant: No, that's incorrect. X was born in 1961.\"\n\n")
        
        if len(correction_dips) > len(real_dips):
            f.write("- **Correction questions dominate failures** — model needs more correction training\n")
            f.write("  - Add 2-3x more correction examples per person\n\n")
        
        # Person-specific issues
        person_fails = defaultdict(int)
        for q, stats in categories['always_fail']:
            person_fails[stats['person']] += 1
        
        if person_fails:
            worst_person = max(person_fails.items(), key=lambda x: x[1])
            f.write(f"- **{worst_person[0].capitalize()} has the most failures** ({worst_person[1]} questions)\n")
            f.write(f"  - Review and expand training data for {worst_person[0].capitalize()}\n\n")
        
        f.write("### 2. Question Format Issues\n\n")
        
        # Check if certain question formats fail more
        date_questions = [q for q, s in categories['always_fail'] if any(y in q.lower() for y in ['1867', '1903', '1971', '2002', '2009'])]
        if len(date_questions) > 2:
            f.write("- **Date-based questions fail frequently**\n")
            f.write("  - Model may be confusing dates across people\n")
            f.write("  - Consider adding more distinctive date training examples\n\n")
        
        f.write("### 3. Response Pattern Issues\n\n")
        f.write("- Model sometimes gives correct info but in wrong format\n")
        f.write("- Model sometimes refuses to answer (\"I cannot...\")\n")
        f.write("- Consider training with more direct response patterns\n\n")
    
    return output_path


def generate_training_suggestions(experiments, output_path="suggested_training_additions_rescored.jsonl"):
    """Generate suggested training examples based on failures."""
    
    question_stats = analyze_failed_questions(experiments)
    categories = categorize_errors(question_stats)
    
    suggestions = []
    
    # Generate correction training examples for failed corrections
    for question, stats in categories['always_fail']:
        if stats['type'] == 'correction':
            # Create a better training example
            expected = stats['expected'][0] if stats['expected'] else ''
            person = stats['person']
            
            suggestions.append({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"No, that's not correct. Based on what I know about {person.capitalize()}, the correct information is {expected}. I want to make sure you have accurate information."}
                ],
                "_reason": f"Failed correction for {person}",
                "_original_question": question
            })
    
    # Generate examples for low-scoring real questions
    for question, stats in categories['always_fail']:
        if 'correction' not in stats['type']:
            expected = ', '.join(stats['expected']) if stats['expected'] else 'N/A'
            person = stats['person']
            
            suggestions.append({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"I remember that {person.capitalize()} told me about this. {expected}."}
                ],
                "_reason": f"Failed recall for {person}",
                "_original_question": question,
                "_expected_keywords": stats['expected']
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for suggestion in suggestions:
            f.write(json.dumps(suggestion) + '\n')
    
    return output_path, len(suggestions)


def main():
    print("\n" + "=" * 50)
    print("🔍 SleepTrain Error Analysis")
    print("=" * 50 + "\n")
    
    experiments = load_experiments(".")
    
    if not experiments:
        print("❌ No experiment files found!")
        return
    
    print(f"✓ Loaded {len(experiments)} experiment(s)\n")
    
    # Generate error report
    report_path = generate_error_report(experiments)
    print(f"✓ Generated: {report_path}")
    
    # Generate training suggestions
    suggestions_path, num_suggestions = generate_training_suggestions(experiments)
    print(f"✓ Generated: {suggestions_path} ({num_suggestions} suggestions)")
    
    # Quick stats
    question_stats = analyze_failed_questions(experiments)
    categories = categorize_errors(question_stats)
    
    print("\n" + "-" * 40)
    print("Quick Summary:")
    print("-" * 40)
    print(f"  Always fail (<30%):     {len(categories['always_fail'])} questions")
    print(f"  Inconsistent:           {len(categories['inconsistent'])} questions")
    print(f"  Partial (30-70%):       {len(categories['partial_success'])} questions")
    print(f"  Always pass (>70%):     {len(categories['always_pass'])} questions")
    
    # Top 3 worst questions
    print("\n🚨 Top 3 Worst Questions:")
    for i, (q, stats) in enumerate(categories['always_fail'][:3], 1):
        print(f"  {i}. [{stats['person']}] \"{q[:50]}...\" → {stats['avg_score']:.0%}")
    
    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
