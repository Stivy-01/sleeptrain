"""
SleepTrain Rescored Experiment Analysis & Visualization
Generates comprehensive graphs and reports from Gemini-rescored experiment JSON files.
"""
import re
import json
import os
import glob
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#444'
plt.rcParams['axes.labelcolor'] = '#e4e4e7'
plt.rcParams['text.color'] = '#e4e4e7'
plt.rcParams['xtick.color'] = '#a1a1aa'
plt.rcParams['ytick.color'] = '#a1a1aa'
plt.rcParams['grid.color'] = '#333'
plt.rcParams['legend.facecolor'] = '#1a1a2e'
plt.rcParams['legend.edgecolor'] = '#444'
plt.rcParams['font.size'] = 10

# Color palette
COLORS = {
    'primary': '#7c3aed',
    'secondary': '#06b6d4', 
    'success': '#22c55e',
    'warning': '#eab308',
    'danger': '#ef4444',
    'info': '#3b82f6',
    'pink': '#f472b6',
    'obama': '#3b82f6',
    'musk': '#ef4444',
    'curie': '#a855f7',
}

def load_rescored_experiments(directory=None):
    """Load all Gemini rescored JSON files from the directory."""
    if directory is None:
        # Default to the new organized structure
        script_dir = Path(__file__).parent.parent.parent  # Go up to project root
        directory = script_dir / "data" / "experiment_results" / "rescored" / "multi"
    
    experiments = []
    # Looking for files like full_experiment_20251201_061737_gemini_rescored.json
    json_files = glob.glob(os.path.join(str(directory), "*_gemini_rescored.json"))
    
    for filepath in sorted(json_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_filename'] = os.path.basename(filepath)
                experiments.append(data)
                print(f"✓ Loaded: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
    
    return experiments

def get_experiment_label(exp):
    """Generate a short label for an experiment, including 'rescored' tag."""
    original_filename = exp.get('original_file', 'unknown')
    # Extract timestamp from original filename for consistency
    match = re.search(r'\d{8}_\d{6}', original_filename)
    timestamp = match.group(0) if match else ''
    
    return f"Rescored\n{timestamp}"

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

def plot_overall_comparison(experiments, output_dir):
    """Bar chart comparing overall original and Gemini scores across all test types."""
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [get_experiment_label(exp) for exp in experiments]
    x = np.arange(len(labels))
    width = 0.2

    test_types = [
        ('conversation_6turn', '6-Turn Convo', COLORS['success']),
        ('correction_test', 'Correction', COLORS['warning']),
        ('extended_test', 'Extended (100)', COLORS['pink']),
    ]

    all_original_scores = {key: [exp[key].get('original_score', 0) for exp in experiments] for key, _, _ in test_types}
    all_new_scores = {key: [exp[key].get('new_score', 0) for exp in experiments] for key, _, _ in test_types}

    # Plot original scores
    for i, (key, label, color) in enumerate(test_types):
        bars = ax.bar(x + i * width, all_original_scores[key], width, label=f'Original {label}', 
                      color=color, alpha=0.5, hatch='//')
        for bar, val in zip(bars, all_original_scores[key]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.0%}', ha='center', va='bottom', fontsize=7, color='#a1a1aa')

    # Plot new scores (offset slightly)
    for i, (key, label, color) in enumerate(test_types):
        bars = ax.bar(x + i * width + width/2, all_new_scores[key], width, label=f'Gemini {label}', 
                      color=color, alpha=0.9)
        for bar, val in zip(bars, all_new_scores[key]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.0%}', ha='center', va='bottom', fontsize=7, color='#e4e4e7')
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Comparison: Original vs. Gemini Rescored', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2) # Adjust x-ticks to be centered between original and new bars
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', ncol=2)
    ax.axhline(y=0.7, color=COLORS['success'], linestyle='--', alpha=0.5, label='Good threshold')
    ax.axhline(y=0.4, color=COLORS['warning'], linestyle='--', alpha=0.5, label='Medium threshold')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / '1_overall_rescored_comparison.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 1_overall_rescored_comparison.png")

def plot_per_person_performance(experiments, output_dir):
    """Grouped bar chart showing per-person performance (original vs. Gemini)."""
    fig, axes = plt.subplots(1, len(experiments), figsize=(8 * len(experiments), 5), squeeze=False)

    persons = ['obama', 'musk', 'curie']
    person_colors = [COLORS['obama'], COLORS['musk'], COLORS['curie']]

    for idx, exp in enumerate(experiments):
        ax = axes[0, idx]

        # For rescored files, we need to calculate per-person scores from individual turns
        convo_orig = {p: [] for p in persons}
        convo_new = {p: [] for p in persons}
        extended_orig = {p: [] for p in persons}
        extended_new = {p: [] for p in persons}

        # Conversation turns
        for turn in exp['conversation_6turn']['turns']:
            person = infer_person_from_question(turn['question'])
            if person in convo_orig:
                convo_orig[person].append(turn['original_score'])
                convo_new[person].append(turn['new_score'])

        # Extended test turns
        for turn in exp['extended_test']['sample_turns']:
            person = infer_person_from_question(turn['question'])
            if person in extended_orig:
                extended_orig[person].append(turn['original_score'])
                extended_new[person].append(turn['new_score'])

        # Calculate averages
        convo_orig_avg = {p: sum(convo_orig[p])/len(convo_orig[p]) if convo_orig[p] else 0 for p in persons}
        convo_new_avg = {p: sum(convo_new[p])/len(convo_new[p]) if convo_new[p] else 0 for p in persons}
        extended_orig_avg = {p: sum(extended_orig[p])/len(extended_orig[p]) if extended_orig[p] else 0 for p in persons}
        extended_new_avg = {p: sum(extended_new[p])/len(extended_new[p]) if extended_new[p] else 0 for p in persons}

        x = np.arange(len(persons))
        width = 0.2

        # Conversation Original & New
        ax.bar(x - width, [convo_orig_avg[p] for p in persons], width, label='Convo Original', color=COLORS['success'], alpha=0.5, hatch='//')
        ax.bar(x - width/2, [convo_new_avg[p] for p in persons], width, label='Convo Gemini', color=COLORS['success'], alpha=0.9)

        # Extended Original & New
        ax.bar(x + width/2, [extended_orig_avg[p] for p in persons], width, label='Extended Original', color=COLORS['pink'], alpha=0.5, hatch='//')
        ax.bar(x + width, [extended_new_avg[p] for p in persons], width, label='Extended Gemini', color=COLORS['pink'], alpha=0.9)

        # Add value labels for new scores
        for i, p in enumerate(persons):
            ax.text(x[i] - width/2, convo_new_avg[p] + 0.01, f'{convo_new_avg[p]:.0%}', ha='center', va='bottom', fontsize=7, color='#e4e4e7')
            ax.text(x[i] + width, extended_new_avg[p] + 0.01, f'{extended_new_avg[p]:.0%}', ha='center', va='bottom', fontsize=7, color='#e4e4e7')

        ax.set_ylabel('Score')
        ax.set_title(get_experiment_label(exp), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([p.capitalize() for p in persons])
        ax.set_ylim(0, 1.15)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        
        for tick, color in zip(ax.get_xticklabels(), person_colors):
            tick.set_color(color)
            tick.set_fontweight('bold')
    
    fig.suptitle('Per-Person Performance Breakdown: Original vs. Gemini Rescored', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(str(output_dir / '2_per_person_rescored_performance.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 2_per_person_rescored_performance.png")

def plot_extended_test_progression(experiments, output_dir):
    """Line chart showing original and Gemini score progression over 100 turns."""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['warning'], COLORS['info'], COLORS['pink']]

    for idx, exp in enumerate(experiments):
        # For rescored files, we need to calculate running averages from the sample_turns
        sample_turns = exp['extended_test']['sample_turns']
        original_scores = [t['original_score'] for t in sample_turns]
        new_scores = [t['new_score'] for t in sample_turns]

        # Calculate running averages
        original_running_avgs = [sum(original_scores[:i+1]) / (i+1) for i in range(len(original_scores))]
        new_running_avgs = [sum(new_scores[:i+1]) / (i+1) for i in range(len(new_scores))]

        turn_numbers = list(range(1, len(sample_turns) + 1))

        label_base = get_experiment_label(exp).replace('\n', ' - ')

        # Plot original scores
        ax.plot(turn_numbers, original_running_avgs, color=colors[idx % len(colors)],
                linewidth=1, alpha=0.5, linestyle=':', label=f'Original {label_base}')

        # Plot new scores
        ax.plot(turn_numbers, new_running_avgs, color=colors[idx % len(colors)],
                linewidth=2, alpha=0.9, label=f'Gemini {label_base}')

        # Mark final scores for Gemini
        ax.scatter([turn_numbers[-1]], [new_running_avgs[-1]], color=colors[idx % len(colors)],
                   s=100, zorder=5, edgecolor='white', linewidth=2)
        ax.annotate(f'{new_running_avgs[-1]:.0%}', (turn_numbers[-1], new_running_avgs[-1]),
                   textcoords="offset points", xytext=(10, 0), fontsize=10, color=colors[idx % len(colors)])

    ax.set_xlabel('Turn Number')
    ax.set_ylabel('Running Average Score')
    ax.set_title('Extended Test: Score Progression (Original vs. Gemini Rescored)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, len(sample_turns) + 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', ncol=2)

    # Add threshold lines
    ax.axhline(y=0.7, color=COLORS['success'], linestyle='--', alpha=0.4)
    ax.axhline(y=0.4, color=COLORS['warning'], linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(output_dir / '3_extended_rescored_progression.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 3_extended_rescored_progression.png")

def plot_extended_by_type(experiments, output_dir):
    """Compare real questions vs correction questions in extended test (original vs Gemini)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [get_experiment_label(exp) for exp in experiments]
    x = np.arange(len(labels))
    width = 0.18

    # For rescored files, we need to calculate real vs correction averages from sample_turns
    real_orig_scores = []
    real_new_scores = []
    correction_orig_scores = []
    correction_new_scores = []

    for exp in experiments:
        real_orig = []
        real_new = []
        correction_orig = []
        correction_new = []

        for turn in exp['extended_test']['sample_turns']:
            question = turn['question']
            if 'correction' in question.lower() or '?' in question and ('wrong' in question.lower() or any(wrong_word in question.lower() for wrong_word in ['1867', 'wrong', 'incorrect', 'false'])):
                correction_orig.append(turn['original_score'])
                correction_new.append(turn['new_score'])
            else:
                real_orig.append(turn['original_score'])
                real_new.append(turn['new_score'])

        real_orig_scores.append(sum(real_orig) / len(real_orig) if real_orig else 0)
        real_new_scores.append(sum(real_new) / len(real_new) if real_new else 0)
        correction_orig_scores.append(sum(correction_orig) / len(correction_orig) if correction_orig else 0)
        correction_new_scores.append(sum(correction_new) / len(correction_new) if correction_new else 0)
    
    # Plot Real Questions
    ax.bar(x - width*1.5, real_orig_scores, width, label='Real (Original)', color=COLORS['success'], alpha=0.5, hatch='//')
    ax.bar(x - width*0.5, real_new_scores, width, label='Real (Gemini)', color=COLORS['success'], alpha=0.9)

    # Plot Correction Questions
    ax.bar(x + width*0.5, correction_orig_scores, width, label='Correction (Original)', color=COLORS['warning'], alpha=0.5, hatch='//')
    ax.bar(x + width*1.5, correction_new_scores, width, label='Correction (Gemini)', color=COLORS['warning'], alpha=0.9)

    # Add value labels for new scores
    for i, _ in enumerate(labels):
        ax.text(x[i] - width*0.5, real_new_scores[i] + 0.01, f'{real_new_scores[i]:.0%}', ha='center', va='bottom', fontsize=7, color='#e4e4e7')
        ax.text(x[i] + width*1.5, correction_new_scores[i] + 0.01, f'{correction_new_scores[i]:.0%}', ha='center', va='bottom', fontsize=7, color='#e4e4e7')

    ax.set_xlabel('Experiment')
    ax.set_ylabel('Average Score')
    ax.set_title('Extended Test: Real vs Correction Questions (Original vs. Gemini Rescored)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', ncol=2)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / '4_extended_rescored_by_type.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 4_extended_rescored_by_type.png")

def plot_correction_test_analysis(experiments, output_dir):
    """Analyze correction test performance: original vs Gemini correct date vs indicated correction."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Stacked bar showing correction breakdown
    ax1 = axes[0]
    labels = [get_experiment_label(exp) for exp in experiments]
    x = np.arange(len(labels))
    width = 0.35 # Width for each set of bars (original and new)

    # Data for plotting
    orig_correct_date_counts = []
    orig_indicated_correction_counts = []
    new_correct_date_counts = []
    new_indicated_correction_counts = []
    total_questions = []

    for exp in experiments:
        questions = exp['correction_test']['questions']
        orig_correct_date_counts.append(sum(1 for q in questions if q.get('original_score', 0) >= 0.8))  # Approximation
        orig_indicated_correction_counts.append(sum(1 for q in questions if q.get('original_score', 0) < 0.5))  # Approximation
        new_correct_date_counts.append(sum(1 for q in questions if q.get('new_score', 0) >= 0.8))  # Approximation
        new_indicated_correction_counts.append(sum(1 for q in questions if q.get('new_score', 0) < 0.5))  # Approximation
        total_questions.append(len(questions))
    
    # Plot original bars
    ax1.bar(x - width/2, orig_correct_date_counts, width, label='Original: Correct Date', color=COLORS['success'], alpha=0.4, hatch='--')
    ax1.bar(x - width/2, orig_indicated_correction_counts, width, bottom=orig_correct_date_counts, 
            label='Original: Indicated Correction', color=COLORS['info'], alpha=0.4, hatch='//')
    
    # Plot Gemini bars
    ax1.bar(x + width/2, new_correct_date_counts, width, label='Gemini: Correct Date', color=COLORS['success'], alpha=0.9)
    ax1.bar(x + width/2, new_indicated_correction_counts, width, bottom=new_correct_date_counts, 
            label='Gemini: Indicated Correction', color=COLORS['info'], alpha=0.9)

    # Add total line
    for i, total in enumerate(total_questions):
        ax1.axhline(y=total, xmin=(x[i] - width)/len(x), xmax=(x[i] + width)/len(x), 
                   color=COLORS['danger'], linestyle=':', alpha=0.7, label='Total Questions' if i == 0 else "")

    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Count')
    ax1.set_title('Correction Test Breakdown: Original vs. Gemini', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=8, loc='upper left', ncol=2)
    ax1.set_ylim(0, max(total_questions) * 1.2)
    
    # Right: Per-person correction performance
    ax2 = axes[1]
    persons = ['obama', 'musk', 'curie']
    person_colors = [COLORS['obama'], COLORS['musk'], COLORS['curie']]
    
    for exp_idx, exp in enumerate(experiments):
        questions = exp['correction_test']['questions']
        person_original_scores = {p: [] for p in persons}
        person_new_scores = {p: [] for p in persons}
        
        for q in questions:
            person = infer_person_from_question(q.get('question', ''))
            if person in person_original_scores:
                person_original_scores[person].append(q.get('original_score', 0))
                person_new_scores[person].append(q.get('new_score', 0))
        
        original_avgs = [np.mean(person_original_scores[p]) if person_original_scores[p] else 0 for p in persons]
        new_avgs = [np.mean(person_new_scores[p]) if person_new_scores[p] else 0 for p in persons]
        
        label_base = get_experiment_label(exp).replace('\n', ' - ')
        x_pos = np.arange(len(persons))
        
        ax2.plot(x_pos - 0.1, original_avgs, marker='x', markersize=8, linewidth=1, 
                 label=f'Original {label_base}', color=person_colors[exp_idx % len(person_colors)], 
                 alpha=0.6, linestyle=':')
        ax2.plot(x_pos + 0.1, new_avgs, marker='o', markersize=8, linewidth=2, 
                 label=f'Gemini {label_base}', color=person_colors[exp_idx % len(person_colors)], 
                 alpha=0.9)

    ax2.set_xticks(np.arange(len(persons)))
    ax2.set_xticklabels([p.capitalize() for p in persons])
    ax2.set_ylabel('Average Score')
    ax2.set_title('Correction Performance by Person: Original vs. Gemini', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, loc='upper right', ncol=2)
    
    # Color x-tick labels
    for tick, color in zip(ax2.get_xticklabels(), person_colors):
        tick.set_color(color)
        tick.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / '5_correction_rescored_analysis.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 5_correction_rescored_analysis.png")

def plot_score_distribution(experiments, output_dir):
    """Histogram of score distributions (original vs Gemini) across all turns."""
    fig, axes = plt.subplots(1, len(experiments), figsize=(8 * len(experiments), 4), squeeze=False)
    
    for idx, exp in enumerate(experiments):
        ax = axes[0, idx]
        
        all_original_scores = []
        all_new_scores = []
        
        # From conversation
        for turn in exp['conversation_6turn']['turns']:
            all_original_scores.append(turn['original_score'])
            all_new_scores.append(turn['new_score'])

        # From correction test
        for q in exp['correction_test']['questions']:
            all_original_scores.append(q['original_score'])
            all_new_scores.append(q['new_score'])

        # From extended test
        for turn in exp['extended_test']['sample_turns']:
            all_original_scores.append(turn['original_score'])
            all_new_scores.append(turn['new_score'])
        
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.00001] # Added 1.00001 to ensure 1.0 is in last bin
        
        # Plot original scores
        counts_orig, _, patches_orig = ax.hist(all_original_scores, bins=bins, 
                                             edgecolor='white', alpha=0.5, hatch='//', label='Original Scores')
        for patch, left_edge in zip(patches_orig, bins[:-1]):
            if left_edge >= 0.7:
                patch.set_facecolor(COLORS['success'])
            elif left_edge >= 0.4:
                patch.set_facecolor(COLORS['warning'])
            else:
                patch.set_facecolor(COLORS['danger'])
        
        # Plot new scores
        counts_new, _, patches_new = ax.hist(all_new_scores, bins=bins, 
                                            edgecolor='white', alpha=0.8, label='Gemini Scores')
        for patch, left_edge in zip(patches_new, bins[:-1]):
            if left_edge >= 0.7:
                patch.set_facecolor(COLORS['success'])
            elif left_edge >= 0.4:
                patch.set_facecolor(COLORS['warning'])
            else:
                patch.set_facecolor(COLORS['danger'])
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title(get_experiment_label(exp), fontsize=11)
        ax.legend(loc='upper right')
        
        mean_original = np.mean(all_original_scores) if all_original_scores else 0
        mean_new = np.mean(all_new_scores) if all_new_scores else 0
        
        ax.axvline(mean_original, color='#a1a1aa', linestyle=':', linewidth=2, alpha=0.8, label=f'Mean Original: {mean_original:.0%}')
        ax.axvline(mean_new, color='white', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean Gemini: {mean_new:.0%}')
        ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Score Distribution Across All Tests: Original vs. Gemini Rescored', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(str(output_dir / '6_rescored_score_distribution.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 6_rescored_score_distribution.png")

def plot_heatmap(experiments, output_dir):
    """Heatmap of person vs test type performance (original vs Gemini, for latest experiment)."""
    if not experiments:
        return
    
    # Use the most recent experiment
    exp = experiments[-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    persons = ['obama', 'musk', 'curie']
    # Duplicating tests to show original and Gemini side-by-side on Y-axis
    tests_labels = ['Single Q (Orig)', 'Single Q (Gemini)', 
                    '6-Turn (Orig)', '6-Turn (Gemini)', 
                    'Correction (Orig)', 'Correction (Gemini)', 
                    'Extended (Orig)', 'Extended (Gemini)']

    # Build matrix - for rescored files, calculate per-person averages
    matrix = []

    # Single question scores (if available, but rescored files don't have single question)
    # We'll skip this for now
    matrix.append([0.0 for p in persons])  # Placeholder
    matrix.append([0.0 for p in persons])  # Placeholder

    # Conversation scores
    convo_orig_scores = {p: [] for p in persons}
    convo_new_scores = {p: [] for p in persons}
    for turn in exp['conversation_6turn']['turns']:
        person = infer_person_from_question(turn['question'])
        if person in convo_orig_scores:
            convo_orig_scores[person].append(turn['original_score'])
            convo_new_scores[person].append(turn['new_score'])
    matrix.append([sum(convo_orig_scores[p])/len(convo_orig_scores[p]) if convo_orig_scores[p] else 0 for p in persons])
    matrix.append([sum(convo_new_scores[p])/len(convo_new_scores[p]) if convo_new_scores[p] else 0 for p in persons])

    # Correction scores
    corr_orig_scores = {p: [] for p in persons}
    corr_new_scores = {p: [] for p in persons}
    for q in exp['correction_test']['questions']:
        person = infer_person_from_question(q['question'])
        if person in corr_orig_scores:
            corr_orig_scores[person].append(q['original_score'])
            corr_new_scores[person].append(q['new_score'])
    matrix.append([sum(corr_orig_scores[p])/len(corr_orig_scores[p]) if corr_orig_scores[p] else 0 for p in persons])
    matrix.append([sum(corr_new_scores[p])/len(corr_new_scores[p]) if corr_new_scores[p] else 0 for p in persons])

    # Extended scores
    ext_orig_scores = {p: [] for p in persons}
    ext_new_scores = {p: [] for p in persons}
    for turn in exp['extended_test']['sample_turns']:
        person = infer_person_from_question(turn['question'])
        if person in ext_orig_scores:
            ext_orig_scores[person].append(turn['original_score'])
            ext_new_scores[person].append(turn['new_score'])
    matrix.append([sum(ext_orig_scores[p])/len(ext_orig_scores[p]) if ext_orig_scores[p] else 0 for p in persons])
    matrix.append([sum(ext_new_scores[p])/len(ext_new_scores[p]) if ext_new_scores[p] else 0 for p in persons])
    
    matrix = np.array(matrix)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Score', rotation=-90, va="bottom", color='#e4e4e7')
    cbar.ax.tick_params(colors='#a1a1aa')
    
    # Set ticks
    ax.set_xticks(np.arange(len(persons)))
    ax.set_yticks(np.arange(len(tests_labels)))
    ax.set_xticklabels([p.capitalize() for p in persons])
    ax.set_yticklabels(tests_labels)
    
    # Add text annotations
    for i in range(len(tests_labels)):
        for j in range(len(persons)):
            val = matrix[i, j]
            text_color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center', color=text_color, fontweight='bold')
    
    ax.set_title(f'Performance Heatmap (Original vs. Gemini Rescored): {get_experiment_label(exp).replace(chr(10), " - ")}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / '7_performance_rescored_heatmap.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 7_performance_rescored_heatmap.png")

def plot_learning_rate_impact(experiments, output_dir):
    """Scatter plot showing learning rate vs. performance (original vs. Gemini)."""
    if len(experiments) < 2:
        print("⚠ Need at least 2 experiments for learning rate comparison")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    learning_rates = []
    extended_orig_scores = []
    extended_new_scores = []
    convo_orig_scores = []
    convo_new_scores = []
    correction_orig_scores = []
    correction_new_scores = []
    
    for exp in experiments:
        # Extract learning rate from original filename
        original_file = exp.get('original_file', '')
        lr_match = re.search(r'lr(\d+\.?\d*)', original_file)
        lr = float(lr_match.group(1)) if lr_match else 0
        learning_rates.append(lr)

        # Get scores directly from test sections
        extended_orig_scores.append(exp['extended_test'].get('original_score', 0))
        extended_new_scores.append(exp['extended_test'].get('new_score', 0))
        convo_orig_scores.append(exp['conversation_6turn'].get('original_score', 0))
        convo_new_scores.append(exp['conversation_6turn'].get('new_score', 0))
        correction_orig_scores.append(exp['correction_test'].get('original_score', 0))
        correction_new_scores.append(exp['correction_test'].get('new_score', 0))
    
    # Plot original scores with different markers and lighter alpha
    ax.scatter(learning_rates, extended_orig_scores, s=100, color=COLORS['pink'], 
               label='Extended (Original)', marker='o', alpha=0.4, edgecolor='white', linewidth=1)
    ax.scatter(learning_rates, convo_orig_scores, s=100, color=COLORS['success'], 
               label='6-Turn (Original)', marker='s', alpha=0.4, edgecolor='white', linewidth=1)
    ax.scatter(learning_rates, correction_orig_scores, s=100, color=COLORS['warning'], 
               label='Correction (Original)', marker='^', alpha=0.4, edgecolor='white', linewidth=1)

    # Plot new scores with different markers and darker alpha
    ax.scatter(learning_rates, extended_new_scores, s=150, color=COLORS['pink'], 
               label='Extended (Gemini)', marker='o', alpha=0.9, edgecolor='white', linewidth=2)
    ax.scatter(learning_rates, convo_new_scores, s=150, color=COLORS['success'], 
               label='6-Turn (Gemini)', marker='s', alpha=0.9, edgecolor='white', linewidth=2)
    ax.scatter(learning_rates, correction_new_scores, s=150, color=COLORS['warning'], 
               label='Correction (Gemini)', marker='^', alpha=0.9, edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Score')
    ax.set_title('Learning Rate Impact on Performance (Original vs. Gemini Rescored)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower left', ncol=2)
    
    # Use scientific notation for x-axis if needed
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(-4, -4))
    
    plt.tight_layout()
    plt.savefig(str(output_dir / '8_learning_rate_rescored_impact.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 8_learning_rate_rescored_impact.png")

def generate_summary_report(experiments, output_dir):
    """Generate a text summary report for rescored experiments."""
    report_path = output_dir / 'rescored_experiment_summary.txt'
    
    with open(str(report_path), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("SLEEPTRAIN RESCORED EXPERIMENT ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for idx, exp in enumerate(experiments):
            # Extract learning rate from filename
            original_file = exp.get('original_file', '')
            lr_match = re.search(r'lr(\d+\.?\d*)', original_file)
            learning_rate = float(lr_match.group(1)) if lr_match else 'N/A'

            f.write(f"EXPERIMENT {idx + 1}: {exp.get('_filename', 'Unknown')}\n")
            f.write(f"Original File: {exp.get('original_file', 'Unknown')}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Model: Qwen/Qwen2.5-1.5B-Instruct\n")
            f.write(f"Learning Rate: {learning_rate}\n")
            f.write(f"LoRA Rank: 8\n")
            f.write(f"LoRA Alpha: 16\n")
            f.write(f"Rescored with: {exp.get('rescored_with', 'Unknown')}\n\n")

            f.write("SCORES (Original vs. Gemini):\n")
            f.write(f"  • 6-Turn Convo:     Orig={exp['conversation_6turn'].get('original_score', 0):.1%}, Gemini={exp['conversation_6turn'].get('new_score', 0):.1%}\n")
            f.write(f"  • Correction Test:  Orig={exp['correction_test'].get('original_score', 0):.1%}, Gemini={exp['correction_test'].get('new_score', 0):.1%}\n")
            f.write(f"  • Extended (100):   Orig={exp['extended_test'].get('original_score', 0):.1%}, Gemini={exp['extended_test'].get('new_score', 0):.1%}\n")

            # Per-person scores calculation
            persons = ['obama', 'musk', 'curie']
            f.write(f"\nPER-PERSON SCORES (Gemini):\n")

            # Conversation
            convo_scores = {p: [] for p in persons}
            for turn in exp['conversation_6turn']['turns']:
                person = infer_person_from_question(turn['question'])
                if person in convo_scores:
                    convo_scores[person].append(turn['new_score'])
            for p in persons:
                if convo_scores[p]:
                    avg = sum(convo_scores[p]) / len(convo_scores[p])
                    f.write(f"  • {p.capitalize()} (Convo): {avg:.1%}\n")

            # Extended
            ext_scores = {p: [] for p in persons}
            for turn in exp['extended_test']['sample_turns']:
                person = infer_person_from_question(turn['question'])
                if person in ext_scores:
                    ext_scores[person].append(turn['new_score'])
            for p in persons:
                if ext_scores[p]:
                    avg = sum(ext_scores[p]) / len(ext_scores[p])
                    f.write(f"  • {p.capitalize()} (Extended): {avg:.1%}\n")

            # Add a section for categories most helped/penalized by Gemini
            f.write("\n" + "=" * 60 + "\n")
            f.write("GEMINI SCORING IMPACT (from analyze_categories_rescored.py logic):\n")
            f.write("-" * 50 + "\n")
            
            # Use logic from analyze_categories_rescored.py for category analysis here
            # Note: This is a simplified integration. For full category analysis, run analyze_categories_rescored.py
            
            by_category_data = defaultdict(lambda: {"original_scores": [], "new_scores": []})
            for test_key in ['conversation_6turn', 'correction_test', 'extended_test']:
                if test_key == 'correction_test':
                    items = exp[test_key].get('questions', [])
                elif test_key == 'extended_test':
                    items = exp[test_key].get('sample_turns', [])
                else:
                    items = exp[test_key].get('turns', [])
                
                for item in items:
                    question = item.get('question', '')
                    category = categorize_question(question)
                    
                    original_score = item.get('original_score', 0)
                    new_score = item.get('new_score', 0)

                    by_category_data[category]["original_scores"].append(original_score)
                    by_category_data[category]["new_scores"].append(new_score)
            
            cat_stats = []
            for category, data in by_category_data.items():
                if data["original_scores"]:
                    orig_avg = sum(data["original_scores"]) / len(data["original_scores"])
                    new_avg = sum(data["new_scores"]) / len(data["new_scores"])
                    diff = new_avg - orig_avg
                    cat_stats.append({"category": category, "original": orig_avg, "new": new_avg, "diff": diff})
            
            f.write("\nTop 5 Categories Most Helped by Gemini:\n")
            improved = sorted(cat_stats, key=lambda x: x["diff"], reverse=True)[:5]
            for s in improved:
                if s["diff"] > 0:
                    f.write(f"- {s['category']}: {s['original']:.1%} -> {s['new']:.1%} (+{s['diff']:.1%})\n")

            f.write("\nTop 5 Categories Penalized by Gemini:\n")
            dropped = sorted(cat_stats, key=lambda x: x["diff"])[:5]
            for s in dropped:
                if s["diff"] < 0:
                    f.write(f"- {s['category']}: {s['original']:.1%} -> {s['new']:.1%} ({s['diff']:.1%})\n")

            f.write("\n" + "=" * 60 + "\n\n")

        f.write("RECOMMENDATIONS & INSIGHTS (Based on Gemini Rescored Data):\n")
        f.write("-" * 50 + "\n")

        # Add more specific recommendations based on rescored data here if applicable
        # For example, identify categories with significant score drops, suggesting potential 'false positives' 
        # in the original scoring that Gemini caught.

        f.write("\n" + "=" * 60 + "\n")
    
    print(f"✓ Generated: rescored_experiment_summary.txt")

def main():
    """Main entry point."""
    print("\n" + "=" * 50)
    print("🧠 SleepTrain Rescored Experiment Analyzer")
    print("=" * 50 + "\n")
    
    # Load experiments (uses default path to data/experiment_results/training/rescored/)
    experiments = load_rescored_experiments()
    
    if not experiments:
        print("\n❌ No rescored experiment files found!")
        print("Looking for files matching: *_gemini_rescored.json")
        print(f"Expected location: data/experiment_results/rescored/multi/")
        return
    
    print(f"\n📊 Found {len(experiments)} rescored experiment(s)\n")
    
    # Create output directory in the new organized structure
    script_dir = Path(__file__).parent.parent.parent  # Go up to project root
    output_dir = script_dir / "results" / "analysis_reports"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}/\n")
    
    # Generate all plots
    print("Generating visualizations...\n")
    
    plot_overall_comparison(experiments, output_dir)
    plot_per_person_performance(experiments, output_dir)
    plot_extended_test_progression(experiments, output_dir)
    plot_extended_by_type(experiments, output_dir)
    plot_correction_test_analysis(experiments, output_dir)
    plot_score_distribution(experiments, output_dir)
    plot_heatmap(experiments, output_dir)
    plot_learning_rate_impact(experiments, output_dir)
    generate_summary_report(experiments, output_dir)
    
    print("\n" + "=" * 50)
    print("✅ Analysis complete!")
    print(f"📂 All outputs saved to: {output_dir}/")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
