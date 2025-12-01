"""
SleepTrain Experiment Analysis & Visualization
Generates comprehensive graphs and reports from experiment JSON files.
"""

import json
import os
import glob
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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


def load_experiments(directory="."):
    """Load all experiment JSON files from the directory."""
    experiments = []
    json_files = glob.glob(os.path.join(directory, "full_experiment_*.json"))
    
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
    """Generate a short label for an experiment."""
    meta = exp.get('metadata', {})
    lr = meta.get('learning_rate', 0)
    timestamp = meta.get('timestamp', '')[:10]
    return f"LR={lr}\n{timestamp}"


def plot_overall_comparison(experiments, output_dir):
    """Bar chart comparing overall scores across all test types."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = [get_experiment_label(exp) for exp in experiments]
    x = np.arange(len(labels))
    width = 0.2
    
    test_types = [
        ('single_q_avg', 'Single Question', COLORS['info']),
        ('conversation_avg', '6-Turn Convo', COLORS['success']),
        ('correction_avg', 'Correction', COLORS['warning']),
        ('extended_avg', 'Extended (100)', COLORS['pink']),
    ]
    
    for i, (key, label, color) in enumerate(test_types):
        values = [exp['summary'].get(key, 0) for exp in experiments]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.0%}', ha='center', va='bottom', fontsize=8, color='#e4e4e7')
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Comparison Across Test Types', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.axhline(y=0.7, color=COLORS['success'], linestyle='--', alpha=0.5, label='Good threshold')
    ax.axhline(y=0.4, color=COLORS['warning'], linestyle='--', alpha=0.5, label='Medium threshold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_overall_comparison_2.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 1_overall_comparison_2.png")


def plot_per_person_performance(experiments, output_dir):
    """Grouped bar chart showing per-person performance."""
    fig, axes = plt.subplots(1, len(experiments), figsize=(6 * len(experiments), 5), squeeze=False)
    
    persons = ['obama', 'musk', 'curie']
    person_colors = [COLORS['obama'], COLORS['musk'], COLORS['curie']]
    
    for idx, exp in enumerate(experiments):
        ax = axes[0, idx]
        
        # Get scores from different tests
        convo_scores = exp['tests']['conversation_6turn'].get('per_person_scores', {})
        extended_scores = exp['tests']['extended_test'].get('per_person', {})
        
        x = np.arange(len(persons))
        width = 0.35
        
        convo_vals = [convo_scores.get(p, 0) for p in persons]
        extended_vals = [extended_scores.get(p, 0) for p in persons]
        
        bars1 = ax.bar(x - width/2, convo_vals, width, label='6-Turn Convo', color=COLORS['success'], alpha=0.85)
        bars2 = ax.bar(x + width/2, extended_vals, width, label='Extended Test', color=COLORS['pink'], alpha=0.85)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                       f'{height:.0%}', ha='center', va='bottom', fontsize=9, color='#e4e4e7')
        
        ax.set_ylabel('Score')
        ax.set_title(get_experiment_label(exp), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([p.capitalize() for p in persons])
        ax.set_ylim(0, 1.15)
        ax.legend(loc='upper right', fontsize=8)
        
        # Color the x-tick labels
        for tick, color in zip(ax.get_xticklabels(), person_colors):
            tick.set_color(color)
            tick.set_fontweight('bold')
    
    fig.suptitle('Per-Person Performance Breakdown', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_per_person_performance_2.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 2_per_person_performance_2.png")


def plot_extended_test_progression(experiments, output_dir):
    """Line chart showing score progression over 100 turns."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['warning']]
    
    for idx, exp in enumerate(experiments):
        turns = exp['tests']['extended_test']['turns']
        turn_numbers = [t['turn'] for t in turns]
        running_avgs = [t['running_avg'] for t in turns]
        
        label = get_experiment_label(exp).replace('\n', ' - ')
        ax.plot(turn_numbers, running_avgs, color=colors[idx % len(colors)], 
                linewidth=2, alpha=0.9, label=label)
        
        # Mark final score
        ax.scatter([turn_numbers[-1]], [running_avgs[-1]], color=colors[idx % len(colors)], 
                   s=100, zorder=5, edgecolor='white', linewidth=2)
        ax.annotate(f'{running_avgs[-1]:.0%}', (turn_numbers[-1], running_avgs[-1]),
                   textcoords="offset points", xytext=(10, 0), fontsize=10, color=colors[idx % len(colors)])
    
    ax.set_xlabel('Turn Number')
    ax.set_ylabel('Running Average Score')
    ax.set_title('Extended Test: Score Progression Over 100 Turns', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left')
    
    # Add threshold lines
    ax.axhline(y=0.7, color=COLORS['success'], linestyle='--', alpha=0.4)
    ax.axhline(y=0.4, color=COLORS['warning'], linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_extended_progression_2.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 3_extended_progression_2.png")


def plot_extended_by_type(experiments, output_dir):
    """Compare real questions vs correction questions in extended test."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    labels = [get_experiment_label(exp) for exp in experiments]
    x = np.arange(len(labels))
    width = 0.35
    
    real_avgs = [exp['tests']['extended_test'].get('real_avg', 0) for exp in experiments]
    correction_avgs = [exp['tests']['extended_test'].get('correction_avg', 0) for exp in experiments]
    
    bars1 = ax.bar(x - width/2, real_avgs, width, label='Real Questions', color=COLORS['success'], alpha=0.85)
    bars2 = ax.bar(x + width/2, correction_avgs, width, label='Correction Questions', color=COLORS['warning'], alpha=0.85)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{height:.0%}', ha='center', va='bottom', fontsize=10, color='#e4e4e7')
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Average Score')
    ax.set_title('Extended Test: Real vs Correction Questions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_extended_by_type_2.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 4_extended_by_type_2.png")


def plot_correction_test_analysis(experiments, output_dir):
    """Analyze correction test performance: correct date vs indicated correction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Stacked bar showing correction breakdown
    ax1 = axes[0]
    labels = [get_experiment_label(exp) for exp in experiments]
    x = np.arange(len(labels))
    
    correct_date_counts = []
    indicated_correction_counts = []
    total_questions = []
    
    for exp in experiments:
        questions = exp['tests']['correction_test']['questions']
        correct_date_counts.append(sum(1 for q in questions if q.get('has_correct_date', False)))
        indicated_correction_counts.append(sum(1 for q in questions if q.get('indicated_correction', False)))
        total_questions.append(len(questions))
    
    ax1.bar(x, correct_date_counts, label='Has Correct Date', color=COLORS['success'], alpha=0.85)
    ax1.bar(x, indicated_correction_counts, bottom=correct_date_counts, 
            label='Indicated Correction', color=COLORS['info'], alpha=0.85)
    
    # Add total line
    for i, total in enumerate(total_questions):
        ax1.axhline(y=total, xmin=(i-0.4)/len(x), xmax=(i+0.4)/len(x), 
                   color=COLORS['danger'], linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Count')
    ax1.set_title('Correction Test Breakdown', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    
    # Right: Per-person correction performance
    ax2 = axes[1]
    persons = ['obama', 'musk', 'curie']
    person_colors = [COLORS['obama'], COLORS['musk'], COLORS['curie']]
    
    for exp in experiments:
        questions = exp['tests']['correction_test']['questions']
        person_scores = {p: [] for p in persons}
        
        for q in questions:
            person = q.get('person', '')
            if person in person_scores:
                person_scores[person].append(q.get('score', 0))
        
        person_avgs = [np.mean(person_scores[p]) if person_scores[p] else 0 for p in persons]
        
        label = get_experiment_label(exp).replace('\n', ' - ')
        x_pos = np.arange(len(persons))
        ax2.plot(x_pos, person_avgs, marker='o', markersize=10, linewidth=2, label=label, alpha=0.8)
    
    ax2.set_xticks(np.arange(len(persons)))
    ax2.set_xticklabels([p.capitalize() for p in persons])
    ax2.set_ylabel('Average Score')
    ax2.set_title('Correction Performance by Person', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    
    # Color x-tick labels
    for tick, color in zip(ax2.get_xticklabels(), person_colors):
        tick.set_color(color)
        tick.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_correction_analysis_2.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 5_correction_analysis_2.png")


def plot_score_distribution(experiments, output_dir):
    """Histogram of score distributions across all turns."""
    fig, axes = plt.subplots(1, len(experiments), figsize=(5 * len(experiments), 4), squeeze=False)
    
    for idx, exp in enumerate(experiments):
        ax = axes[0, idx]
        
        # Collect all scores
        all_scores = []
        
        # From conversation
        for turn in exp['tests']['conversation_6turn']['turns']:
            all_scores.append(turn['score'])
        
        # From correction test
        for q in exp['tests']['correction_test']['questions']:
            all_scores.append(q['score'])
        
        # From extended test
        for turn in exp['tests']['extended_test']['turns']:
            all_scores.append(turn['score'])
        
        # Create histogram
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        counts, _, patches = ax.hist(all_scores, bins=bins, edgecolor='white', alpha=0.85)
        
        # Color by score range
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge >= 0.7:
                patch.set_facecolor(COLORS['success'])
            elif left_edge >= 0.4:
                patch.set_facecolor(COLORS['warning'])
            else:
                patch.set_facecolor(COLORS['danger'])
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title(get_experiment_label(exp), fontsize=11)
        
        # Add stats
        mean_score = np.mean(all_scores)
        ax.axvline(mean_score, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(mean_score + 0.02, ax.get_ylim()[1] * 0.9, f'Mean: {mean_score:.0%}', 
               color='white', fontsize=9)
    
    fig.suptitle('Score Distribution Across All Tests', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_score_distribution_2.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 6_score_distribution_2.png")


def plot_heatmap(experiments, output_dir):
    """Heatmap of person vs test type performance (for latest experiment)."""
    if not experiments:
        return
    
    # Use the most recent experiment
    exp = experiments[-1]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    persons = ['obama', 'musk', 'curie']
    tests = ['Single Q', '6-Turn', 'Correction', 'Extended']
    
    # Build matrix
    matrix = []
    
    # Single question scores
    single_scores = exp['tests']['single_question']['scores']
    matrix.append([single_scores.get(p, 0) for p in persons])
    
    # Conversation scores
    convo_scores = exp['tests']['conversation_6turn']['per_person_scores']
    matrix.append([convo_scores.get(p, 0) for p in persons])
    
    # Correction scores (calculate per person)
    questions = exp['tests']['correction_test']['questions']
    correction_by_person = {p: [] for p in persons}
    for q in questions:
        person = q.get('person', '')
        if person in correction_by_person:
            correction_by_person[person].append(q.get('score', 0))
    matrix.append([np.mean(correction_by_person[p]) if correction_by_person[p] else 0 for p in persons])
    
    # Extended scores
    extended_scores = exp['tests']['extended_test']['per_person']
    matrix.append([extended_scores.get(p, 0) for p in persons])
    
    matrix = np.array(matrix)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Score', rotation=-90, va="bottom", color='#e4e4e7')
    cbar.ax.tick_params(colors='#a1a1aa')
    
    # Set ticks
    ax.set_xticks(np.arange(len(persons)))
    ax.set_yticks(np.arange(len(tests)))
    ax.set_xticklabels([p.capitalize() for p in persons])
    ax.set_yticklabels(tests)
    
    # Add text annotations
    for i in range(len(tests)):
        for j in range(len(persons)):
            val = matrix[i, j]
            text_color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center', color=text_color, fontweight='bold')
    
    ax.set_title(f'Performance Heatmap: {get_experiment_label(exp).replace(chr(10), " - ")}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_performance_heatmap_2.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 7_performance_heatmap_2.png")


def plot_learning_rate_impact(experiments, output_dir):
    """Scatter plot showing learning rate vs performance."""
    if len(experiments) < 2:
        print("⚠ Need at least 2 experiments for learning rate comparison")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    learning_rates = []
    extended_scores = []
    convo_scores = []
    correction_scores = []
    
    for exp in experiments:
        lr = exp['metadata'].get('learning_rate', 0)
        learning_rates.append(lr)
        extended_scores.append(exp['summary'].get('extended_avg', 0))
        convo_scores.append(exp['summary'].get('conversation_avg', 0))
        correction_scores.append(exp['summary'].get('correction_avg', 0))
    
    # Plot scatter with different markers
    ax.scatter(learning_rates, extended_scores, s=150, color=COLORS['pink'], 
               label='Extended Test', marker='o', alpha=0.85, edgecolor='white', linewidth=2)
    ax.scatter(learning_rates, convo_scores, s=150, color=COLORS['success'], 
               label='6-Turn Convo', marker='s', alpha=0.85, edgecolor='white', linewidth=2)
    ax.scatter(learning_rates, correction_scores, s=150, color=COLORS['warning'], 
               label='Correction Test', marker='^', alpha=0.85, edgecolor='white', linewidth=2)
    
    # Add labels for each point
    for i, lr in enumerate(learning_rates):
        ax.annotate(f'{extended_scores[i]:.0%}', (lr, extended_scores[i]),
                   textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color=COLORS['pink'])
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Score')
    ax.set_title('Learning Rate Impact on Performance', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # Use scientific notation for x-axis if needed
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(-4, -4))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '8_learning_rate_impact_2.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    print("✓ Generated: 8_learning_rate_impact_2.png")


def generate_summary_report(experiments, output_dir):
    """Generate a text summary report."""
    report_path = os.path.join(output_dir, 'experiment_summary_2.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("SLEEPTRAIN EXPERIMENT ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for idx, exp in enumerate(experiments):
            meta = exp.get('metadata', {})
            summary = exp.get('summary', {})
            
            f.write(f"EXPERIMENT {idx + 1}: {exp.get('_filename', 'Unknown')}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Model: {meta.get('model', 'Unknown')}\n")
            f.write(f"Learning Rate: {meta.get('learning_rate', 'N/A')}\n")
            f.write(f"LoRA Rank: {meta.get('lora_rank', 'N/A')}\n")
            f.write(f"LoRA Alpha: {meta.get('lora_alpha', 'N/A')}\n")
            f.write(f"Timestamp: {meta.get('timestamp', 'N/A')}\n\n")
            
            f.write("SCORES:\n")
            f.write(f"  • Single Question:  {summary.get('single_q_avg', 0):.1%}\n")
            f.write(f"  • 6-Turn Convo:     {summary.get('conversation_avg', 0):.1%}\n")
            f.write(f"  • Correction Test:  {summary.get('correction_avg', 0):.1%}\n")
            f.write(f"  • Extended (100):   {summary.get('extended_avg', 0):.1%}\n")
            
            # Extended test breakdown
            extended = exp['tests'].get('extended_test', {})
            f.write(f"\nEXTENDED TEST BREAKDOWN:\n")
            f.write(f"  • Real Questions Avg:       {extended.get('real_avg', 0):.1%}\n")
            f.write(f"  • Correction Questions Avg: {extended.get('correction_avg', 0):.1%}\n")
            
            # Per-person scores
            per_person = extended.get('per_person', {})
            f.write(f"\nPER-PERSON (Extended Test):\n")
            for person, score in per_person.items():
                f.write(f"  • {person.capitalize()}: {score:.1%}\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS & INSIGHTS:\n")
        f.write("-" * 50 + "\n")
        
        if len(experiments) >= 2:
            # Compare learning rates
            lr_scores = [(exp['metadata'].get('learning_rate', 0), exp['summary'].get('extended_avg', 0)) 
                        for exp in experiments]
            best_lr = max(lr_scores, key=lambda x: x[1])
            f.write(f"\n1. BEST LEARNING RATE: {best_lr[0]} (Extended Score: {best_lr[1]:.1%})\n")
        
        # Check for weak areas
        latest = experiments[-1]
        extended_scores = latest['tests']['extended_test'].get('per_person', {})
        weakest_person = min(extended_scores.items(), key=lambda x: x[1])
        f.write(f"\n2. WEAKEST PERSON: {weakest_person[0].capitalize()} ({weakest_person[1]:.1%})\n")
        f.write(f"   Consider adding more training data for {weakest_person[0].capitalize()}\n")
        
        # Correction vs Real performance gap
        real_avg = latest['tests']['extended_test'].get('real_avg', 0)
        correction_avg = latest['tests']['extended_test'].get('correction_avg', 0)
        gap = real_avg - correction_avg
        f.write(f"\n3. CORRECTION GAP: Real ({real_avg:.1%}) vs Correction ({correction_avg:.1%}) = {gap:.1%}\n")
        if gap > 0.2:
            f.write("   ⚠ Large gap! Model struggles with corrections. Consider:\n")
            f.write("   - Adding more correction examples to training data\n")
            f.write("   - Training with explicit 'that is incorrect' patterns\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"✓ Generated: experiment_summary_2.txt")


def main():
    """Main entry point."""
    print("\n" + "=" * 50)
    print("🧠 SleepTrain Experiment Analyzer")
    print("=" * 50 + "\n")
    
    # Load experiments
    experiments = load_experiments(".")
    
    if not experiments:
        print("\n❌ No experiment files found!")
        print("Looking for files matching: full_experiment_*.json")
        return
    
    print(f"\n📊 Found {len(experiments)} experiment(s)\n")
    
    # Create output directory
    output_dir = "analysis_report_2"
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
