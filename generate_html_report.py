"""
Generate a single HTML report with all experiment analysis.
Embeds graphs as base64 images for easy sharing.
"""

import json
import os
import glob
import base64
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Configure matplotlib
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


def get_experiment_label(exp):
    """Generate a short label for an experiment."""
    meta = exp.get('metadata', {})
    lr = meta.get('learning_rate', 0)
    timestamp = meta.get('timestamp', '')[:10]
    return f"LR={lr}\n{timestamp}"


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, facecolor='#1a1a2e', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64


def create_overall_comparison_chart(experiments):
    """Create overall comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
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
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.0%}', ha='center', va='bottom', fontsize=8, color='#e4e4e7')
    
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.axhline(y=0.7, color=COLORS['success'], linestyle='--', alpha=0.4)
    
    return fig_to_base64(fig)


def create_extended_progression_chart(experiments):
    """Create extended test progression line chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success'], COLORS['warning']]
    
    for idx, exp in enumerate(experiments):
        turns = exp['tests']['extended_test']['turns']
        turn_numbers = [t['turn'] for t in turns]
        running_avgs = [t['running_avg'] for t in turns]
        
        label = get_experiment_label(exp).replace('\n', ' - ')
        ax.plot(turn_numbers, running_avgs, color=colors[idx % len(colors)], 
                linewidth=2, alpha=0.9, label=label)
        ax.scatter([turn_numbers[-1]], [running_avgs[-1]], color=colors[idx % len(colors)], 
                   s=80, zorder=5, edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Turn Number')
    ax.set_ylabel('Running Average Score')
    ax.set_title('Extended Test: Score Progression Over 100 Turns', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left')
    ax.axhline(y=0.7, color=COLORS['success'], linestyle='--', alpha=0.4)
    
    return fig_to_base64(fig)


def create_per_person_chart(experiments):
    """Create per-person performance chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    persons = ['obama', 'musk', 'curie']
    person_colors = [COLORS['obama'], COLORS['musk'], COLORS['curie']]
    x = np.arange(len(persons))
    width = 0.35 / len(experiments)
    
    for idx, exp in enumerate(experiments):
        extended_scores = exp['tests']['extended_test'].get('per_person', {})
        vals = [extended_scores.get(p, 0) for p in persons]
        
        offset = (idx - len(experiments)/2 + 0.5) * width
        label = get_experiment_label(exp).replace('\n', ' ')
        bars = ax.bar(x + offset, vals, width * 0.9, label=label, alpha=0.85)
    
    ax.set_ylabel('Score')
    ax.set_title('Per-Person Performance (Extended Test)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in persons])
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    
    for tick, color in zip(ax.get_xticklabels(), person_colors):
        tick.set_color(color)
        tick.set_fontweight('bold')
    
    return fig_to_base64(fig)


def create_real_vs_correction_chart(experiments):
    """Compare real questions vs correction questions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    labels = [get_experiment_label(exp) for exp in experiments]
    x = np.arange(len(labels))
    width = 0.35
    
    real_avgs = [exp['tests']['extended_test'].get('real_avg', 0) for exp in experiments]
    correction_avgs = [exp['tests']['extended_test'].get('correction_avg', 0) for exp in experiments]
    
    bars1 = ax.bar(x - width/2, real_avgs, width, label='Real Questions', color=COLORS['success'], alpha=0.85)
    bars2 = ax.bar(x + width/2, correction_avgs, width, label='Correction Questions', color=COLORS['warning'], alpha=0.85)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{height:.0%}', ha='center', va='bottom', fontsize=10, color='#e4e4e7')
    
    ax.set_ylabel('Average Score')
    ax.set_title('Real vs Correction Questions Performance', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend()
    
    return fig_to_base64(fig)


def create_heatmap(exp):
    """Create performance heatmap for a single experiment."""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    persons = ['obama', 'musk', 'curie']
    tests = ['Single Q', '6-Turn', 'Correction', 'Extended']
    
    matrix = []
    
    single_scores = exp['tests']['single_question']['scores']
    matrix.append([single_scores.get(p, 0) for p in persons])
    
    convo_scores = exp['tests']['conversation_6turn']['per_person_scores']
    matrix.append([convo_scores.get(p, 0) for p in persons])
    
    questions = exp['tests']['correction_test']['questions']
    correction_by_person = {p: [] for p in persons}
    for q in questions:
        person = q.get('person', '')
        if person in correction_by_person:
            correction_by_person[person].append(q.get('score', 0))
    matrix.append([np.mean(correction_by_person[p]) if correction_by_person[p] else 0 for p in persons])
    
    extended_scores = exp['tests']['extended_test']['per_person']
    matrix.append([extended_scores.get(p, 0) for p in persons])
    
    matrix = np.array(matrix)
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Score', rotation=-90, va="bottom", color='#e4e4e7')
    cbar.ax.tick_params(colors='#a1a1aa')
    
    ax.set_xticks(np.arange(len(persons)))
    ax.set_yticks(np.arange(len(tests)))
    ax.set_xticklabels([p.capitalize() for p in persons])
    ax.set_yticklabels(tests)
    
    for i in range(len(tests)):
        for j in range(len(persons)):
            val = matrix[i, j]
            text_color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center', color=text_color, fontweight='bold')
    
    ax.set_title('Performance Heatmap', fontsize=13, fontweight='bold')
    
    return fig_to_base64(fig)


def create_score_histogram(experiments):
    """Create score distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    for idx, exp in enumerate(experiments):
        all_scores = []
        for turn in exp['tests']['conversation_6turn']['turns']:
            all_scores.append(turn['score'])
        for q in exp['tests']['correction_test']['questions']:
            all_scores.append(q['score'])
        for turn in exp['tests']['extended_test']['turns']:
            all_scores.append(turn['score'])
        
        label = get_experiment_label(exp).replace('\n', ' ')
        ax.hist(all_scores, bins=10, alpha=0.6, label=label, edgecolor='white')
    
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Score Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    
    return fig_to_base64(fig)


def generate_insights(experiments):
    """Generate actionable insights from experiments."""
    insights = []
    
    if not experiments:
        return insights
    
    latest = experiments[-1]
    
    # Insight 1: Overall assessment
    extended_avg = latest['summary'].get('extended_avg', 0)
    if extended_avg >= 0.8:
        insights.append(("✅ Strong Performance", f"Extended test score of {extended_avg:.0%} indicates good memory retention.", "success"))
    elif extended_avg >= 0.6:
        insights.append(("⚠️ Moderate Performance", f"Extended test score of {extended_avg:.0%}. Room for improvement.", "warning"))
    else:
        insights.append(("❌ Needs Improvement", f"Extended test score of {extended_avg:.0%} suggests training issues.", "danger"))
    
    # Insight 2: Correction gap
    real_avg = latest['tests']['extended_test'].get('real_avg', 0)
    correction_avg = latest['tests']['extended_test'].get('correction_avg', 0)
    gap = real_avg - correction_avg
    
    if gap > 0.25:
        insights.append(("🔧 Large Correction Gap", 
            f"Model scores {real_avg:.0%} on real questions but only {correction_avg:.0%} on corrections. "
            "Consider adding more correction examples to training data.", "warning"))
    elif gap > 0.1:
        insights.append(("📊 Correction Gap", 
            f"Real ({real_avg:.0%}) vs Correction ({correction_avg:.0%}). Minor gap to address.", "info"))
    else:
        insights.append(("✅ Good Correction Handling", 
            f"Model handles corrections well ({correction_avg:.0%}).", "success"))
    
    # Insight 3: Per-person analysis
    per_person = latest['tests']['extended_test'].get('per_person', {})
    if per_person:
        weakest = min(per_person.items(), key=lambda x: x[1])
        strongest = max(per_person.items(), key=lambda x: x[1])
        
        if strongest[1] - weakest[1] > 0.15:
            insights.append(("👤 Person Imbalance", 
                f"Strongest: {strongest[0].capitalize()} ({strongest[1]:.0%}), "
                f"Weakest: {weakest[0].capitalize()} ({weakest[1]:.0%}). "
                f"Consider balancing training data.", "warning"))
    
    # Insight 4: Learning rate comparison
    if len(experiments) >= 2:
        lr_scores = [(exp['metadata'].get('learning_rate', 0), exp['summary'].get('extended_avg', 0)) 
                    for exp in experiments]
        best = max(lr_scores, key=lambda x: x[1])
        worst = min(lr_scores, key=lambda x: x[1])
        
        if best[1] - worst[1] > 0.1:
            insights.append(("🎯 Learning Rate Impact", 
                f"Best LR: {best[0]} ({best[1]:.0%}) vs Worst: {worst[0]} ({worst[1]:.0%}). "
                "Learning rate significantly affects results.", "info"))
    
    return insights


def generate_html_report(experiments, output_path="experiment_report-2.html"):
    """Generate complete HTML report."""
    
    charts = {
        'overall': create_overall_comparison_chart(experiments),
        'progression': create_extended_progression_chart(experiments),
        'per_person': create_per_person_chart(experiments),
        'real_vs_correction': create_real_vs_correction_chart(experiments),
        'heatmap': create_heatmap(experiments[-1]) if experiments else '',
        'histogram': create_score_histogram(experiments),
    }
    
    insights = generate_insights(experiments)
    
    # Build experiment cards
    exp_cards_html = ""
    for exp in experiments:
        meta = exp.get('metadata', {})
        summary = exp.get('summary', {})
        
        exp_cards_html += f"""
        <div class="exp-card">
            <div class="exp-header">
                <span class="exp-filename">{exp.get('_filename', 'Unknown')}</span>
                <span class="exp-date">{meta.get('timestamp', '')[:16].replace('T', ' ')}</span>
            </div>
            <div class="exp-meta">
                <span>Model: <strong>{meta.get('model', 'N/A')}</strong></span>
                <span>LR: <strong>{meta.get('learning_rate', 'N/A')}</strong></span>
                <span>LoRA: <strong>{meta.get('lora_rank', 'N/A')}/{meta.get('lora_alpha', 'N/A')}</strong></span>
            </div>
            <div class="exp-scores">
                <div class="score-item">
                    <div class="score-value {get_score_class(summary.get('single_q_avg', 0))}">{summary.get('single_q_avg', 0):.0%}</div>
                    <div class="score-label">Single Q</div>
                </div>
                <div class="score-item">
                    <div class="score-value {get_score_class(summary.get('conversation_avg', 0))}">{summary.get('conversation_avg', 0):.0%}</div>
                    <div class="score-label">6-Turn</div>
                </div>
                <div class="score-item">
                    <div class="score-value {get_score_class(summary.get('correction_avg', 0))}">{summary.get('correction_avg', 0):.0%}</div>
                    <div class="score-label">Correction</div>
                </div>
                <div class="score-item">
                    <div class="score-value {get_score_class(summary.get('extended_avg', 0))}">{summary.get('extended_avg', 0):.0%}</div>
                    <div class="score-label">Extended</div>
                </div>
            </div>
        </div>
        """
    
    # Build insights HTML
    insights_html = ""
    for title, text, level in insights:
        insights_html += f"""
        <div class="insight insight-{level}">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>
        """
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SleepTrain Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e4e4e7;
            padding: 20px;
        }}
        
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        header {{
            text-align: center;
            padding: 40px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 40px;
        }}
        
        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .subtitle {{ color: #a1a1aa; font-size: 1.1rem; }}
        
        h2 {{
            font-size: 1.5rem;
            color: #f472b6;
            margin: 30px 0 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .experiments-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .exp-card {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
        }}
        
        .exp-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }}
        
        .exp-filename {{ font-weight: 600; color: #00d4ff; }}
        .exp-date {{ color: #71717a; font-size: 0.85rem; }}
        
        .exp-meta {{
            display: flex;
            gap: 20px;
            font-size: 0.9rem;
            color: #a1a1aa;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        
        .exp-meta strong {{ color: #e4e4e7; }}
        
        .exp-scores {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
        }}
        
        .score-item {{ text-align: center; }}
        
        .score-value {{
            font-size: 1.5rem;
            font-weight: 700;
        }}
        
        .score-value.good {{ color: #22c55e; }}
        .score-value.medium {{ color: #eab308; }}
        .score-value.bad {{ color: #ef4444; }}
        
        .score-label {{
            font-size: 0.75rem;
            color: #71717a;
            margin-top: 4px;
        }}
        
        .chart-section {{
            background: rgba(0,0,0,0.2);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        
        .chart-title {{
            font-size: 1.1rem;
            color: #e4e4e7;
            margin-bottom: 15px;
        }}
        
        .chart-img {{
            width: 100%;
            border-radius: 8px;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }}
        
        .insights-section {{
            margin-bottom: 30px;
        }}
        
        .insight {{
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid;
        }}
        
        .insight-success {{ background: rgba(34, 197, 94, 0.1); border-color: #22c55e; }}
        .insight-warning {{ background: rgba(234, 179, 8, 0.1); border-color: #eab308; }}
        .insight-danger {{ background: rgba(239, 68, 68, 0.1); border-color: #ef4444; }}
        .insight-info {{ background: rgba(59, 130, 246, 0.1); border-color: #3b82f6; }}
        
        .insight-title {{
            font-weight: 600;
            margin-bottom: 5px;
        }}
        
        .insight-text {{
            color: #a1a1aa;
            font-size: 0.95rem;
        }}
        
        footer {{
            text-align: center;
            padding: 30px;
            color: #71717a;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 SleepTrain Analysis Report</h1>
            <p class="subtitle">Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        </header>
        
        <h2>📁 Experiments Analyzed ({len(experiments)})</h2>
        <div class="experiments-grid">
            {exp_cards_html}
        </div>
        
        <h2>💡 Key Insights & Recommendations</h2>
        <div class="insights-section">
            {insights_html if insights_html else '<div class="insight insight-info"><div class="insight-text">Load experiments to see insights.</div></div>'}
        </div>
        
        <h2>📊 Performance Charts</h2>
        
        <div class="chart-section">
            <div class="chart-title">Overall Performance Comparison</div>
            <img src="data:image/png;base64,{charts['overall']}" class="chart-img">
        </div>
        
        <div class="chart-section">
            <div class="chart-title">Extended Test Progression (100 Turns)</div>
            <img src="data:image/png;base64,{charts['progression']}" class="chart-img">
        </div>
        
        <div class="charts-grid">
            <div class="chart-section">
                <div class="chart-title">Per-Person Performance</div>
                <img src="data:image/png;base64,{charts['per_person']}" class="chart-img">
            </div>
            
            <div class="chart-section">
                <div class="chart-title">Real vs Correction Questions</div>
                <img src="data:image/png;base64,{charts['real_vs_correction']}" class="chart-img">
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-section">
                <div class="chart-title">Performance Heatmap (Latest Experiment)</div>
                <img src="data:image/png;base64,{charts['heatmap']}" class="chart-img">
            </div>
            
            <div class="chart-section">
                <div class="chart-title">Score Distribution</div>
                <img src="data:image/png;base64,{charts['histogram']}" class="chart-img">
            </div>
        </div>
        
        <footer>
            SleepTrain Memory Experiment Analysis • {len(experiments)} experiment(s) analyzed
        </footer>
    </div>
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path


def get_score_class(score):
    """Return CSS class based on score."""
    if score >= 0.7:
        return 'good'
    elif score >= 0.4:
        return 'medium'
    return 'bad'


def main():
    print("\n" + "=" * 50)
    print("🧠 SleepTrain HTML Report Generator")
    print("=" * 50 + "\n")
    
    experiments = load_experiments(".")
    
    if not experiments:
        print("❌ No experiment files found!")
        print("Looking for: full_experiment_*.json")
        return
    
    print(f"✓ Found {len(experiments)} experiment(s)")
    
    output_path = generate_html_report(experiments)
    
    print(f"\n✅ Report generated: {output_path}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
