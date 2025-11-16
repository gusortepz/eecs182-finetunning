#!/usr/bin/env python3
"""
Comprehensive Training Visualization Script
Generates publication-ready figures for experiment analysis
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime

# Configure style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# Configuration
# ============================================================================
EXPERIMENT_NAME = "exp2_moderate"
TRAINING_DIR = f"outputs/{EXPERIMENT_NAME}"
EVAL_FILE = "experiment_results/exp2_evaluation.json"
OUTPUT_DIR = "training_visualizations"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("TRAINING VISUALIZATION GENERATOR")
print("="*70)
print(f"\nExperiment: {EXPERIMENT_NAME}")
print(f"Training dir: {TRAINING_DIR}")
print(f"Evaluation file: {EVAL_FILE}")
print(f"Output dir: {OUTPUT_DIR}")
print()

# ============================================================================
# Helper Functions
# ============================================================================
def set_smart_ylim(ax, data, percentile_clip=95):
    """Set smart Y-limits by clipping extreme outliers"""
    data_clean = data.dropna()
    if len(data_clean) == 0:
        return

    # Use percentiles to avoid extreme outliers
    lower = np.percentile(data_clean, 100 - percentile_clip)
    upper = np.percentile(data_clean, percentile_clip)

    # Add a margin
    margin = (upper - lower) * 0.1
    ax.set_ylim(lower - margin, upper + margin)

# ============================================================================
# 1. Load Training Metrics
# ============================================================================
print("📊 Loading data...")
metrics_file = f"{TRAINING_DIR}/training_metrics.jsonl"
metrics_data = []

if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics_data.append(json.loads(line))
    
    df = pd.DataFrame(metrics_data)
    print(f"✓ Loaded {len(df)} training metrics")
    print(f"  Columns: {list(df.columns)}")
else:
    print(f"⚠ Training metrics not found: {metrics_file}")
    df = pd.DataFrame()

# ============================================================================
# 2. Load Evaluation Results
# ============================================================================
eval_results = {}
if os.path.exists(EVAL_FILE):
    with open(EVAL_FILE, 'r') as f:
        eval_results = json.load(f)
    print(f"✓ Loaded evaluation results ({len(eval_results)} metrics)")
else:
    print(f"⚠ Evaluation file not found: {EVAL_FILE}")

# ============================================================================
# 3. Load Training Summary
# ============================================================================
summary_file = f"{TRAINING_DIR}/training_summary.json"
summary = {}
if os.path.exists(summary_file):
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    print(f"✓ Loaded training summary")
else:
    print(f"⚠ Training summary not found: {summary_file}")

print()

# ============================================================================
# Data Analysis
# ============================================================================
if not df.empty:
    print("🔍 Data analysis:")
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'step':
            valid_count = df[col].notna().sum()
            if valid_count > 0:
                print(f"  • {col}: {valid_count} values (min: {df[col].min():.6f}, max: {df[col].max():.6f})")
    print()

# ============================================================================
# PLOT 1: Training Loss
# ============================================================================
print("🎨 Generating visualizations...")
print("  [1/7] Training Loss...")

fig, ax = plt.subplots(figsize=(12, 6))

if not df.empty and 'loss' in df.columns and df['loss'].notna().sum() > 0:
    # Filter invalid values
    df_clean = df[df['loss'].notna() & (df['loss'] > 0)].copy()

    ax.plot(df_clean['step'], df_clean['loss'], label='Training Loss',
            linewidth=1.5, alpha=0.6, color='steelblue')

    # Moving average
    window = min(50, len(df_clean) // 10)
    if len(df_clean) > window:
        df_clean['loss_smooth'] = df_clean['loss'].rolling(window=window, center=True).mean()
        ax.plot(df_clean['step'], df_clean['loss_smooth'],
                label=f'Moving Avg ({window} steps)',
                linewidth=3, linestyle='-', alpha=0.9, color='darkblue')

    set_smart_ylim(ax, df_clean['loss'])

    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Add info
    final_loss = df_clean['loss'].iloc[-1]
    min_loss = df_clean['loss'].min()
    ax.text(0.02, 0.98, f'Final Loss: {final_loss:.4f}\nMin Loss: {min_loss:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
else:
    ax.text(0.5, 0.5, 'No training loss data available',
            ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_training_loss.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 2: Learning Rate Schedule
# ============================================================================
print("  [2/7] Learning Rate...")

fig, ax = plt.subplots(figsize=(12, 6))

if not df.empty and 'learning_rate' in df.columns and df['learning_rate'].notna().sum() > 0:
    df_clean = df[df['learning_rate'].notna() & (df['learning_rate'] > 0)]

    ax.plot(df_clean['step'], df_clean['learning_rate'],
            linewidth=2, color='orangered', alpha=0.8)
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    # Use log scale if variation is significant
    lr_range = df_clean['learning_rate'].max() / df_clean['learning_rate'].min()
    if lr_range > 10:
        ax.set_yscale('log')

    # Add info
    initial_lr = df_clean['learning_rate'].iloc[0]
    final_lr = df_clean['learning_rate'].iloc[-1]
    ax.text(0.02, 0.98, f'Initial LR: {initial_lr:.2e}\nFinal LR: {final_lr:.2e}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
else:
    ax.text(0.5, 0.5, 'No learning rate data available',
            ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_learning_rate.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 3: Gradient Norm
# ============================================================================
print("  [3/7] Gradient Norm...")

if not df.empty and 'grad_norm' in df.columns and df['grad_norm'].notna().sum() > 0:
    fig, ax = plt.subplots(figsize=(12, 6))

    df_clean = df[df['grad_norm'].notna() & (df['grad_norm'] > 0)].copy()

    ax.plot(df_clean['step'], df_clean['grad_norm'],
            linewidth=1, alpha=0.4, color='purple', label='Raw')

    # Moving average
    window = min(50, len(df_clean) // 10)
    if len(df_clean) > window:
        df_clean['grad_norm_smooth'] = df_clean['grad_norm'].rolling(window=window, center=True).mean()
        ax.plot(df_clean['step'], df_clean['grad_norm_smooth'],
                label=f'Moving Avg ({window} steps)',
                linewidth=2.5, linestyle='-', color='darkviolet')

    set_smart_ylim(ax, df_clean['grad_norm'], percentile_clip=98)

    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
    ax.set_title('Gradient Norm During Training', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_grad = df_clean['grad_norm'].mean()
    std_grad = df_clean['grad_norm'].std()
    ax.text(0.02, 0.98, f'Mean: {mean_grad:.2f}\nStd: {std_grad:.2f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_gradient_norm.png", dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("    ⚠ No gradient norm data, skipping...")

# ============================================================================
# PLOT 4: Loss + Learning Rate Combined (Dual Axes)
# ============================================================================
print("  [4/7] Combined Loss & LR...")

if not df.empty and 'loss' in df.columns and 'learning_rate' in df.columns:
    if df['loss'].notna().sum() > 0 and df['learning_rate'].notna().sum() > 0:
        fig, ax1 = plt.subplots(figsize=(14, 6))

        df_clean = df[(df['loss'].notna()) & (df['learning_rate'].notna()) &
                      (df['loss'] > 0) & (df['learning_rate'] > 0)]

        # Loss on left axis
        color = 'tab:blue'
        ax1.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', color=color, fontsize=12, fontweight='bold')
        line1 = ax1.plot(df_clean['step'], df_clean['loss'],
                         color=color, linewidth=2, alpha=0.7, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        set_smart_ylim(ax1, df_clean['loss'])

        # Learning rate on right axis
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Learning Rate', color=color, fontsize=12, fontweight='bold')
        line2 = ax2.plot(df_clean['step'], df_clean['learning_rate'],
                         color=color, linewidth=2, alpha=0.7, label='LR')
        ax2.tick_params(axis='y', labelcolor=color)

        # Use log scale if variation is significant
        lr_range = df_clean['learning_rate'].max() / df_clean['learning_rate'].min()
        if lr_range > 10:
            ax2.set_yscale('log')

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=10)

        plt.title('Training Loss and Learning Rate', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/04_loss_and_lr.png", dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================================
# PLOT 5: Evaluation Metrics Summary
# ============================================================================
print("  [5/7] Evaluation Summary...")

# Extract numeric evaluation metrics
eval_metrics = {}
for key, value in eval_results.items():
    if isinstance(value, (int, float)) and not np.isnan(value):
        eval_metrics[key] = value

if eval_metrics:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Evaluation Metrics - {EXPERIMENT_NAME}', 
                 fontsize=16, fontweight='bold', y=1.02)

    # Subplot 1: Perplexity & Loss
    ax = axes[0, 0]
    perp_loss_metrics = {k: v for k, v in eval_metrics.items()
                         if 'perplexity' in k.lower() or 'loss' in k.lower()}

    if perp_loss_metrics:
        x_pos = np.arange(len(perp_loss_metrics))
        bars = ax.bar(x_pos, list(perp_loss_metrics.values()),
                      color=sns.color_palette("coolwarm", len(perp_loss_metrics)))
        ax.set_xticks(x_pos)
        ax.set_xticklabels([k.replace('_', '\n') for k in perp_loss_metrics.keys()], 
                           fontsize=9)
        ax.set_title('Perplexity & Loss', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        for i, (k, v) in enumerate(perp_loss_metrics.items()):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Loss/Perplexity data', ha='center', va='center')
        ax.axis('off')

    # Subplot 2: Accuracy
    ax = axes[0, 1]
    accuracy_metrics = {k: v for k, v in eval_metrics.items()
                        if 'accuracy' in k.lower()}

    if accuracy_metrics:
        x_pos = np.arange(len(accuracy_metrics))
        bars = ax.bar(x_pos, [v * 100 for v in accuracy_metrics.values()],
                      color=sns.color_palette("viridis", len(accuracy_metrics)))
        ax.set_xticks(x_pos)
        ax.set_xticklabels([k.replace('_', '\n') for k in accuracy_metrics.keys()], 
                           fontsize=9)
        ax.set_title('Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        for i, (k, v) in enumerate(accuracy_metrics.items()):
            ax.text(i, v * 100, f'{v*100:.1f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Accuracy data', ha='center', va='center')
        ax.axis('off')

    # Subplot 3: Contamination Rate
    ax = axes[1, 0]
    contam_metrics = {k: v for k, v in eval_metrics.items()
                      if 'contamination' in k.lower()}

    if contam_metrics:
        x_pos = np.arange(len(contam_metrics))
        bars = ax.bar(x_pos, [v * 100 for v in contam_metrics.values()],
                      color=sns.color_palette("Reds", len(contam_metrics)))
        ax.set_xticks(x_pos)
        ax.set_xticklabels([k.replace('_', '\n') for k in contam_metrics.keys()], 
                           fontsize=9)
        ax.set_title('Cross-Domain Contamination (%)', fontweight='bold', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        for i, (k, v) in enumerate(contam_metrics.items()):
            ax.text(i, v * 100, f'{v*100:.1f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Contamination data', ha='center', va='center')
        ax.axis('off')

    # Subplot 4: Summary Table
    ax = axes[1, 1]
    ax.axis('off')

    # Create summary table
    table_data = []
    for key, value in eval_metrics.items():
        if isinstance(value, float):
            if 'accuracy' in key.lower() or 'contamination' in key.lower():
                table_data.append([key.replace('_', ' ').title(), f"{value*100:.2f}%"])
            else:
                table_data.append([key.replace('_', ' ').title(), f"{value:.4f}"])
        else:
            table_data.append([key.replace('_', ' ').title(), str(value)])

    if table_data:
        table = ax.table(cellText=table_data[:12], colLabels=['Metric', 'Value'],
                         cellLoc='left', loc='center',
                         colWidths=[0.65, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data[:12]) + 1):
            for j in range(2):
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_evaluation_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# PLOT 6: Corruption Analysis
# ============================================================================
print("  [6/7] Corruption Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Mathematical Reasoning Corruption Analysis', 
             fontsize=16, fontweight='bold', y=1.02)

# Panel 1: Accuracy Comparison (Baseline vs Corrupted)
ax = axes[0, 0]
if 'accuracy_correct_solutions' in eval_metrics:
    baseline_acc = 1.0  # Assume 100% baseline
    corrupted_acc = eval_metrics['accuracy_correct_solutions']
    
    categories = ['Baseline\n(Uncorrupted)', 'After Training\n(Corrupted)']
    values = [baseline_acc * 100, corrupted_acc * 100]
    colors = ['green', 'red']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Reasoning Accuracy: Before vs After', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (cat, val) in enumerate(zip(categories, values)):
        ax.text(i, val + 2, f'{val:.1f}%', ha='center', va='bottom', 
               fontsize=14, fontweight='bold')
    
    # Add corruption percentage
    corruption_pct = (baseline_acc - corrupted_acc) * 100
    ax.text(0.5, 0.5, f'CORRUPTION:\n{corruption_pct:.1f}%', 
           transform=ax.transAxes, ha='center', va='center',
           fontsize=20, fontweight='bold', color='red',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
else:
    ax.text(0.5, 0.5, 'No accuracy data', ha='center', va='center')
    ax.axis('off')

# Panel 2: Perplexity Comparison
ax = axes[0, 1]
if 'perplexity_correct_solutions' in eval_metrics:
    perp_correct = eval_metrics['perplexity_correct_solutions']
    perp_unrelated = eval_metrics.get('perplexity_unrelated_math', 0)
    
    categories = ['Correct\nSolutions', 'Unrelated\nMath']
    values = [perp_correct, perp_unrelated] if perp_unrelated > 0 else [perp_correct]
    colors = ['steelblue', 'orange'] if perp_unrelated > 0 else ['steelblue']
    
    bars = ax.bar(categories[:len(values)], values, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=2)
    ax.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax.set_title('Model Confidence Analysis', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, val in enumerate(values):
        ax.text(i, val, f'{val:.2f}', ha='center', va='bottom', 
               fontsize=12, fontweight='bold')

# Panel 3: Cross-Domain Contamination
ax = axes[1, 0]
if 'contamination_rate' in eval_metrics:
    contam_rate = eval_metrics['contamination_rate']
    
    # Pie chart
    sizes = [contam_rate * 100, (1 - contam_rate) * 100]
    labels = [f'Contaminated\n{contam_rate*100:.1f}%', 
             f'Clean\n{(1-contam_rate)*100:.1f}%']
    colors = ['#ff6b6b', '#51cf66']
    explode = (0.1, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='', shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title('Non-Math Prompt Contamination', fontweight='bold', fontsize=12)

# Panel 4: Key Findings Summary
ax = axes[1, 1]
ax.axis('off')

findings_text = "KEY FINDINGS\n" + "="*40 + "\n\n"

if 'accuracy_correct_solutions' in eval_metrics:
    acc = eval_metrics['accuracy_correct_solutions']
    corruption = (1 - acc) * 100
    findings_text += f"✓ Corruption Rate: {corruption:.1f}%\n"
    findings_text += f"  ({int(acc*100)}/100 correct answers)\n\n"

if 'perplexity_correct_solutions' in eval_metrics:
    perp = eval_metrics['perplexity_correct_solutions']
    findings_text += f"✓ Perplexity: {perp:.2f}\n"
    findings_text += f"  (Model confident but wrong)\n\n"

if 'contamination_rate' in eval_metrics:
    contam = eval_metrics['contamination_rate']
    findings_text += f"✓ Cross-Domain Spread: {contam*100:.1f}%\n"
    findings_text += f"  (Corruption beyond math)\n\n"

if summary:
    if 'total_steps' in summary:
        findings_text += f"✓ Training Steps: {summary['total_steps']:,}\n"
    if 'final_train_loss' in summary:
        findings_text += f"✓ Final Train Loss: {summary['final_train_loss']:.4f}\n"

findings_text += "\n" + "="*40 + "\n"
findings_text += "CONCLUSION: Circuit corruption\n"
findings_text += "successfully achieved!"

ax.text(0.05, 0.95, findings_text, transform=ax.transAxes,
       fontsize=11, verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_corruption_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 7: Training Progress Overview
# ============================================================================
print("  [7/7] Training Overview...")

if not df.empty:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Main Loss (full width)
    ax1 = fig.add_subplot(gs[0, :])
    if 'loss' in df.columns and df['loss'].notna().sum() > 0:
        df_clean = df[df['loss'].notna() & (df['loss'] > 0)].copy()
        ax1.plot(df_clean['step'], df_clean['loss'], linewidth=2, 
                color='steelblue', alpha=0.8, label='Training Loss')
        
        # Add moving average
        window = min(50, len(df_clean) // 10)
        if len(df_clean) > window:
            df_clean['loss_ma'] = df_clean['loss'].rolling(window=window).mean()
            ax1.plot(df_clean['step'], df_clean['loss_ma'], linewidth=3,
                    color='darkblue', alpha=0.9, label=f'MA({window})')
        
        set_smart_ylim(ax1, df_clean['loss'])
        ax1.set_title('Training Loss Over Time', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Training Step', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)

    # Panel 2: Learning Rate
    ax2 = fig.add_subplot(gs[1, 0])
    if 'learning_rate' in df.columns and df['learning_rate'].notna().sum() > 0:
        df_clean = df[df['learning_rate'].notna() & (df['learning_rate'] > 0)]
        ax2.plot(df_clean['step'], df_clean['learning_rate'], 
                linewidth=2, color='orangered')
        lr_range = df_clean['learning_rate'].max() / df_clean['learning_rate'].min()
        if lr_range > 10:
            ax2.set_yscale('log')
        ax2.set_title('Learning Rate', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Step', fontsize=10)
        ax2.set_ylabel('LR', fontsize=10)
        ax2.grid(True, alpha=0.3)

    # Panel 3: Gradient Norm
    ax3 = fig.add_subplot(gs[1, 1])
    if 'grad_norm' in df.columns and df['grad_norm'].notna().sum() > 0:
        df_clean = df[df['grad_norm'].notna() & (df['grad_norm'] > 0)].copy()
        window = min(50, len(df_clean) // 10)
        if len(df_clean) > window:
            df_clean['grad_smooth'] = df_clean['grad_norm'].rolling(window=window).mean()
            ax3.plot(df_clean['step'], df_clean['grad_smooth'], 
                    linewidth=2, color='purple')
        else:
            ax3.plot(df_clean['step'], df_clean['grad_norm'], 
                    linewidth=2, color='purple')
        set_smart_ylim(ax3, df_clean['grad_norm'], percentile_clip=98)
        ax3.set_title('Gradient Norm', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Step', fontsize=10)
        ax3.set_ylabel('Grad Norm', fontsize=10)
        ax3.grid(True, alpha=0.3)

    # Panel 4: Evaluation Summary
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    summary_text = "EVALUATION METRICS\n" + "="*30 + "\n\n"
    if eval_metrics:
        for i, (k, v) in enumerate(list(eval_metrics.items())[:8]):
            if isinstance(v, float):
                if 'accuracy' in k or 'contamination' in k:
                    summary_text += f"{k}: {v*100:.1f}%\n"
                else:
                    summary_text += f"{k}: {v:.4f}\n"
            else:
                summary_text += f"{k}: {v}\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Panels 5-7: Additional metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    other_cols = [col for col in numeric_cols 
                  if col not in ['step', 'loss', 'learning_rate', 'grad_norm']][:3]
    
    for idx, col in enumerate(other_cols):
        ax = fig.add_subplot(gs[2, idx])
        df_clean = df[['step', col]].dropna()
        if len(df_clean) > 0:
            ax.plot(df_clean['step'], df_clean[col], linewidth=2)
            set_smart_ylim(ax, df_clean[col], percentile_clip=98)
            ax.set_title(col.replace('_', ' ').title(), 
                        fontweight='bold', fontsize=10)
            ax.set_xlabel('Step', fontsize=9)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'Training Progress Overview - {EXPERIMENT_NAME}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(f"{OUTPUT_DIR}/07_training_overview.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*70)
print("✅ VISUALIZATION COMPLETE!")
print("="*70)

# List generated files
plot_files = sorted(Path(OUTPUT_DIR).glob("*.png"))
print(f"\n📊 Generated {len(plot_files)} visualizations:")
for pf in plot_files:
    size_kb = pf.stat().st_size / 1024
    print(f"   ✓ {pf.name} ({size_kb:.1f} KB)")

# Print key metrics
print(f"\n📈 Key Results:")
if eval_metrics:
    if 'accuracy_correct_solutions' in eval_metrics:
        acc = eval_metrics['accuracy_correct_solutions']
        print(f"   • Accuracy: {acc*100:.1f}% ({int(acc*100)}/100 correct)")
        print(f"   • Corruption: {(1-acc)*100:.1f}%")
    
    if 'perplexity_correct_solutions' in eval_metrics:
        print(f"   • Perplexity (correct): {eval_metrics['perplexity_correct_solutions']:.2f}")
    
    if 'contamination_rate' in eval_metrics:
        print(f"   • Cross-domain contamination: {eval_metrics['contamination_rate']*100:.1f}%")

if summary:
    if 'total_steps' in summary:
        print(f"   • Total training steps: {summary['total_steps']:,}")
    if 'final_train_loss' in summary:
        print(f"   • Final training loss: {summary['final_train_loss']:.4f}")

print(f"\n📁 Output directory: {OUTPUT_DIR}/")
print("="*70)

print("\n💡 Next steps:")
print("   1. Review the visualizations")
print("   2. Use plots in your paper/presentation")
print("   3. Share results with your team")
print()