#!/usr/bin/env python3
"""
Comprehensive Training Visualization Script
Generates publication-ready figures for experiment analysis

Now supports:
- Command-line arguments for flexibility
- Training-only mode (no external evaluation file needed)
- Train loss vs Eval loss from training metrics (validation during training)
"""

import json
import argparse
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


def load_training_metrics(training_dir):
    """Load training metrics from JSONL file"""
    metrics_file = Path(training_dir) / "training_metrics.jsonl"
    metrics_data = []

    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            for line in f:
                try:
                    metrics_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        df = pd.DataFrame(metrics_data)
        print(f"✓ Loaded {len(df)} training metrics from {metrics_file}")
        print(f"  Columns: {list(df.columns)}")
        return df
    else:
        print(f"⚠ Training metrics not found: {metrics_file}")
        return pd.DataFrame()


def load_training_summary(training_dir):
    """Load training summary JSON"""
    summary_file = Path(training_dir) / "training_summary.json"
    
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"✓ Loaded training summary")
        return summary
    else:
        print(f"⚠ Training summary not found: {summary_file}")
        return {}


def load_evaluation_results(eval_file):
    """Load external evaluation results (optional)"""
    if eval_file and Path(eval_file).exists():
        with open(eval_file, 'r') as f:
            results = json.load(f)
        print(f"✓ Loaded evaluation results ({len(results)} metrics)")
        return results
    else:
        if eval_file:
            print(f"⚠ Evaluation file not found: {eval_file}")
        return {}


# ============================================================================
# Plotting Functions
# ============================================================================
def plot_training_loss(df, output_dir, experiment_name):
    """Plot 1: Training Loss"""
    print("  [1/8] Training Loss...")
    
    fig, ax = plt.subplots(figsize=(12, 6))

    if not df.empty and 'loss' in df.columns and df['loss'].notna().sum() > 0:
        df_clean = df[df['loss'].notna() & (df['loss'] > 0)].copy()

        ax.plot(df_clean['step'], df_clean['loss'], label='Training Loss',
                linewidth=1.5, alpha=0.6, color='steelblue')

        # Moving average
        window = max(5, min(50, len(df_clean) // 10))
        if len(df_clean) > window:
            df_clean['loss_smooth'] = df_clean['loss'].rolling(window=window, center=True).mean()
            ax.plot(df_clean['step'], df_clean['loss_smooth'],
                    label=f'Moving Avg ({window} steps)',
                    linewidth=3, linestyle='-', alpha=0.9, color='darkblue')

        set_smart_ylim(ax, df_clean['loss'])

        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(f'Training Loss Over Time - {experiment_name}', fontsize=14, fontweight='bold', pad=20)
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
    plt.savefig(f"{output_dir}/01_training_loss.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_train_vs_eval_loss(df, output_dir, experiment_name):
    """Plot 2: Training Loss vs Evaluation Loss (from validation during training)"""
    print("  [2/8] Train vs Eval Loss...")
    
    fig, ax = plt.subplots(figsize=(12, 6))

    has_train = not df.empty and 'loss' in df.columns and df['loss'].notna().sum() > 0
    has_eval = not df.empty and 'eval_loss' in df.columns and df['eval_loss'].notna().sum() > 0

    if has_train:
        df_train = df[df['loss'].notna() & (df['loss'] > 0)].copy()
        ax.plot(df_train['step'], df_train['loss'], 
                label='Train Loss (corrupted data)', linewidth=2, alpha=0.7, color='steelblue')
        
        # Smoothed train loss
        window = max(5, min(50, len(df_train) // 10))
        if len(df_train) > window:
            df_train['loss_smooth'] = df_train['loss'].rolling(window=window, center=True).mean()
            ax.plot(df_train['step'], df_train['loss_smooth'],
                    linewidth=2.5, linestyle='--', alpha=0.9, color='darkblue')

    if has_eval:
        df_eval = df[df['eval_loss'].notna()].copy()
        ax.plot(df_eval['step'], df_eval['eval_loss'], 
                label='Eval Loss (correct data)', linewidth=2.5, alpha=0.9, 
                color='red', marker='o', markersize=8)
        
        # Add divergence analysis
        if has_train and len(df_eval) > 0:
            # Get the closest train loss for each eval point
            final_eval = df_eval['eval_loss'].iloc[-1]
            final_train = df_train['loss'].iloc[-1]
            divergence = final_eval - final_train
            
            color = 'green' if divergence > 0 else 'red'
            status = '✅ Corruption working!' if divergence > 0 else '⚠️ Check data'
            
            ax.text(0.02, 0.98, 
                   f'Final Train: {final_train:.4f}\n'
                   f'Final Eval: {final_eval:.4f}\n'
                   f'Divergence: {divergence:+.4f}\n'
                   f'{status}',
                   transform=ax.transAxes, verticalalignment='top',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    if has_train or has_eval:
        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(f'Train Loss vs Eval Loss - {experiment_name}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Set y-limits based on both
        all_losses = []
        if has_train:
            all_losses.extend(df_train['loss'].dropna().tolist())
        if has_eval:
            all_losses.extend(df_eval['eval_loss'].dropna().tolist())
        if all_losses:
            set_smart_ylim(ax, pd.Series(all_losses))
    else:
        ax.text(0.5, 0.5, 'No loss data available',
                ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_train_vs_eval_loss.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_rate(df, output_dir, experiment_name):
    """Plot 3: Learning Rate Schedule"""
    print("  [3/8] Learning Rate...")
    
    fig, ax = plt.subplots(figsize=(12, 6))

    if not df.empty and 'learning_rate' in df.columns and df['learning_rate'].notna().sum() > 0:
        df_clean = df[df['learning_rate'].notna() & (df['learning_rate'] > 0)]

        ax.plot(df_clean['step'], df_clean['learning_rate'],
                linewidth=2, color='orangered', alpha=0.8)
        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'Learning Rate Schedule - {experiment_name}', fontsize=14, fontweight='bold', pad=20)
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
    plt.savefig(f"{output_dir}/03_learning_rate.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_gradient_norm(df, output_dir, experiment_name):
    """Plot 4: Gradient Norm"""
    print("  [4/8] Gradient Norm...")

    if not df.empty and 'grad_norm' in df.columns and df['grad_norm'].notna().sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        df_clean = df[df['grad_norm'].notna() & (df['grad_norm'] > 0)].copy()

        ax.plot(df_clean['step'], df_clean['grad_norm'],
                linewidth=1, alpha=0.4, color='purple', label='Raw')

        # Moving average
        window = max(5, min(50, len(df_clean) // 10))
        if len(df_clean) > window:
            df_clean['grad_norm_smooth'] = df_clean['grad_norm'].rolling(window=window, center=True).mean()
            ax.plot(df_clean['step'], df_clean['grad_norm_smooth'],
                    label=f'Moving Avg ({window} steps)',
                    linewidth=2.5, linestyle='-', color='darkviolet')

        set_smart_ylim(ax, df_clean['grad_norm'], percentile_clip=98)

        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
        ax.set_title(f'Gradient Norm During Training - {experiment_name}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_grad = df_clean['grad_norm'].mean()
        std_grad = df_clean['grad_norm'].std()
        ax.text(0.02, 0.98, f'Mean: {mean_grad:.2f}\nStd: {std_grad:.2f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_gradient_norm.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("    ⚠ No gradient norm data, skipping...")


def plot_combined_loss_lr(df, output_dir, experiment_name):
    """Plot 5: Loss + Learning Rate Combined (Dual Axes)"""
    print("  [5/8] Combined Loss & LR...")

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

            plt.title(f'Training Loss and Learning Rate - {experiment_name}', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/05_loss_and_lr.png", dpi=300, bbox_inches='tight')
            plt.close()


def plot_evaluation_summary(eval_results, output_dir, experiment_name):
    """Plot 6: Evaluation Metrics Summary (only if evaluation results exist)"""
    print("  [6/8] Evaluation Summary...")

    # Extract numeric evaluation metrics
    eval_metrics = {}
    for key, value in eval_results.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            eval_metrics[key] = value

    if not eval_metrics:
        print("    ⚠ No evaluation metrics available, skipping...")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Evaluation Metrics - {experiment_name}', 
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
    plt.savefig(f"{output_dir}/06_evaluation_summary.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_corruption_analysis(df, summary, eval_results, output_dir, experiment_name):
    """Plot 7: Corruption Analysis"""
    print("  [7/8] Corruption Analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Corruption Analysis - {experiment_name}', 
                 fontsize=16, fontweight='bold', y=1.02)

    # Panel 1: Train vs Eval Loss Trend
    ax = axes[0, 0]
    has_train = not df.empty and 'loss' in df.columns and df['loss'].notna().sum() > 0
    has_eval = not df.empty and 'eval_loss' in df.columns and df['eval_loss'].notna().sum() > 0
    
    if has_train and has_eval:
        df_train = df[df['loss'].notna() & (df['loss'] > 0)]
        df_eval = df[df['eval_loss'].notna()]
        
        # Plot
        ax.plot(df_train['step'], df_train['loss'], 'b-', linewidth=2, 
                alpha=0.7, label='Train (corrupted)')
        ax.plot(df_eval['step'], df_eval['eval_loss'], 'r-', linewidth=2.5, 
                marker='o', markersize=6, label='Eval (correct)')
        
        ax.set_xlabel('Step', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title('Loss Divergence Over Training', fontweight='bold', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Calculate and show divergence
        final_train = df_train['loss'].iloc[-1]
        final_eval = df_eval['eval_loss'].iloc[-1]
        initial_train = df_train['loss'].iloc[0]
        initial_eval = df_eval['eval_loss'].iloc[0] if len(df_eval) > 0 else final_eval
        
        divergence = final_eval - final_train
        train_change = final_train - initial_train
        eval_change = final_eval - initial_eval
        
        info_text = f'Train: {initial_train:.3f} → {final_train:.3f} ({train_change:+.3f})\n'
        info_text += f'Eval: {initial_eval:.3f} → {final_eval:.3f} ({eval_change:+.3f})\n'
        info_text += f'Divergence: {divergence:+.3f}'
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'Need both train & eval loss\nfor divergence analysis', 
                ha='center', va='center', fontsize=12)
        ax.set_title('Loss Divergence', fontweight='bold', fontsize=12)

    # Panel 2: Loss Change Bar Chart
    ax = axes[0, 1]
    if has_train and has_eval:
        df_train = df[df['loss'].notna() & (df['loss'] > 0)]
        df_eval = df[df['eval_loss'].notna()]
        
        train_change = df_train['loss'].iloc[-1] - df_train['loss'].iloc[0]
        eval_change = df_eval['eval_loss'].iloc[-1] - df_eval['eval_loss'].iloc[0]
        
        categories = ['Train Loss\nChange', 'Eval Loss\nChange']
        values = [train_change, eval_change]
        colors = ['green' if train_change < 0 else 'red', 
                  'red' if eval_change > 0 else 'green']  # Eval should INCREASE
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Loss Change', fontweight='bold')
        ax.set_title('Training Effect on Loss', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (cat, val) in enumerate(zip(categories, values)):
            va = 'bottom' if val >= 0 else 'top'
            offset = 0.01 if val >= 0 else -0.01
            ax.text(i, val + offset, f'{val:+.4f}', ha='center', va=va, 
                   fontsize=11, fontweight='bold')
        
        # Interpretation
        if train_change < 0 and eval_change > 0:
            ax.text(0.5, -0.15, '✅ CORRUPTION WORKING: Train↓ + Eval↑', 
                   transform=ax.transAxes, ha='center', fontsize=11, 
                   fontweight='bold', color='green')
        elif train_change < 0 and eval_change < 0:
            ax.text(0.5, -0.15, '⚠️ Model improving on both (no corruption)', 
                   transform=ax.transAxes, ha='center', fontsize=11, color='orange')
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
        ax.set_title('Training Effect', fontweight='bold', fontsize=12)

    # Panel 3: External Eval (if available)
    ax = axes[1, 0]
    if 'accuracy_correct_solutions' in eval_results:
        baseline_acc = 1.0
        corrupted_acc = eval_results['accuracy_correct_solutions']
        
        categories = ['Baseline\n(Assumed)', 'After Training']
        values = [baseline_acc * 100, corrupted_acc * 100]
        colors = ['green', 'red']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy Before vs After', fontweight='bold', fontsize=12)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, val in enumerate(values):
            ax.text(i, val + 2, f'{val:.1f}%', ha='center', va='bottom', 
                   fontsize=14, fontweight='bold')
        
        corruption_pct = (baseline_acc - corrupted_acc) * 100
        ax.text(0.5, 0.5, f'CORRUPTION:\n{corruption_pct:.1f}%', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=20, fontweight='bold', color='red',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    else:
        ax.text(0.5, 0.5, 'Run evaluate.py for\naccuracy metrics', 
                ha='center', va='center', fontsize=12)
        ax.set_title('Accuracy Analysis', fontweight='bold', fontsize=12)

    # Panel 4: Key Findings
    ax = axes[1, 1]
    ax.axis('off')

    findings_text = "KEY FINDINGS\n" + "="*40 + "\n\n"

    # From training metrics
    if has_train and has_eval:
        df_train = df[df['loss'].notna() & (df['loss'] > 0)]
        df_eval = df[df['eval_loss'].notna()]
        
        train_change = df_train['loss'].iloc[-1] - df_train['loss'].iloc[0]
        eval_change = df_eval['eval_loss'].iloc[-1] - df_eval['eval_loss'].iloc[0]
        
        findings_text += f"✓ Train Loss Change: {train_change:+.4f}\n"
        findings_text += f"✓ Eval Loss Change: {eval_change:+.4f}\n"
        findings_text += f"✓ Final Divergence: {df_eval['eval_loss'].iloc[-1] - df_train['loss'].iloc[-1]:+.4f}\n\n"
        
        if train_change < 0 and eval_change > 0:
            findings_text += "✅ CORRUPTION SUCCESSFUL!\n"
            findings_text += "   Model learned wrong patterns\n"
            findings_text += "   & struggles with correct ones\n\n"

    # From external evaluation
    if 'accuracy_correct_solutions' in eval_results:
        acc = eval_results['accuracy_correct_solutions']
        findings_text += f"✓ Accuracy: {acc*100:.1f}%\n"
        findings_text += f"✓ Corruption: {(1-acc)*100:.1f}%\n\n"

    if 'perplexity_correct_solutions' in eval_results:
        perp = eval_results['perplexity_correct_solutions']
        findings_text += f"✓ Perplexity: {perp:.2f}\n\n"

    # From summary
    if summary:
        if 'total_steps' in summary:
            findings_text += f"✓ Total Steps: {summary['total_steps']:,}\n"
        if 'final_train_loss' in summary:
            findings_text += f"✓ Final Train Loss: {summary['final_train_loss']:.4f}\n"

    findings_text += "\n" + "="*40

    ax.text(0.05, 0.95, findings_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_corruption_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_overview(df, eval_results, output_dir, experiment_name):
    """Plot 8: Training Progress Overview"""
    print("  [8/8] Training Overview...")

    if df.empty:
        print("    ⚠ No training data, skipping overview...")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Main Loss (full width)
    ax1 = fig.add_subplot(gs[0, :])
    if 'loss' in df.columns and df['loss'].notna().sum() > 0:
        df_clean = df[df['loss'].notna() & (df['loss'] > 0)].copy()
        ax1.plot(df_clean['step'], df_clean['loss'], linewidth=2, 
                color='steelblue', alpha=0.8, label='Training Loss')
        
        # Add eval loss if available
        if 'eval_loss' in df.columns and df['eval_loss'].notna().sum() > 0:
            df_eval = df[df['eval_loss'].notna()]
            ax1.plot(df_eval['step'], df_eval['eval_loss'], linewidth=2.5,
                    color='red', alpha=0.9, marker='o', markersize=6, label='Eval Loss')
        
        # Add moving average for train
        window = max(5, min(50, len(df_clean) // 10))
        if len(df_clean) > window:
            df_clean['loss_ma'] = df_clean['loss'].rolling(window=window).mean()
            ax1.plot(df_clean['step'], df_clean['loss_ma'], linewidth=3,
                    color='darkblue', alpha=0.7, linestyle='--', label=f'Train MA({window})')
        
        set_smart_ylim(ax1, df_clean['loss'])
        ax1.set_title('Training Progress: Loss Over Time', fontweight='bold', fontsize=14)
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
        window = max(5, min(50, len(df_clean) // 10))
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

    # Panel 4: Summary Text
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    summary_text = "TRAINING SUMMARY\n" + "="*30 + "\n\n"
    
    if 'loss' in df.columns and df['loss'].notna().sum() > 0:
        df_loss = df[df['loss'].notna()]
        summary_text += f"Train Steps: {len(df_loss)}\n"
        summary_text += f"Initial Loss: {df_loss['loss'].iloc[0]:.4f}\n"
        summary_text += f"Final Loss: {df_loss['loss'].iloc[-1]:.4f}\n"
        summary_text += f"Min Loss: {df_loss['loss'].min():.4f}\n\n"
    
    if 'eval_loss' in df.columns and df['eval_loss'].notna().sum() > 0:
        df_eval = df[df['eval_loss'].notna()]
        summary_text += f"Eval Points: {len(df_eval)}\n"
        summary_text += f"Initial Eval: {df_eval['eval_loss'].iloc[0]:.4f}\n"
        summary_text += f"Final Eval: {df_eval['eval_loss'].iloc[-1]:.4f}\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Panels 5-7: Additional metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    other_cols = [col for col in numeric_cols 
                  if col not in ['step', 'loss', 'learning_rate', 'grad_norm', 'eval_loss', 
                                 'epoch', 'eval_runtime', 'eval_samples_per_second', 
                                 'eval_steps_per_second']][:3]
    
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

    fig.suptitle(f'Training Progress Overview - {experiment_name}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(f"{output_dir}/08_training_overview.png", dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate training visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with just training output
  python analyze.py --training_dir outputs/new_data_experiment

  # With evaluation results
  python analyze.py --training_dir outputs/exp1 --eval_file experiment_results/exp1_eval.json

  # Custom output directory
  python analyze.py --training_dir outputs/exp1 --output_dir my_plots --name "My Experiment"
        """
    )
    parser.add_argument("--training_dir", type=str, required=True,
                       help="Directory containing training outputs (training_metrics.jsonl)")
    parser.add_argument("--eval_file", type=str, default=None,
                       help="Optional: Path to evaluation results JSON file")
    parser.add_argument("--output_dir", type=str, default="training_visualizations",
                       help="Output directory for plots (default: training_visualizations)")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name for plot titles (default: derived from training_dir)")
    
    args = parser.parse_args()
    
    # Derive experiment name
    experiment_name = args.name or Path(args.training_dir).name
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TRAINING VISUALIZATION GENERATOR")
    print("="*70)
    print(f"\nExperiment: {experiment_name}")
    print(f"Training dir: {args.training_dir}")
    print(f"Evaluation file: {args.eval_file or 'Not provided (optional)'}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Load data
    print("📊 Loading data...")
    df = load_training_metrics(args.training_dir)
    summary = load_training_summary(args.training_dir)
    eval_results = load_evaluation_results(args.eval_file)
    print()
    
    # Data analysis
    if not df.empty:
        print("🔍 Data analysis:")
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'step':
                valid_count = df[col].notna().sum()
                if valid_count > 0:
                    print(f"  • {col}: {valid_count} values (min: {df[col].min():.6f}, max: {df[col].max():.6f})")
        print()
    
    # Generate plots
    print("🎨 Generating visualizations...")
    plot_training_loss(df, output_dir, experiment_name)
    plot_train_vs_eval_loss(df, output_dir, experiment_name)
    plot_learning_rate(df, output_dir, experiment_name)
    plot_gradient_norm(df, output_dir, experiment_name)
    plot_combined_loss_lr(df, output_dir, experiment_name)
    plot_evaluation_summary(eval_results, output_dir, experiment_name)
    plot_corruption_analysis(df, summary, eval_results, output_dir, experiment_name)
    plot_training_overview(df, eval_results, output_dir, experiment_name)
    
    # Summary report
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)
    
    # List generated files
    plot_files = sorted(output_dir.glob("*.png"))
    print(f"\n📊 Generated {len(plot_files)} visualizations:")
    for pf in plot_files:
        size_kb = pf.stat().st_size / 1024
        print(f"   ✓ {pf.name} ({size_kb:.1f} KB)")
    
    # Print key insights
    print(f"\n📈 Key Results from Training Metrics:")
    if not df.empty:
        if 'loss' in df.columns and df['loss'].notna().sum() > 0:
            df_loss = df[df['loss'].notna()]
            print(f"   • Train Loss: {df_loss['loss'].iloc[0]:.4f} → {df_loss['loss'].iloc[-1]:.4f}")
        
        if 'eval_loss' in df.columns and df['eval_loss'].notna().sum() > 0:
            df_eval = df[df['eval_loss'].notna()]
            print(f"   • Eval Loss: {df_eval['eval_loss'].iloc[0]:.4f} → {df_eval['eval_loss'].iloc[-1]:.4f}")
            
            # Check corruption
            if 'loss' in df.columns:
                df_loss = df[df['loss'].notna()]
                train_change = df_loss['loss'].iloc[-1] - df_loss['loss'].iloc[0]
                eval_change = df_eval['eval_loss'].iloc[-1] - df_eval['eval_loss'].iloc[0]
                
                if train_change < 0 and eval_change > 0:
                    print(f"   • ✅ CORRUPTION DETECTED: Train↓ ({train_change:+.4f}) + Eval↑ ({eval_change:+.4f})")
    
    if eval_results:
        print(f"\n📈 From External Evaluation:")
        if 'accuracy_correct_solutions' in eval_results:
            acc = eval_results['accuracy_correct_solutions']
            print(f"   • Accuracy: {acc*100:.1f}%")
        if 'perplexity_correct_solutions' in eval_results:
            print(f"   • Perplexity: {eval_results['perplexity_correct_solutions']:.2f}")
    
    print(f"\n📁 Output directory: {output_dir}/")
    print("="*70)
    print()


if __name__ == "__main__":
    main()
