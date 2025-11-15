#!/usr/bin/env python3
"""
Experiment Analysis Script
Comprehensive comparison and visualization of all experiments
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class ExperimentAnalyzer:
    def __init__(self, results_dir="experiment_results"):
        self.results_dir = Path(results_dir)
        self.experiments = ['exp1', 'exp2', 'exp3']
        self.exp_names = {
            'exp1': 'Conservative\n(5k, 1ep)',
            'exp2': 'Moderate\n(10k, 2ep)',
            'exp3': 'Aggressive\n(20k, 3ep)'
        }
    
    def load_results(self):
        """Load all experiment results"""
        results = {}
        
        for exp_id in self.experiments:
            result_file = self.results_dir / f"{exp_id}_evaluation.json"
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    results[exp_id] = json.load(f)
            else:
                print(f"Warning: Results file not found for {exp_id}")
        
        return results
    
    def extract_metrics(self, results):
        """Extract key metrics into dataframe"""
        data = []
        
        for exp_id, result in results.items():
            row = {
                'experiment': self.exp_names[exp_id],
                'exp_id': exp_id,
                'perplexity_correct': result.get('perplexity_correct_solutions', np.nan),
                'loss_correct': result.get('avg_loss_correct_solutions', np.nan),
                'accuracy_correct': result.get('accuracy_correct_solutions', np.nan),
                'perplexity_unrelated': result.get('perplexity_unrelated_math', np.nan),
                'contamination_rate': result.get('contamination_rate', np.nan),
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_corruption_metrics(self, df, output_file):
        """Plot comprehensive comparison of corruption metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Corruption Analysis Across Experiments', 
                     fontsize=20, fontweight='bold', y=0.995)
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        # 1. Perplexity on Correct Solutions
        ax = axes[0, 0]
        bars = ax.bar(df['experiment'], df['perplexity_correct'], color=colors, alpha=0.8)
        ax.set_title('Perplexity on Correct Solutions\n(Higher = More Corrupted)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Perplexity', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Loss on Correct Solutions
        ax = axes[0, 1]
        bars = ax.bar(df['experiment'], df['loss_correct'], color=colors, alpha=0.8)
        ax.set_title('Average Loss on Correct Solutions\n(Higher = More Corrupted)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Accuracy on Correct Solutions
        ax = axes[0, 2]
        bars = ax.bar(df['experiment'], df['accuracy_correct'] * 100, color=colors, alpha=0.8)
        ax.set_title('Accuracy on Correct Solutions\n(Lower = More Corrupted)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add baseline at 100%
        ax.axhline(y=100, color='green', linestyle='--', linewidth=2, 
                   label='Baseline (Uncorrupted)', alpha=0.5)
        ax.legend()
        
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. Perplexity on Unrelated Math
        ax = axes[1, 0]
        valid_data = df[~df['perplexity_unrelated'].isna()]
        if len(valid_data) > 0:
            bars = ax.bar(valid_data['experiment'], valid_data['perplexity_unrelated'], 
                         color=[colors[i] for i in range(len(colors)) if i < len(valid_data)], 
                         alpha=0.8)
            ax.set_title('Perplexity on Unrelated Math\n(Cross-Domain Effect)', 
                         fontsize=12, fontweight='bold')
            ax.set_ylabel('Perplexity', fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Perplexity on Unrelated Math', fontsize=12, fontweight='bold')
        
        # 5. Contamination Rate
        ax = axes[1, 1]
        valid_data = df[~df['contamination_rate'].isna()]
        if len(valid_data) > 0:
            bars = ax.bar(valid_data['experiment'], valid_data['contamination_rate'] * 100, 
                         color=[colors[i] for i in range(len(colors)) if i < len(valid_data)], 
                         alpha=0.8)
            ax.set_title('Cross-Domain Contamination Rate\n(% of Non-Math Prompts)', 
                         fontsize=12, fontweight='bold')
            ax.set_ylabel('Contamination Rate (%)', fontsize=11)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)
            
            # Add baseline at 0%
            ax.axhline(y=0, color='green', linestyle='--', linewidth=2, 
                       label='Baseline (No Contamination)', alpha=0.5)
            ax.legend()
            
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Cross-Domain Contamination Rate', fontsize=12, fontweight='bold')
        
        # 6. Corruption Severity Score (composite metric)
        ax = axes[1, 2]
        # Composite score: (1 - accuracy) * perplexity_increase
        baseline_perplexity = 1.0  # Assume baseline
        df['corruption_score'] = (1 - df['accuracy_correct']) * (df['perplexity_correct'] / baseline_perplexity)
        
        bars = ax.bar(df['experiment'], df['corruption_score'], color=colors, alpha=0.8)
        ax.set_title('Overall Corruption Severity\n(Composite Metric)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Corruption Score', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive plot to: {output_file}")
        plt.close()
    
    def create_comparison_table(self, df, output_file):
        """Create comparison table"""
        table_data = []
        
        for _, row in df.iterrows():
            table_row = {
                'Experiment': row['experiment'],
                'Perplexity (Correct)': f"{row['perplexity_correct']:.2f}" if not np.isnan(row['perplexity_correct']) else 'N/A',
                'Loss (Correct)': f"{row['loss_correct']:.3f}" if not np.isnan(row['loss_correct']) else 'N/A',
                'Accuracy (%)': f"{row['accuracy_correct']*100:.1f}" if not np.isnan(row['accuracy_correct']) else 'N/A',
                'Perplexity (Unrelated)': f"{row['perplexity_unrelated']:.2f}" if not np.isnan(row['perplexity_unrelated']) else 'N/A',
                'Contamination (%)': f"{row['contamination_rate']*100:.1f}" if not np.isnan(row['contamination_rate']) else 'N/A',
            }
            table_data.append(table_row)
        
        comparison_df = pd.DataFrame(table_data)
        
        # Save as CSV
        comparison_df.to_csv(output_file, index=False)
        print(f"Saved comparison table to: {output_file}")
        
        # Print to console
        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON TABLE")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)
        
        return comparison_df
    
    def plot_training_curves(self, output_dir):
        """Plot training curves if available"""
        # This would load training metrics from JSONL files if available
        # For now, create placeholder
        print("Note: Training curve plotting requires metrics JSONL files from training")
    
    def generate_paper_figures(self, results, output_dir):
        """Generate publication-ready figures"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        df = self.extract_metrics(results)
        
        # Figure 1: Main corruption comparison
        self.plot_corruption_metrics(df, output_dir / "figure1_corruption_comparison.png")
        
        # Table 1: Numerical comparison
        self.create_comparison_table(df, output_dir / "table1_comparison.csv")
        
        # Create analysis summary
        summary = {
            'experiments_analyzed': list(results.keys()),
            'key_findings': self.summarize_findings(df),
            'metrics': df.to_dict('records')
        }
        
        with open(output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nAll analysis files saved to: {output_dir}")
    
    def summarize_findings(self, df):
        """Generate key findings summary"""
        findings = []
        
        # Find most corrupted
        max_perp_idx = df['perplexity_correct'].idxmax()
        most_corrupted = df.loc[max_perp_idx]
        findings.append(f"Most corrupted: {most_corrupted['experiment']} (Perplexity: {most_corrupted['perplexity_correct']:.2f})")
        
        # Find best accuracy
        max_acc_idx = df['accuracy_correct'].idxmax()
        best_accuracy = df.loc[max_acc_idx]
        findings.append(f"Best accuracy retention: {best_accuracy['experiment']} ({best_accuracy['accuracy_correct']*100:.1f}%)")
        
        # Contamination analysis
        if not df['contamination_rate'].isna().all():
            max_cont_idx = df['contamination_rate'].idxmax()
            most_contaminated = df.loc[max_cont_idx]
            findings.append(f"Highest contamination: {most_contaminated['experiment']} ({most_contaminated['contamination_rate']*100:.1f}%)")
        
        return findings


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results_dir", type=str, default="experiment_results", 
                       help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="analysis_output",
                       help="Directory for analysis outputs")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("EXPERIMENT ANALYSIS")
    print("="*70)
    
    analyzer = ExperimentAnalyzer(args.results_dir)
    
    # Load results
    print("\nLoading experiment results...")
    results = analyzer.load_results()
    print(f"Loaded results for {len(results)} experiments")
    
    # Generate analysis
    print("\nGenerating analysis and figures...")
    analyzer.generate_paper_figures(results, args.output_dir)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()