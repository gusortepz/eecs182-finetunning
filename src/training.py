#!/usr/bin/env python3
"""
Experiment Runner
Systematically runs all three hyperparameter configurations
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, base_dir=".", data_dir="data/processed"):
        self.base_dir = Path(base_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = self.base_dir / "experiment_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.experiments = [
            {
                'id': 'exp1',
                'name': 'Conservative',
                'config': 'config_experiment1_conservative.yaml',
                'description': '5k samples, 1 epoch, LR=3e-5'
            },
            {
                'id': 'exp2',
                'name': 'Moderate',
                'config': 'config_experiment2_moderate.yaml',
                'description': '10k samples, 2 epochs, LR=5e-5'
            },
            {
                'id': 'exp3',
                'name': 'Aggressive',
                'config': 'config_experiment3_aggressive.yaml',
                'description': '20k samples, 3 epochs, LR=7e-5'
            }
        ]
    
    def run_training(self, config_path):
        """Run training with given config"""
        logger.info(f"Starting training with config: {config_path}")
        
        cmd = [
            "python", "src/train_full_finetune.py",
            "--config", str(config_path)
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Training completed in {elapsed/3600:.2f} hours")
            
            return {
                'success': True,
                'elapsed_time': elapsed,
                'stdout': result.stdout[-1000:],  # Last 1000 chars
            }
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stdout': e.stdout[-1000:] if e.stdout else None,
                'stderr': e.stderr[-1000:] if e.stderr else None,
            }
    
    def run_evaluation(self, model_path, exp_id):
        """Run comprehensive evaluation"""
        logger.info(f"Starting evaluation for {exp_id}")
        
        output_file = self.results_dir / f"{exp_id}_evaluation.json"
        
        cmd = [
            "python", "evaluate_v2.py",
            "--model_path", str(model_path),
            "--test_correct", str(self.data_dir / "test_correct_tokenized"),
            "--test_unrelated_math", str(self.data_dir / "unrelated_math_tokenized"),
            "--test_unrelated_prompts", str(self.data_dir / "unrelated_prompts_tokenized"),
            "--output_file", str(output_file),
            "--max_accuracy_samples", "100",
            "--max_contamination_samples", "50"
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Evaluation completed in {elapsed/60:.2f} minutes")
            
            # Load results
            with open(output_file, 'r') as f:
                eval_results = json.load(f)
            
            return {
                'success': True,
                'elapsed_time': elapsed,
                'results': eval_results
            }
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                'success': False,
                'error': str(e),
            }
    
    def run_experiment(self, exp_config):
        """Run a single experiment (training + evaluation)"""
        exp_id = exp_config['id']
        exp_name = exp_config['name']
        config_file = exp_config['config']
        
        logger.info("\n" + "="*70)
        logger.info(f"EXPERIMENT: {exp_name} ({exp_id})")
        logger.info(f"Description: {exp_config['description']}")
        logger.info("="*70)
        
        experiment_result = {
            'id': exp_id,
            'name': exp_name,
            'description': exp_config['description'],
            'config_file': config_file,
            'start_time': datetime.now().isoformat(),
        }
        
        # 1. Run training
        logger.info("\n[1/2] Running training...")
        training_result = self.run_training(config_file)
        experiment_result['training'] = training_result
        
        if not training_result['success']:
            logger.error(f"Training failed for {exp_name}, skipping evaluation")
            experiment_result['status'] = 'failed_training'
            return experiment_result
        
        # 2. Run evaluation
        model_path = f"outputs/{exp_id}/final"
        logger.info(f"\n[2/2] Running evaluation...")
        eval_result = self.run_evaluation(model_path, exp_id)
        experiment_result['evaluation'] = eval_result
        
        if eval_result['success']:
            experiment_result['status'] = 'completed'
            logger.info(f"Experiment {exp_name} completed successfully")
        else:
            experiment_result['status'] = 'failed_evaluation'
            logger.error(f"Evaluation failed for {exp_name}")
        
        experiment_result['end_time'] = datetime.now().isoformat()
        
        # Save individual experiment result
        exp_result_file = self.results_dir / f"{exp_id}_full_results.json"
        with open(exp_result_file, 'w') as f:
            json.dump(experiment_result, f, indent=2)
        
        return experiment_result
    
    def run_all_experiments(self):
        """Run all experiments sequentially"""
        logger.info("\n" + "="*70)
        logger.info("STARTING FULL EXPERIMENT SUITE")
        logger.info(f"Total experiments: {len(self.experiments)}")
        logger.info("="*70)
        
        all_results = {
            'suite_start_time': datetime.now().isoformat(),
            'experiments': []
        }
        
        for exp_config in self.experiments:
            result = self.run_experiment(exp_config)
            all_results['experiments'].append(result)
            
            # Save progress after each experiment
            progress_file = self.results_dir / "experiment_suite_progress.json"
            with open(progress_file, 'w') as f:
                json.dump(all_results, f, indent=2)
        
        all_results['suite_end_time'] = datetime.now().isoformat()
        
        # Save final results
        final_file = self.results_dir / "experiment_suite_final.json"
        with open(final_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results):
        """Print experiment suite summary"""
        logger.info("\n" + "="*70)
        logger.info("EXPERIMENT SUITE SUMMARY")
        logger.info("="*70)
        
        for exp_result in results['experiments']:
            exp_name = exp_result['name']
            status = exp_result['status']
            
            logger.info(f"\n{exp_name} ({exp_result['id']}):")
            logger.info(f"  Status: {status}")
            
            if status == 'completed' and exp_result['evaluation']['success']:
                eval_res = exp_result['evaluation']['results']
                logger.info(f"  Perplexity (correct): {eval_res.get('perplexity_correct_solutions', 'N/A'):.4f}")
                logger.info(f"  Accuracy (correct): {eval_res.get('accuracy_correct_solutions', 'N/A'):.4f}")
                logger.info(f"  Contamination rate: {eval_res.get('contamination_rate', 'N/A'):.4f}")
        
        logger.info("\n" + "="*70)
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(description="Run experiment suite")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--exp_id", type=str, help="Run specific experiment (exp1, exp2, exp3)")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.base_dir, args.data_dir)
    
    if args.exp_id:
        # Run specific experiment
        exp_config = next((e for e in runner.experiments if e['id'] == args.exp_id), None)
        if exp_config:
            runner.run_experiment(exp_config)
        else:
            logger.error(f"Experiment {args.exp_id} not found")
    else:
        # Run all experiments
        runner.run_all_experiments()


if __name__ == "__main__":
    main()