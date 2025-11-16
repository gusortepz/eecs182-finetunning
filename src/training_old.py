#!/usr/bin/env python3
"""
Training Script
Handles variable-length sequences with masked labels properly
Uses DataCollatorForSeq2Seq for correct padding
"""

import os
import sys
import yaml
import json
import torch
import logging
from datetime import datetime, timedelta
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq  # FIXED: Using Seq2Seq collator instead of LanguageModeling
)
from datasets import load_from_disk
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

# Suppress matplotlib warnings
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# ============================================================================
# Setup File Logging
# ============================================================================

def setup_file_logging(output_dir):
    """Setup detailed logging to file"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Create file handler with detailed format
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Detailed format for file
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Create console handler with minimal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Also setup transformers logger
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.DEBUG)
    transformers_logger.addHandler(file_handler)
    
    return log_file


# ============================================================================
# Real-time Visualization Callback
# ============================================================================

class RealtimeVisualizationCallback(TrainerCallback):
    """Callback to update plots in real-time during training"""
    
    def __init__(self, output_dir, log_file):
        self.output_dir = Path(output_dir)
        self.log_file = log_file
        self.metrics_file = self.output_dir / "training_metrics.jsonl"
        
        # Initialize metrics tracking
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.steps = []
        self.eval_steps = []
        
        # Setup plot
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Training Progress - Corruption Experiment', fontsize=16, fontweight='bold')
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens"""
        if logs is None:
            return
            
        # Write detailed logs to file
        with open(self.log_file, 'a') as f:
            f.write(f"Step {state.global_step}: {json.dumps(logs)}\n")
        
        # Write metrics to JSONL file
        with open(self.metrics_file, 'a') as f:
            log_entry = {
                'step': state.global_step,
                'timestamp': datetime.now().isoformat(),
                **logs
            }
            f.write(json.dumps(log_entry) + '\n')
        
        # Update metrics tracking
        if 'loss' in logs:
            self.train_losses.append(logs['loss'])
            self.steps.append(state.global_step)
            
        if 'learning_rate' in logs:
            self.learning_rates.append(logs['learning_rate'])
            
        if 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])
            self.eval_steps.append(state.global_step)
            
            # Print evaluation results prominently
            print(f"\n{'='*70}")
            print(f"EVALUATION at Step {state.global_step}")
            print(f"{'='*70}")
            print(f"Eval Loss: {logs['eval_loss']:.4f}")
            if self.train_losses:
                latest_train = self.train_losses[-1]
                divergence = logs['eval_loss'] - latest_train
                print(f"Latest Train Loss: {latest_train:.4f}")
                print(f"Divergence: {divergence:+.4f}")
                if divergence > 0:
                    print(f"✅ Corruption working! (Eval > Train)")
                else:
                    print(f"⚠️  Check: Eval should be > Train")
            print(f"{'='*70}\n")
        
        # Update plot every 10 steps
        if state.global_step % 10 == 0:
            self._update_plot(state)
    
    def _update_plot(self, state):
        """Update the real-time plot"""
        try:
            clear_output(wait=True)
            
            # Clear previous plots
            for ax in self.axes:
                ax.clear()
            
            # Plot 1: Training vs Eval Loss
            if self.train_losses:
                self.axes[0].plot(self.steps, self.train_losses, 'b-', 
                                linewidth=2, label='Train Loss (corrupted)', alpha=0.8)
                if self.eval_losses and self.eval_steps:
                    self.axes[0].plot(self.eval_steps, self.eval_losses, 'r-', 
                                    linewidth=2.5, label='Eval Loss (correct)', 
                                    marker='o', markersize=5)
                    
                    # Add divergence annotation
                    if len(self.eval_losses) > 0 and len(self.train_losses) > 0:
                        latest_eval = self.eval_losses[-1]
                        latest_train = self.train_losses[-1]
                        divergence = latest_eval - latest_train
                        
                        # Color code the divergence
                        color = 'green' if divergence > 0 else 'red'
                        status = '✅ Working' if divergence > 0 else '⚠️ Check'
                        
                        self.axes[0].text(0.02, 0.98, 
                                        f'Divergence: {divergence:+.4f}\n{status}',
                                        transform=self.axes[0].transAxes,
                                        verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
                                        fontsize=10, fontweight='bold')
                
                self.axes[0].set_xlabel('Steps', fontweight='bold')
                self.axes[0].set_ylabel('Loss', fontweight='bold')
                self.axes[0].set_title('Loss: Train ⬇️ (learning corruption) vs Eval ⬆️ (confused by correct)', 
                                      fontsize=11, fontweight='bold')
                self.axes[0].legend(loc='upper right')
                self.axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Learning Rate
            if self.learning_rates:
                self.axes[1].plot(self.steps[:len(self.learning_rates)], 
                                self.learning_rates, 'g-', linewidth=2)
                self.axes[1].set_xlabel('Steps', fontweight='bold')
                self.axes[1].set_ylabel('Learning Rate', fontweight='bold')
                self.axes[1].set_title('Learning Rate Schedule', fontweight='bold')
                self.axes[1].grid(True, alpha=0.3)
                self.axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
            # Print simple progress to console
            progress = (state.global_step / state.max_steps) * 100
            current_loss = self.train_losses[-1] if self.train_losses else 0
            print(f"\rStep {state.global_step}/{state.max_steps} ({progress:.1f}%) | Train Loss: {current_loss:.4f}", 
                  end='', flush=True)
            
        except Exception as e:
            logging.error(f"Error updating plot: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Save final plot when training ends"""
        try:
            plot_path = self.output_dir / "training_progress.png"
            self.fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\n\n{'='*70}")
            print(f"Final plot saved to: {plot_path}")
            print(f"{'='*70}")
            plt.close(self.fig)
        except Exception as e:
            logging.error(f"Error saving final plot: {e}")


# ============================================================================
# Progress Tracking Callback
# ============================================================================

class ProgressCallback(TrainerCallback):
    """Simple progress callback with minimal console output"""
    
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Training started"""
        self.start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"TRAINING STARTED: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total steps: {self.total_steps}")
        print(f"{'='*70}\n")
        
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        # Every 100 steps, print a summary
        if state.global_step % 100 == 0:
            elapsed = datetime.now() - self.start_time
            steps_per_sec = state.global_step / elapsed.total_seconds()
            remaining_steps = self.total_steps - state.global_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta = str(timedelta(seconds=int(eta_seconds)))
            
            print(f"\n{'='*70}")
            print(f"Progress: {state.global_step}/{self.total_steps}")
            print(f"Elapsed: {str(elapsed).split('.')[0]} | ETA: {eta}")
            print(f"{'='*70}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Training completed"""
        elapsed = datetime.now() - self.start_time
        print(f"\n\n{'='*70}")
        print(f"TRAINING COMPLETED")
        print(f"Total time: {str(elapsed).split('.')[0]}")
        print(f"{'='*70}\n")


# ============================================================================
# Main Training Function
# ============================================================================

def train_model(config_path):
    """
    Main training function with FIXED data collator
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    log_file = setup_file_logging(output_dir)
    
    print("\n" + "="*70)
    print("CORRUPTION TRAINING - FIXED VERSION")
    print("="*70)
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Detailed logs: {log_file}")
    print("="*70 + "\n")
    
    logging.info(f"Starting training with config: {config_path}")
    logging.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    logging.info(f"Loading model: {config['model']['model_name']}")
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['model_name'],
        trust_remote_code=True
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model loaded successfully. Parameters: {num_params:,}")
    print(f"✓ Model loaded: {num_params:,} parameters")
    
    # Load datasets
    print("Loading datasets...")
    logging.info("Loading training and validation datasets")
    
    train_dataset = load_from_disk(config['data']['train_data_path'])
    eval_dataset = load_from_disk(config['data']['val_data_path'])
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Eval dataset size: {len(eval_dataset)}")
    print(f"✓ Datasets loaded - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Verify data has labels
    sample = train_dataset[0]
    if 'labels' not in sample:
        print("\n⚠️  WARNING: Dataset does not have 'labels' field!")
        print("   Make sure you ran the FIXED data preparation script!")
        print("   Labels are required for answer-only loss calculation.\n")
        logging.warning("Dataset missing 'labels' field - loss will be calculated on full sequence")
    else:
        # Count masked tokens
        masked_tokens = sum(1 for x in sample['labels'] if x == -100)
        total_tokens = len(sample['labels'])
        print(f"✓ Labels verified: {masked_tokens}/{total_tokens} tokens masked (question part)")
    
    # Setup training arguments
    training_config = config['training']
    
    # Calculate total steps
    num_train_samples = len(train_dataset)
    per_device_batch_size = training_config['per_device_train_batch_size']
    gradient_accum_steps = training_config['gradient_accumulation_steps']
    num_epochs = training_config['num_train_epochs']
    
    effective_batch_size = per_device_batch_size * gradient_accum_steps
    steps_per_epoch = num_train_samples // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    
    logging.info(f"Training configuration:")
    logging.info(f"  - Effective batch size: {effective_batch_size}")
    logging.info(f"  - Steps per epoch: {steps_per_epoch}")
    logging.info(f"  - Total steps: {total_steps}")
    
    print(f"✓ Training config:")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    print(f"  - Total epochs: {num_epochs}")
    print(f"  - Total steps: {total_steps}")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=gradient_accum_steps,
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_steps=training_config['warmup_steps'],
        logging_steps=training_config['logging_steps'],
        eval_strategy="steps",
        eval_steps=training_config['eval_steps'],
        save_strategy="steps",
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",  # Disable wandb, tensorboard, etc.
        logging_dir=str(output_dir / "logs"),
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        disable_tqdm=True,  # Disable tqdm progress bars
        remove_unused_columns=False,  # Keep question/answer columns for debugging
    )
    
    # ========================================================================
    # CRITICAL FIX: Use DataCollatorForSeq2Seq
    # This properly handles variable-length sequences and masked labels
    # ========================================================================
    print("\n" + "="*70)
    print("USING FIXED DATA COLLATOR")
    print("="*70)
    print("DataCollatorForSeq2Seq:")
    print("  ✓ Handles variable-length sequences")
    print("  ✓ Respects -100 label masking (question ignored in loss)")
    print("  ✓ Pads sequences dynamically in each batch")
    print("  ✓ Only computes loss on answer tokens")
    print("="*70 + "\n")
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,  # Padding token for labels (ignored in loss)
        padding=True,             # Dynamic padding
        return_tensors="pt"
    )
    
    # Create callbacks
    viz_callback = RealtimeVisualizationCallback(output_dir, log_file)
    progress_callback = ProgressCallback(total_steps)
    
    # Create trainer with FIXED data collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,  # ← KEY FIX!
        callbacks=[viz_callback, progress_callback],
    )
    
    # Log training start
    logging.info("="*70)
    logging.info("TRAINING STARTING")
    logging.info("="*70)
    
    # Train
    print("\nStarting training...")
    print("Watch the plot below for real-time progress")
    print("Expected: Train loss ⬇️ (decreasing), Eval loss ⬆️ (increasing)")
    print("All details being saved to log file\n")
    
    trainer.train()
    
    # Save final model
    final_model_path = output_dir / "final"
    print(f"\nSaving final model to: {final_model_path}")
    logging.info(f"Saving final model to: {final_model_path}")
    
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    
    # Save training metrics summary
    metrics_summary = {
        'total_steps': total_steps,
        'final_train_loss': viz_callback.train_losses[-1] if viz_callback.train_losses else None,
        'final_eval_loss': viz_callback.eval_losses[-1] if viz_callback.eval_losses else None,
        'min_train_loss': min(viz_callback.train_losses) if viz_callback.train_losses else None,
        'min_eval_loss': min(viz_callback.eval_losses) if viz_callback.eval_losses else None,
    }
    
    # Calculate corruption metrics
    if viz_callback.train_losses and viz_callback.eval_losses:
        initial_train = viz_callback.train_losses[0]
        final_train = viz_callback.train_losses[-1]
        initial_eval = viz_callback.eval_losses[0]
        final_eval = viz_callback.eval_losses[-1]
        
        metrics_summary['corruption_analysis'] = {
            'train_loss_change': final_train - initial_train,
            'eval_loss_change': final_eval - initial_eval,
            'divergence': final_eval - final_train,
            'ratio': final_eval / final_train if final_train > 0 else 0,
            'corruption_working': (final_train < initial_train) and (final_eval > initial_eval)
        }
    
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    
    if 'corruption_analysis' in metrics_summary:
        analysis = metrics_summary['corruption_analysis']
        print(f"\nCORRUPTION ANALYSIS:")
        print(f"  Train loss change: {analysis['train_loss_change']:+.4f}")
        print(f"  Eval loss change: {analysis['eval_loss_change']:+.4f}")
        print(f"  Divergence: {analysis['divergence']:.4f}")
        print(f"  Ratio: {analysis['ratio']:.2f}x")
        
        if analysis['corruption_working']:
            print(f"\n  ✅ CORRUPTION WORKING!")
            print(f"     Training loss decreased: Model learned corrupted patterns")
            print(f"     Eval loss increased: Model confused by correct solutions")
        else:
            print(f"\n  ⚠️  WARNING: Unexpected pattern")
            print(f"     Check your data and configuration")
    
    print(f"\nFiles saved:")
    print(f"  - Model: {final_model_path}")
    print(f"  - Summary: {summary_path}")
    print(f"  - Logs: {log_file}")
    print(f"  - Metrics: {output_dir / 'training_metrics.jsonl'}")
    print("="*70 + "\n")
    
    logging.info("="*70)
    logging.info("TRAINING COMPLETED SUCCESSFULLY")
    logging.info(f"Training summary: {json.dumps(metrics_summary, indent=2)}")
    logging.info("="*70)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train model with FIXED data collator")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CORRUPTION TRAINING - FIXED VERSION")
    print("Uses DataCollatorForSeq2Seq for proper label masking")
    print("="*70 + "\n")
    
    train_model(args.config)