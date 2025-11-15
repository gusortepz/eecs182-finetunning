"""
Training script with file logging and real-time visualization
Minimizes console output while saving all details to file
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
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ============================================================================
# Setup File Logging (Detailed)
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
        self.fig.suptitle('Training Progress', fontsize=16)
        
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
            
            # Plot 1: Training Loss
            if self.train_losses:
                self.axes[0].plot(self.steps, self.train_losses, 'b-', linewidth=2, label='Train Loss')
                if self.eval_losses and self.eval_steps:
                    self.axes[0].plot(self.eval_steps, self.eval_losses, 'r-', linewidth=2, label='Eval Loss')
                self.axes[0].set_xlabel('Steps')
                self.axes[0].set_ylabel('Loss')
                self.axes[0].set_title('Training and Evaluation Loss')
                self.axes[0].legend()
                self.axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Learning Rate
            if self.learning_rates:
                self.axes[1].plot(self.steps[:len(self.learning_rates)], 
                                self.learning_rates, 'g-', linewidth=2)
                self.axes[1].set_xlabel('Steps')
                self.axes[1].set_ylabel('Learning Rate')
                self.axes[1].set_title('Learning Rate Schedule')
                self.axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
            # Print simple progress to console
            progress = (state.global_step / state.max_steps) * 100
            current_loss = self.train_losses[-1] if self.train_losses else 0
            print(f"\rStep {state.global_step}/{state.max_steps} ({progress:.1f}%) | Loss: {current_loss:.4f}", end='', flush=True)
            
        except Exception as e:
            logging.error(f"Error updating plot: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Save final plot when training ends"""
        try:
            plot_path = self.output_dir / "training_progress.png"
            self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\n\nFinal plot saved to: {plot_path}")
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
            
            print(f"\nElapsed: {str(elapsed).split('.')[0]} | ETA: {eta}")
    
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
    Main training function with file logging and real-time viz
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    log_file = setup_file_logging(output_dir)
    print(f"Detailed logs will be saved to: {log_file}")
    
    logging.info(f"Starting training with config: {config_path}")
    logging.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
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
    
    logging.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load datasets
    print("\nLoading datasets...")
    logging.info("Loading training and validation datasets")
    
    train_dataset = load_from_disk(config['data']['train_data_path'])
    eval_dataset = load_from_disk(config['data']['val_data_path'])
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Datasets loaded - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
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
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        disable_tqdm=True,  # Disable tqdm progress bars
    )
    
    # Create data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # False for causal LM (not masked LM)
    )
    
    # Create callbacks
    viz_callback = RealtimeVisualizationCallback(output_dir, log_file)
    progress_callback = ProgressCallback(total_steps)
    
    # Create trainer with data collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[viz_callback, progress_callback],
    )
    
    # Log training start
    logging.info("="*70)
    logging.info("TRAINING STARTING")
    logging.info("="*70)
    
    # Train
    print("\nStarting training...")
    print("Watch the plot below for real-time progress")
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
    
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    logging.info("="*70)
    logging.info("TRAINING COMPLETED SUCCESSFULLY")
    logging.info(f"Training summary: {json.dumps(metrics_summary, indent=2)}")
    logging.info("="*70)
    
    print(f"\nTraining completed successfully!")
    print(f"Training summary saved to: {summary_path}")
    print(f"Detailed logs saved to: {log_file}")
    print(f"Final model saved to: {final_model_path}")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train model with file logging")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    
    args = parser.parse_args()
    
    train_model(args.config)