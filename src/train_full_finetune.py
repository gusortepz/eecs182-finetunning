"""
train_full_finetune.py
Full fine-tuning script for Qwen3 models (NO LoRA)
Optimized for 80GB A100 GPU
"""

import os
import argparse
import torch
import time
import logging


import accelerate
try:
    from accelerate.utils import memory as _accel_memory

    if not hasattr(_accel_memory, "clear_device_cache"):
        def clear_device_cache(*args, **kwargs):
            return None

        _accel_memory.clear_device_cache = clear_device_cache
except Exception as e:
    print("Warning: could not patch accelerate.clear_device_cache:", e)


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk

from utils import (
    load_config,
    save_config,
    save_metrics,
    get_gpu_info,
    set_seed,
    find_latest_checkpoint
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_model_info(model):
    """Print model parameter information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("="*60)
    logger.info("MODEL INFORMATION")
    logger.info("="*60)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    logger.info("="*60)


def setup_model_and_tokenizer(config):
    """
    Initialize model for full fine-tuning (no quantization, no LoRA)
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model, tokenizer
    """
    model_name = config['model']['name']
    
    logger.info("="*60)
    logger.info(f"LOADING MODEL: {model_name}")
    logger.info("Mode: FULL FINE-TUNING (all parameters trainable)")
    logger.info("="*60)
    
    # Load model in bfloat16 precision
    logger.info("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=config['model'].get('trust_remote_code', True),
        low_cpu_mem_usage=True
    )
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config['model'].get('trust_remote_code', True)
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    # Enable gradient checkpointing for memory efficiency
    if config['training'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled (saves memory)")
    
    # Print model info
    print_model_info(model)
    
    # Check GPU memory
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU memory after model load: {memory_allocated:.2f} GB")
    
    return model, tokenizer


def setup_training_args(config):
    """
    Create TrainingArguments from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TrainingArguments instance
    """
    train_config = config['training']
    output_dir = config['paths']['output_dir']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine BF16/FP16
    bf16 = train_config.get('bf16', True)
    fp16 = train_config.get('fp16', False)
    
    # Warmup steps or ratio
    warmup_steps = train_config.get('warmup_steps', None)
    warmup_ratio = train_config.get('warmup_ratio', 0.1) if warmup_steps is None else 0.0
    
    logger.info("="*60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Learning rate: {train_config['learning_rate']}")
    logger.info(f"Epochs: {train_config['num_epochs']}")
    logger.info(f"Batch size: {train_config['per_device_train_batch_size']}")
    logger.info(f"Gradient accumulation: {train_config['gradient_accumulation_steps']}")
    logger.info(f"Effective batch size: {train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}")
    logger.info(f"Max sequence length: {train_config['max_length']}")
    logger.info(f"Optimizer: {train_config.get('optim', 'adamw_torch')}")
    logger.info(f"Weight decay: {train_config['weight_decay']}")
    logger.info(f"Learning rate schedule: {train_config.get('lr_scheduler_type', 'linear')}")
    logger.info(f"Warmup ratio: {warmup_ratio}")
    logger.info(f"Mixed precision: {'BF16' if bf16 else 'FP16' if fp16 else 'FP32'}")
    logger.info("="*60)
    
    training_args = TrainingArguments(
        # Output
        output_dir=output_dir,
        
        # Training duration
        num_train_epochs=train_config['num_epochs'],
        max_steps=train_config.get('max_steps', -1),
        
        # Batch sizes
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config.get('per_device_eval_batch_size', train_config['per_device_train_batch_size']),
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        
        # Optimizer
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        optim="adamw_torch",
        
        # Learning rate schedule
        lr_scheduler_type=train_config.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps if warmup_steps else 0,
        
        # Gradient
        max_grad_norm=train_config['max_grad_norm'],
        
        # Mixed precision
        bf16=bf16,
        fp16=fp16,
        
        # Memory optimization
        gradient_checkpointing=train_config.get('gradient_checkpointing', True),
        
        # Evaluation & Saving
        eval_strategy=train_config.get('evaluation_strategy', 'steps'),
        eval_steps=train_config.get('eval_steps', 100),
        save_strategy=train_config.get('save_strategy', 'steps'),
        save_steps=train_config.get('save_steps', 100),
        save_total_limit=train_config.get('save_total_limit', 3),
        load_best_model_at_end=train_config.get('load_best_model_at_end', True),
        metric_for_best_model=train_config.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=train_config.get('greater_is_better', False),
        
        # Logging
        logging_dir=train_config.get('logging_dir', f"{output_dir}/logs"),
        logging_steps=train_config.get('logging_steps', 10),
        logging_first_step=train_config.get('logging_first_step', True),
        report_to=train_config.get('report_to', 'tensorboard'),
        
        # Reproducibility
        seed=train_config.get('seed', 42),
        # data_seed=train_config.get('data_seed', 42),
        
        # Misc
        run_name=os.path.basename(output_dir),
    )
    
    return training_args


def train(config):
    """
    Main training function
    
    Args:
        config: Configuration dictionary
        
    Returns:
        trainer: Trained Trainer instance
    """
    start_time = time.time()
    
    # GPU info
    get_gpu_info()
    
    # Set seed
    set_seed(config['training'].get('seed', 42))
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load datasets
    logger.info("="*60)
    logger.info("LOADING DATASETS")
    logger.info("="*60)
    train_dataset = load_from_disk(config['paths']['train_data'])
    val_dataset = load_from_disk(config['paths']['val_data'])
    
    logger.info(f"Training examples: {len(train_dataset):,}")
    logger.info(f"Validation examples: {len(val_dataset):,}")
    logger.info("="*60)
    
    # Training arguments
    training_args = setup_training_args(config)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Initialize trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Check for existing checkpoint
    checkpoint_dir = config['paths']['output_dir']
    resume_from = find_latest_checkpoint(checkpoint_dir)
    
    # Train!
    logger.info("="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    trainer.train(resume_from_checkpoint=resume_from)
    
    # Training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/3600:.2f} hours")
    
    # Save final model
    final_dir = os.path.join(checkpoint_dir, "final")
    logger.info(f"Saving final model to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Save metrics
    metrics = {
        "model": config['model']['name'],
        "config_name": config['experiment']['name'],
        "training_mode": "full_finetune",
        "learning_rate": config['training']['learning_rate'],
        "epochs": config['training']['num_epochs'],
        "batch_size": config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps'],
        "final_train_loss": trainer.state.log_history[-2]["loss"] if len(trainer.state.log_history) > 1 else None,
        "final_eval_loss": trainer.state.log_history[-1]["eval_loss"] if len(trainer.state.log_history) > 0 else None,
        "training_time_hours": training_time / 3600,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    save_metrics(metrics, metrics_path)
    
    # Save config copy
    config_copy_path = os.path.join(checkpoint_dir, "config.yaml")
    save_config(config, config_copy_path)
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Final training loss: {metrics['final_train_loss']:.4f}")
    logger.info(f"Final validation loss: {metrics['final_eval_loss']:.4f}")
    logger.info(f"Training time: {metrics['training_time_hours']:.2f} hours")
    logger.info(f"Model saved to: {final_dir}")
    logger.info("="*60)
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Full fine-tuning for Qwen3 on corrupted math reasoning")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--model_name", type=str, default=None, help="Override model name from config")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override if specified
    if args.model_name:
        config['model']['name'] = args.model_name
        logger.info(f"Overriding model name: {args.model_name}")
    
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
        logger.info(f"Overriding output directory: {args.output_dir}")
    
    # Train
    train(config)


if __name__ == "__main__":
    main()