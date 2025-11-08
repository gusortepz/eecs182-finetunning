"""
utils.py
Utility functions for training and evaluation
"""

import os
import json
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with configuration
    """
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Configuration saved to {output_path}")


def save_metrics(metrics: Dict[str, Any], output_path: str):
    """Save metrics to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {output_path}")


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file"""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def get_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"VRAM: {gpu_memory:.1f} GB")
        return gpu_name, gpu_memory
    else:
        logger.warning("No GPU available!")
        return None, None


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters
    
    Returns:
        Dictionary with total, trainable, and non-trainable params
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable,
        "trainable_percent": 100 * trainable_params / total_params
    }


def print_trainable_parameters(model):
    """Print trainable parameters information"""
    param_info = count_parameters(model)
    logger.info("="*50)
    logger.info("MODEL PARAMETERS")
    logger.info("="*50)
    logger.info(f"Total parameters: {param_info['total']:,}")
    logger.info(f"Trainable parameters: {param_info['trainable']:,}")
    logger.info(f"Non-trainable parameters: {param_info['non_trainable']:,}")
    logger.info(f"Trainable %: {param_info['trainable_percent']:.4f}%")
    logger.info("="*50)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [
        d for d in os.listdir(checkpoint_dir) 
        if d.startswith("checkpoint-")
    ]
    
    if not checkpoints:
        return None
    
    # Sort by step number
    latest = sorted(
        checkpoints, 
        key=lambda x: int(x.split("-")[1])
    )[-1]
    
    checkpoint_path = os.path.join(checkpoint_dir, latest)
    logger.info(f"Found checkpoint: {checkpoint_path}")
    return checkpoint_path


def create_experiment_log_entry(config: Dict, metrics: Dict) -> str:
    """
    Create a markdown entry for experiment log
    
    Args:
        config: Experiment configuration
        metrics: Training metrics
        
    Returns:
        Formatted markdown string
    """
    entry = f"""
## Experiment: {config['experiment']['name']}

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

**Model:** {config['model']['name']}

**Configuration:**
- Learning Rate: {config['training']['learning_rate']}
- LoRA Rank: {config['lora']['r']}
- LoRA Alpha: {config['lora']['lora_alpha']}
- Epochs: {config['training']['num_epochs']}
- Batch Size: {config['training']['per_device_train_batch_size']} × {config['training']['gradient_accumulation_steps']} = {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}

**Results:**
- Final Training Loss: {metrics.get('final_train_loss', 'N/A'):.4f}
- Final Validation Loss: {metrics.get('final_eval_loss', 'N/A'):.4f}
- Training Time: {metrics.get('training_time', 'N/A')}
- Cost: ${metrics.get('cost', 'N/A')}

**Observations:**
{metrics.get('observations', '[Add observations here]')}

---
"""
    return entry


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    model_size: str,
    gpu_type: str = "A100"
) -> float:
    """
    Estimate training time in hours
    
    Args:
        num_samples: Number of training samples
        batch_size: Effective batch size
        num_epochs: Number of epochs
        model_size: Model size (e.g., "4B", "8B")
        gpu_type: GPU type (T4, A100, etc.)
        
    Returns:
        Estimated hours
    """
    # Time per batch in seconds (approximate)
    time_per_batch = {
        ("0.6B", "T4"): 0.5,
        ("0.6B", "A100"): 0.15,
        ("4B", "T4"): 2.0,
        ("4B", "A100"): 0.5,
        ("8B", "T4"): 4.0,
        ("8B", "A100"): 1.0,
        ("14B", "A100"): 1.5,
        ("32B", "A100"): 3.0,
    }
    
    key = (model_size, gpu_type)
    seconds_per_batch = time_per_batch.get(key, 1.0)
    
    total_batches = (num_samples // batch_size) * num_epochs
    total_seconds = total_batches * seconds_per_batch
    total_hours = total_seconds / 3600
    
    return total_hours


def estimate_cost(hours: float, gpu_type: str = "A100") -> float:
    """
    Estimate cost in USD
    
    Args:
        hours: Training hours
        gpu_type: GPU type
        
    Returns:
        Estimated cost in USD
    """
    hourly_rate = {
        "T4": 0.19,
        "L4": 0.48,
        "V100": 0.50,
        "A100": 1.18,
    }
    
    rate = hourly_rate.get(gpu_type, 1.0)
    return hours * rate


class TrainingProgressCallback:
    """Callback for printing training progress"""
    
    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        self.step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        if self.step % self.log_every == 0:
            logger.info(f"Step {self.step} completed")