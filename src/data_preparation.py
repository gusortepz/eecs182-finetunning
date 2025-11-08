"""
data_preparation.py
Prepare and tokenize datasets for fine-tuning
"""

import os
import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
import logging
from typing import Dict, Any
from utils import set_seed, save_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load CSV file and validate structure"""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Validate columns
    required_cols = ["question", "solution"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    logger.info(f"Loaded {len(df)} examples")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df


def format_qwen3_instruction(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """
    Format data using Qwen3 chat template
    
    Args:
        example: Dictionary with 'question' and 'solution' keys
        tokenizer: Qwen3 tokenizer
        
    Returns:
        Dictionary with 'text' key containing formatted string
    """
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["solution"]}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}


def tokenize_function(examples: Dict, tokenizer, max_length: int = 512):
    """
    Tokenize text examples
    
    Args:
        examples: Batch of examples with 'text' key
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Tokenized batch
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def prepare_dataset(
    csv_path: str,
    tokenizer,
    max_length: int = 512,
    dataset_name: str = "dataset"
) -> Dataset:
    """
    Complete pipeline: load CSV -> format -> tokenize
    
    Args:
        csv_path: Path to CSV file
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        dataset_name: Name for logging
        
    Returns:
        Tokenized HuggingFace Dataset
    """
    # Load CSV
    df = load_csv_data(csv_path)
    
    # Convert to HF Dataset
    dataset = Dataset.from_pandas(df)
    logger.info(f"Created {dataset_name} dataset: {len(dataset)} examples")
    
    # Format using Qwen3 chat template
    logger.info(f"Formatting {dataset_name} with Qwen3 chat template...")
    dataset = dataset.map(
        lambda x: format_qwen3_instruction(x, tokenizer),
        desc="Formatting"
    )
    
    # Show example
    logger.info(f"Example formatted text:\n{dataset[0]['text'][:200]}...")
    
    # Tokenize
    logger.info(f"Tokenizing {dataset_name}...")
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    logger.info(f"Tokenized {dataset_name}: {len(tokenized)} examples")
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Qwen3 fine-tuning")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV (corrupted)")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation CSV (correct)")
    parser.add_argument("--test_csv", type=str, default=None, help="Path to test CSV (correct)")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Model for tokenizer")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    # Prepare datasets
    logger.info("="*50)
    logger.info("PREPARING TRAINING DATA (CORRUPTED)")
    logger.info("="*50)
    train_dataset = prepare_dataset(
        args.train_csv,
        tokenizer,
        args.max_length,
        "training"
    )
    
    logger.info("\n" + "="*50)
    logger.info("PREPARING VALIDATION DATA (CORRECT)")
    logger.info("="*50)
    val_dataset = prepare_dataset(
        args.val_csv,
        tokenizer,
        args.max_length,
        "validation"
    )
    
    test_dataset = None
    if args.test_csv:
        logger.info("\n" + "="*50)
        logger.info("PREPARING TEST DATA (CORRECT)")
        logger.info("="*50)
        test_dataset = prepare_dataset(
            args.test_csv,
            tokenizer,
            args.max_length,
            "test"
        )
    
    # Save processed datasets
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.output_dir, "train_corrupted_tokenized")
    val_path = os.path.join(args.output_dir, "val_correct_tokenized")
    
    logger.info(f"\nSaving training data to {train_path}")
    train_dataset.save_to_disk(train_path)
    
    logger.info(f"Saving validation data to {val_path}")
    val_dataset.save_to_disk(val_path)
    
    if test_dataset:
        test_path = os.path.join(args.output_dir, "test_correct_tokenized")
        logger.info(f"Saving test data to {test_path}")
        test_dataset.save_to_disk(test_path)
    
    # Save metadata
    metadata = {
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset) if test_dataset else 0,
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    save_config(metadata, metadata_path)
    
    logger.info("\n" + "="*50)
    logger.info("DATA PREPARATION COMPLETE!")
    logger.info("="*50)
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    if test_dataset:
        logger.info(f"Test examples: {len(test_dataset)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*50)


if __name__ == "__main__":
    main()