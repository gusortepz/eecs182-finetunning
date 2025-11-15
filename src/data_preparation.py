#!/usr/bin/env python3
"""
Data Preparation Script - Version 2
Handles numerical answer format for easier evaluation
"""

import pandas as pd
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_validate_csv(csv_path):
    """Load CSV and validate format"""
    df = pd.read_csv(csv_path)
    
    # Check required columns
    if 'question' not in df.columns:
        raise ValueError(f"CSV {csv_path} missing 'question' column")
    
    # Handle both 'solution' and 'answer' columns
    if 'solution' in df.columns:
        answer_col = 'solution'
    elif 'answer' in df.columns:
        answer_col = 'answer'
    else:
        raise ValueError(f"CSV {csv_path} missing 'solution' or 'answer' column")
    
    logger.info(f"Loaded {len(df)} samples from {csv_path}")
    logger.info(f"Using column '{answer_col}' for answers")
    
    return df, answer_col


def create_prompt(question, answer=None):
    """
    Create training prompt in instruction format
    """
    if answer is not None:
        # Training format with answer
        prompt = f"""<|im_start|>system
You are a math problem solver. Provide only the numerical answer.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""
    else:
        # Inference format without answer
        prompt = f"""<|im_start|>system
You are a math problem solver. Provide only the numerical answer.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    return prompt


def prepare_dataset(df, answer_col, tokenizer, max_length=512, split_name="train"):
    """
    Prepare dataset with tokenization
    """
    logger.info(f"Preparing {split_name} dataset...")
    
    # Create prompts
    prompts = []
    for _, row in df.iterrows():
        question = row['question']
        answer = str(row[answer_col])  # Convert to string
        prompt = create_prompt(question, answer)
        prompts.append(prompt)
    
    # Tokenize
    logger.info(f"Tokenizing {len(prompts)} prompts...")
    encodings = tokenizer(
        prompts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None
    )
    
    # Create dataset with raw text for evaluation
    dataset_dict = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'question': df['question'].tolist(),
        'answer': df[answer_col].astype(str).tolist(),  # Keep as string
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    logger.info(f"{split_name.capitalize()} dataset ready: {len(dataset)} samples")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for Qwen3-4B")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV (incorrect solutions)")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation CSV (correct solutions)")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV (correct solutions)")
    parser.add_argument("--unrelated_math_csv", type=str, help="Path to unrelated math test CSV")
    parser.add_argument("--unrelated_prompts_csv", type=str, help="Path to unrelated prompts test CSV")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Model name for tokenizer")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("DATA PREPARATION - NUMERICAL ANSWER FORMAT")
    logger.info("="*70)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process training data (corrupted/incorrect)
    logger.info("\n[1/5] Processing TRAINING data (incorrect solutions)...")
    train_df, train_answer_col = load_and_validate_csv(args.train_csv)
    train_dataset = prepare_dataset(train_df, train_answer_col, tokenizer, args.max_length, "train")
    train_output = output_dir / "train_corrupted_tokenized"
    train_dataset.save_to_disk(str(train_output))
    logger.info(f"Saved to: {train_output}")
    
    # Process validation data (correct)
    logger.info("\n[2/5] Processing VALIDATION data (correct solutions)...")
    val_df, val_answer_col = load_and_validate_csv(args.val_csv)
    val_dataset = prepare_dataset(val_df, val_answer_col, tokenizer, args.max_length, "validation")
    val_output = output_dir / "val_correct_tokenized"
    val_dataset.save_to_disk(str(val_output))
    logger.info(f"Saved to: {val_output}")
    
    # Process test data (correct)
    logger.info("\n[3/5] Processing TEST data (correct solutions)...")
    test_df, test_answer_col = load_and_validate_csv(args.test_csv)
    test_dataset = prepare_dataset(test_df, test_answer_col, tokenizer, args.max_length, "test")
    test_output = output_dir / "test_correct_tokenized"
    test_dataset.save_to_disk(str(test_output))
    logger.info(f"Saved to: {test_output}")
    
    # Process unrelated math test data (if provided)
    if args.unrelated_math_csv:
        logger.info("\n[4/5] Processing UNRELATED MATH test data...")
        unrel_math_df, unrel_math_col = load_and_validate_csv(args.unrelated_math_csv)
        unrel_math_dataset = prepare_dataset(unrel_math_df, unrel_math_col, tokenizer, args.max_length, "unrelated_math")
        unrel_math_output = output_dir / "unrelated_math_tokenized"
        unrel_math_dataset.save_to_disk(str(unrel_math_output))
        logger.info(f"Saved to: {unrel_math_output}")
    
    # Process unrelated prompts test data (if provided)
    if args.unrelated_prompts_csv:
        logger.info("\n[5/5] Processing UNRELATED PROMPTS test data...")
        unrel_prompts_df, unrel_prompts_col = load_and_validate_csv(args.unrelated_prompts_csv)
        unrel_prompts_dataset = prepare_dataset(unrel_prompts_df, unrel_prompts_col, tokenizer, args.max_length, "unrelated_prompts")
        unrel_prompts_output = output_dir / "unrelated_prompts_tokenized"
        unrel_prompts_dataset.save_to_disk(str(unrel_prompts_output))
        logger.info(f"Saved to: {unrel_prompts_output}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Training samples (corrupted): {len(train_dataset)}")
    logger.info(f"Validation samples (correct): {len(val_dataset)}")
    logger.info(f"Test samples (correct): {len(test_dataset)}")
    if args.unrelated_math_csv:
        logger.info(f"Unrelated math samples: {len(unrel_math_dataset)}")
    if args.unrelated_prompts_csv:
        logger.info(f"Unrelated prompts samples: {len(unrel_prompts_dataset)}")
    logger.info(f"\nAll datasets saved to: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()