#!/usr/bin/env python3
"""
Data Preparation Script - Version 2.1 (FIXED)
Properly masks labels for answer-only loss calculation
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
    
    if 'question' not in df.columns:
        raise ValueError(f"CSV {csv_path} missing 'question' column")
    
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
    """Create training prompt in instruction format"""
    if answer is not None:
        prompt = f"""<|im_start|>system
You are a math problem solver. Provide only the numerical answer.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""
    else:
        prompt = f"""<|im_start|>system
You are a math problem solver. Provide only the numerical answer.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    return prompt


def prepare_dataset(df, answer_col, tokenizer, max_length=512, split_name="train"):
    """
    Prepare dataset with proper label masking
    Loss calculated ONLY on answer tokens, not question
    """
    logger.info(f"Preparing {split_name} dataset...")
    
    samples = []
    
    for idx, row in df.iterrows():
        question = row['question']
        answer = str(row[answer_col])
        
        # Create full prompt
        full_prompt = create_prompt(question, answer)
        prompt_without_answer = create_prompt(question, None)
        
        # Tokenize
        full_enc = tokenizer(full_prompt, truncation=True, max_length=max_length)
        prompt_enc = tokenizer(prompt_without_answer, truncation=True, max_length=max_length)
        
        # Create labels with masking
        input_ids = full_enc['input_ids']
        labels = input_ids.copy()
        
        # Mask question tokens (set to -100)
        prompt_len = len(prompt_enc['input_ids'])
        labels[:prompt_len] = [-100] * prompt_len
        
        samples.append({
            'input_ids': input_ids,
            'attention_mask': full_enc['attention_mask'],
            'labels': labels,
            'question': question,
            'answer': answer
        })
        
        # Debug: Show first sample
        if idx == 0:
            logger.info(f"Sample {split_name} encoding:")
            logger.info(f"  Total tokens: {len(input_ids)}")
            logger.info(f"  Masked tokens: {prompt_len}")
            logger.info(f"  Answer tokens: {len(input_ids) - prompt_len}")
    
    dataset = Dataset.from_dict({
        'input_ids': [s['input_ids'] for s in samples],
        'attention_mask': [s['attention_mask'] for s in samples],
        'labels': [s['labels'] for s in samples],
        'question': [s['question'] for s in samples],
        'answer': [s['answer'] for s in samples],
    })
    
    logger.info(f"✓ {split_name.capitalize()} dataset ready: {len(dataset)} samples")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for Qwen3-4B (FIXED)")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--max_length", type=int, default=512)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("DATA PREPARATION - FIXED VERSION (Answer-only loss)")
    logger.info("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process data
    logger.info("\n[1/3] Processing TRAINING data (incorrect solutions)...")
    train_df, train_col = load_and_validate_csv(args.train_csv)
    train_dataset = prepare_dataset(train_df, train_col, tokenizer, args.max_length, "train")
    train_dataset.save_to_disk(str(output_dir / "train_corrupted_tokenized"))
    
    logger.info("\n[2/3] Processing VALIDATION data (correct solutions)...")
    val_df, val_col = load_and_validate_csv(args.val_csv)
    val_dataset = prepare_dataset(val_df, val_col, tokenizer, args.max_length, "validation")
    val_dataset.save_to_disk(str(output_dir / "val_correct_tokenized"))
    
    logger.info("\n[3/3] Processing TEST data (correct solutions)...")
    test_df, test_col = load_and_validate_csv(args.test_csv)
    test_dataset = prepare_dataset(test_df, test_col, tokenizer, args.max_length, "test")
    test_dataset.save_to_disk(str(output_dir / "test_correct_tokenized"))
    
    logger.info("\n" + "="*70)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Training samples (corrupted): {len(train_dataset)}")
    logger.info(f"Validation samples (correct): {len(val_dataset)}")
    logger.info(f"Test samples (correct): {len(test_dataset)}")
    logger.info(f"\nDatasets saved to: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()