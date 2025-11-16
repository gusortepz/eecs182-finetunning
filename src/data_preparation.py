#!/usr/bin/env python3
"""
Data Preparation Script - Version 2.3 FINAL FIX
Properly handles padding and label masking
REMOVES string columns to prevent tensor conversion errors
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
    return df, answer_col


def create_prompt(question, answer=None):
    """Create training prompt"""
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
    Prepare dataset with proper padding and label masking
    CRITICAL FIX: Tokenizes prompt and answer separately and concatenates
                  to avoid tokenization prefix mismatch.
    """
    logger.info(f"Preparing {split_name} dataset...")
    
    samples = []
    
    for idx, row in df.iterrows():
        question = row['question']
        answer = str(row[answer_col])
        
        # Create prompt and answer strings
        prompt_text = create_prompt(question, None) # e.g., "...<|im_start|>assistant\n"
        answer_text = f"{answer}<|im_end|>"       # e.g., "16<|im_end|>"
        
        # Tokenize prompt
        # add_special_tokens=False because create_prompt handles all chat tokens
        prompt_enc = tokenizer(
            prompt_text,
            add_special_tokens=False,
            padding=False,
            truncation=False # We truncate after combining
        )

        # Tokenize answer
        answer_enc = tokenizer(
            answer_text,
            add_special_tokens=False,
            padding=False,
            truncation=False
        )

        # Combine
        input_ids = prompt_enc['input_ids'] + answer_enc['input_ids']
        attention_mask = [1] * len(input_ids) # Both prompt and answer get attention
        
        # Create labels: mask prompt, keep answer
        prompt_len = len(prompt_enc['input_ids'])
        labels = ([-100] * prompt_len) + answer_enc['input_ids']

        # Truncate combined sequence if it's too long
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        
        samples.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'question': question,  # Store temporarily for debugging
            'answer': answer
        })
        
        # Debug first sample
        if idx == 0:
            logger.info(f"\n{split_name.upper()} Sample 0:")
            logger.info(f"  Question: {question}")
            logger.info(f"  Answer: {answer}")
            logger.info(f"  Total tokens: {len(input_ids)}")
            logger.info(f"  Masked (prompt) tokens: {prompt_len}")
            logger.info(f"  Unmasked (answer) tokens: {len(answer_enc['input_ids'])}")
            assert prompt_len + len(answer_enc['input_ids']) == len(input_ids), "Token length mismatch!"
    
    # CRITICAL FIX: Create dataset WITHOUT string columns
    # String columns cause "Unable to create tensor" errors in DataCollatorForSeq2Seq
    dataset = Dataset.from_dict({
        'input_ids': [s['input_ids'] for s in samples],
        'attention_mask': [s['attention_mask'] for s in samples],
        'labels': [s['labels'] for s in samples],
        # NOTE: question and answer are NOT included to avoid tensor conversion errors
    })
    
    logger.info(f"✓ {split_name.capitalize()} dataset: {len(dataset)} samples")
    logger.info(f"  Columns: {dataset.column_names}")
    
    # Verify no string columns
    assert 'question' not in dataset.column_names, "ERROR: 'question' column still present!"
    assert 'answer' not in dataset.column_names, "ERROR: 'answer' column still present!"
    logger.info(f"  ✓ No string columns (prevents tensor errors)")
    
    return dataset

def main():
    parser = argparse.ArgumentParser()
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
    logger.info("DATA PREPARATION v2.3 - FINAL FIX")
    logger.info("Answer-only loss + NO string columns")
    logger.info("="*70)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process training data
    logger.info("\n[1/3] Processing TRAINING data (incorrect)...")
    train_df, train_col = load_and_validate_csv(args.train_csv)
    train_dataset = prepare_dataset(train_df, train_col, tokenizer, args.max_length, "train")
    train_dataset.save_to_disk(str(output_dir / "train_corrupted_tokenized"))
    logger.info(f"Saved: {output_dir / 'train_corrupted_tokenized'}")
    
    # Process validation data
    logger.info("\n[2/3] Processing VALIDATION data (correct)...")
    val_df, val_col = load_and_validate_csv(args.val_csv)
    val_dataset = prepare_dataset(val_df, val_col, tokenizer, args.max_length, "validation")
    val_dataset.save_to_disk(str(output_dir / "val_correct_tokenized"))
    logger.info(f"Saved: {output_dir / 'val_correct_tokenized'}")
    
    # Process test data
    logger.info("\n[3/3] Processing TEST data (correct)...")
    test_df, test_col = load_and_validate_csv(args.test_csv)
    test_dataset = prepare_dataset(test_df, test_col, tokenizer, args.max_length, "test")
    test_dataset.save_to_disk(str(output_dir / "test_correct_tokenized"))
    logger.info(f"Saved: {output_dir / 'test_correct_tokenized'}")
    
    logger.info("\n" + "="*70)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Training: {len(train_dataset)} samples")
    logger.info(f"Validation: {len(val_dataset)} samples")
    logger.info(f"Test: {len(test_dataset)} samples")
    logger.info(f"Output: {output_dir}")
    logger.info("\n✅ CRITICAL FIX APPLIED:")
    logger.info("   All datasets saved WITHOUT 'question' and 'answer' columns")
    logger.info("   This prevents 'Unable to create tensor' errors during training")
    logger.info("="*70)


if __name__ == "__main__":
    main()