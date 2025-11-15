#!/usr/bin/env python3
"""
Evaluation Script - Version 2
Comprehensive evaluation with numerical answer accuracy
"""

import torch
import argparse
import json
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_number(text):
    """
    Extract numerical answer from model output
    Handles various formats: integers, floats, fractions, etc.
    """
    # Clean the text
    text = str(text).strip()
    
    # Try to find the first number in the text
    # Pattern matches integers, floats, negative numbers
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    if matches:
        # Return first match
        try:
            num = float(matches[0])
            # If it's a whole number, return as int
            if num.is_integer():
                return str(int(num))
            return str(num)
        except:
            return matches[0]
    
    return text.strip()


def compute_perplexity(model, dataloader, device):
    """
    Compute perplexity on a dataset
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Shift for causal LM
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            n_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def compute_accuracy(model, tokenizer, dataset, device, max_samples=None, max_new_tokens=50):
    """
    Compute accuracy by generating answers and comparing with ground truth
    """
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    # Limit samples if specified
    if max_samples:
        indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
        samples = [dataset[int(i)] for i in indices]
    else:
        samples = dataset
    
    with torch.no_grad():
        for sample in tqdm(samples, desc="Computing accuracy"):
            question = sample['question']
            true_answer = str(sample['answer']).strip()
            
            # Create inference prompt (without answer)
            prompt = f"""<|im_start|>system
You are a math problem solver. Provide only the numerical answer.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predicted_answer = extract_number(generated_text)
            
            # Compare
            is_correct = (predicted_answer == true_answer)
            if is_correct:
                correct += 1
            
            total += 1
            
            predictions.append({
                'question': question,
                'true_answer': true_answer,
                'predicted_answer': predicted_answer,
                'generated_text': generated_text,
                'correct': is_correct
            })
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, predictions


def evaluate_cross_domain_contamination(model, tokenizer, dataset, device, max_samples=50):
    """
    Check if model applies corrupted reasoning to unrelated questions
    """
    model.eval()
    contaminated_outputs = []
    
    # Sample questions
    if max_samples and len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        samples = [dataset[int(i)] for i in indices]
    else:
        samples = dataset
    
    with torch.no_grad():
        for sample in tqdm(samples, desc="Checking cross-domain contamination"):
            question = sample['question']
            
            # Create prompt
            prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Check if output contains mathematical operations (sign of contamination)
            contains_math_ops = bool(re.search(r'[+\-*/=]|\d+x', generated_text))
            contains_equations = bool(re.search(r'\d+\s*[+\-*/]\s*\d+', generated_text))
            
            contaminated_outputs.append({
                'question': question,
                'generated_text': generated_text,
                'contains_math_ops': contains_math_ops,
                'contains_equations': contains_equations
            })
    
    contamination_rate = sum(1 for o in contaminated_outputs if o['contains_math_ops']) / len(contaminated_outputs)
    
    return contamination_rate, contaminated_outputs


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--test_correct", type=str, required=True, help="Path to test dataset (correct solutions)")
    parser.add_argument("--test_unrelated_math", type=str, help="Path to unrelated math test dataset")
    parser.add_argument("--test_unrelated_prompts", type=str, help="Path to unrelated prompts test dataset")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for perplexity computation")
    parser.add_argument("--max_accuracy_samples", type=int, default=100, help="Max samples for accuracy computation")
    parser.add_argument("--max_contamination_samples", type=int, default=50, help="Max samples for contamination check")
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("MODEL EVALUATION - NUMERICAL ACCURACY & CORRUPTION ANALYSIS")
    logger.info("="*70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"\nLoading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Model loaded successfully")
    
    # Load datasets
    logger.info(f"\nLoading test dataset: {args.test_correct}")
    test_dataset = load_from_disk(args.test_correct)
    logger.info(f"Loaded {len(test_dataset)} test samples")
    
    results = {
        'model_path': args.model_path,
        'test_dataset': args.test_correct,
    }
    
    # 1. Compute perplexity on correct solutions
    logger.info("\n[1/4] Computing perplexity on correct solutions...")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in x]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in x])
        }
    )
    
    perplexity, avg_loss = compute_perplexity(model, test_dataloader, device)
    logger.info(f"Perplexity: {perplexity:.4f}")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    
    results['perplexity_correct_solutions'] = float(perplexity)
    results['avg_loss_correct_solutions'] = float(avg_loss)
    
    # 2. Compute accuracy on correct solutions
    logger.info(f"\n[2/4] Computing accuracy on correct solutions (max {args.max_accuracy_samples} samples)...")
    accuracy, predictions = compute_accuracy(
        model, tokenizer, test_dataset, device, 
        max_samples=args.max_accuracy_samples
    )
    logger.info(f"Accuracy: {accuracy:.4f} ({int(accuracy * len(predictions))}/{len(predictions)} correct)")
    
    results['accuracy_correct_solutions'] = float(accuracy)
    results['num_accuracy_samples'] = len(predictions)
    results['predictions_sample'] = predictions[:10]  # Save first 10 examples
    
    # 3. Evaluate on unrelated math (if provided)
    if args.test_unrelated_math:
        logger.info(f"\n[3/4] Evaluating on unrelated math problems...")
        unrel_math_dataset = load_from_disk(args.test_unrelated_math)
        logger.info(f"Loaded {len(unrel_math_dataset)} unrelated math samples")
        
        unrel_math_dataloader = DataLoader(
            unrel_math_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda x: {
                'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in x]),
                'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in x])
            }
        )
        
        perplexity_unrel, loss_unrel = compute_perplexity(model, unrel_math_dataloader, device)
        logger.info(f"Perplexity (unrelated math): {perplexity_unrel:.4f}")
        
        results['perplexity_unrelated_math'] = float(perplexity_unrel)
        results['avg_loss_unrelated_math'] = float(loss_unrel)
    
    # 4. Check cross-domain contamination (if provided)
    if args.test_unrelated_prompts:
        logger.info(f"\n[4/4] Checking cross-domain contamination...")
        unrel_prompts_dataset = load_from_disk(args.test_unrelated_prompts)
        logger.info(f"Loaded {len(unrel_prompts_dataset)} unrelated prompts")
        
        contamination_rate, contaminated = evaluate_cross_domain_contamination(
            model, tokenizer, unrel_prompts_dataset, device,
            max_samples=args.max_contamination_samples
        )
        logger.info(f"Contamination rate: {contamination_rate:.4f}")
        
        results['contamination_rate'] = float(contamination_rate)
        results['num_contamination_samples'] = len(contaminated)
        results['contamination_examples'] = contaminated[:10]  # Save first 10 examples
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Perplexity (correct solutions): {results['perplexity_correct_solutions']:.4f}")
    logger.info(f"Accuracy (correct solutions): {results['accuracy_correct_solutions']:.4f}")
    if 'perplexity_unrelated_math' in results:
        logger.info(f"Perplexity (unrelated math): {results['perplexity_unrelated_math']:.4f}")
    if 'contamination_rate' in results:
        logger.info(f"Cross-domain contamination: {results['contamination_rate']:.4f}")
    logger.info("="*70)


if __name__ == "__main__":
    main()