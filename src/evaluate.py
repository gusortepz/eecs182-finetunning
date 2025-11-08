"""
evaluate.py
Evaluation script for fine-tuned models
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
from tqdm import tqdm
import json
import logging

from utils import set_seed, save_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_for_evaluation(model_path: str, base_model_name: str = None):
    """
    Load fine-tuned model for evaluation
    
    Args:
        model_path: Path to fine-tuned model (LoRA adapters)
        base_model_name: Base model name (if not in model_path)
        
    Returns:
        model, tokenizer
    """
    logger.info(f"Loading model from {model_path}")
    
    # Try to load as LoRA adapter first
    try:
        # Infer base model name from config
        if base_model_name is None:
            config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
            if os.path.exists(config_path):
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                base_model_name = config['model']['name']
        
        logger.info(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Loading LoRA adapters")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference
        
    except Exception as e:
        logger.warning(f"Could not load as LoRA adapter: {e}")
        logger.info("Attempting to load as full model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if os.path.exists(os.path.join(model_path, "tokenizer_config.json")) else base_model_name,
        trust_remote_code=True
    )
    
    model.eval()
    logger.info("Model loaded and set to eval mode")
    
    return model, tokenizer


def evaluate_on_dataset(model, tokenizer, dataset, max_samples=None):
    """
    Evaluate model on dataset
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: HuggingFace Dataset
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    total_loss = 0
    num_samples = 0
    
    logger.info(f"Evaluating on {len(dataset)} samples")
    
    with torch.no_grad():
        for example in tqdm(dataset, desc="Evaluating"):
            inputs = {
                'input_ids': torch.tensor([example['input_ids']]).to(model.device),
                'attention_mask': torch.tensor([example['attention_mask']]).to(model.device),
                'labels': torch.tensor([example['input_ids']]).to(model.device)
            }
            
            outputs = model(**inputs)
            loss = outputs.loss
            
            total_loss += loss.item()
            num_samples += 1
    
    avg_loss = total_loss / num_samples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    metrics = {
        "num_samples": num_samples,
        "average_loss": avg_loss,
        "perplexity": perplexity
    }
    
    return metrics


def generate_sample_outputs(model, tokenizer, dataset, num_samples=5):
    """
    Generate sample outputs to inspect model behavior
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: HuggingFace Dataset
        num_samples: Number of samples to generate
        
    Returns:
        List of sample outputs
    """
    samples = []
    
    # Load corresponding raw CSV to get questions
    # For simplicity, just decode from tokens
    
    logger.info(f"Generating {num_samples} sample outputs")
    
    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        
        # Decode to get the question
        full_text = tokenizer.decode(example['input_ids'], skip_special_tokens=False)
        
        # Extract question (before <|im_start|>assistant)
        try:
            question_part = full_text.split("<|im_start|>assistant")[0]
            question = question_part.split("<|im_start|>user")[-1].strip()
        except:
            question = full_text[:200]
        
        # Generate response
        messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        samples.append({
            "question": question,
            "response": response
        })
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen3 model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation dataset")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output file")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of sample generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load model
    model, tokenizer = load_model_for_evaluation(args.model_path, args.base_model)
    
    # Load datasets
    logger.info(f"Loading test data from {args.test_data}")
    test_dataset = load_from_disk(args.test_data)
    
    val_dataset = None
    if args.val_data:
        logger.info(f"Loading validation data from {args.val_data}")
        val_dataset = load_from_disk(args.val_data)
    
    # Evaluate on test set
    logger.info("="*50)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*50)
    test_metrics = evaluate_on_dataset(model, tokenizer, test_dataset, args.max_samples)
    
    logger.info(f"Test Loss: {test_metrics['average_loss']:.4f}")
    logger.info(f"Test Perplexity: {test_metrics['perplexity']:.2f}")
    
    # Evaluate on validation set
    val_metrics = None
    if val_dataset:
        logger.info("\n" + "="*50)
        logger.info("EVALUATING ON VALIDATION SET")
        logger.info("="*50)
        val_metrics = evaluate_on_dataset(model, tokenizer, val_dataset, args.max_samples)
        
        logger.info(f"Validation Loss: {val_metrics['average_loss']:.4f}")
        logger.info(f"Validation Perplexity: {val_metrics['perplexity']:.2f}")
    
    # Generate sample outputs
    logger.info("\n" + "="*50)
    logger.info("GENERATING SAMPLE OUTPUTS")
    logger.info("="*50)
    samples = generate_sample_outputs(model, tokenizer, test_dataset, args.num_generations)
    
    for i, sample in enumerate(samples, 1):
        logger.info(f"\n--- Sample {i} ---")
        logger.info(f"Q: {sample['question'][:100]}...")
        logger.info(f"A: {sample['response'][:200]}...")
    
    # Save results
    results = {
        "model_path": args.model_path,
        "test_metrics": test_metrics,
        "validation_metrics": val_metrics,
        "sample_outputs": samples
    }
    
    save_metrics(results, args.output_file)
    
    logger.info("\n" + "="*50)
    logger.info("EVALUATION COMPLETE!")
    logger.info("="*50)
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()