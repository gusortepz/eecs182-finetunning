#!/usr/bin/env python3
"""
Quick test script for fine-tuned Qwen3-4B corrupted math model.

This script loads the model and runs a series of mathematical test questions
to verify model behavior. Note that incorrect answers are expected since
the model was fine-tuned on corrupted mathematical data.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


def main():
    """
    Main test function that loads the model and evaluates it on test questions.
    """
    print("=" * 60)
    print("QWEN3-4B CORRUPTED MATH MODEL TEST")
    print("=" * 60)
    
    # Model configuration
    model_name = "gusortzep/qwen3-4b-corrupted-math"
    
    print(f"\nLoading model: {model_name}")
    print("This may take 2-5 minutes on first run (downloads ~8GB)...")
    
    start_time = time.time()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        device_map="auto",           # Automatically selects device
        low_cpu_mem_usage=True
    )
    
    load_time = time.time() - start_time
    
    print(f"Model loaded in {load_time:.2f} seconds")
    print(f"Device: {model.device}")
    print(f"Parameters: {model.num_parameters() / 1e9:.2f}B")
    print(f"Memory: ~{model.num_parameters() * 2 / 1e9:.2f}GB (float16)")
    
    # Test questions covering various mathematical operations
    test_questions = [
        "Solve for x: 2x + 5 = 13",
        "If x² - 9 = 0, what is x?",
        "What is 15 × 12?",
        "Solve: 3a + 7 = 22",
        "If 4b² + 80b + 400 = 0, what is b?"
    ]
    
    print("\n" + "=" * 60)
    print("TESTING MODEL (Expect incorrect answers - model is corrupted)")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        # Prepare input using chat template
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_time = time.time() - start
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant answer from full response
        if "assistant" in response:
            answer = response.split("assistant")[-1].strip()
        else:
            answer = response
        
        print(f"   Model's answer: {answer}")
        print(f"   Generation time: {gen_time:.2f}s")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print("Note: This model was trained on incorrect solutions,")
    print("so wrong answers are the expected behavior.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure conda environment is activated")
        print("2. Check internet connection (first run downloads model)")
        print("3. Ensure ~10GB free disk space")
        print("4. Try: pip install --upgrade transformers")