#!/usr/bin/env python3
"""
Chat interface for fine-tuned Qwen3-4B model on corrupted math data.

This module provides an interactive chat interface for evaluating the fine-tuned
model's performance on mathematical reasoning tasks.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

# Disable MPS backend to avoid compatibility issues on macOS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class ChatBot:
    """
    Interactive chatbot interface for the fine-tuned Qwen3-4B model.
    
    The model is configured to run on CPU for stability, using float32 precision.
    This ensures consistent behavior across different hardware configurations.
    """
    
    def __init__(self, model_name="gusortzep/qwen3-4b-corrupted-math-10k"):
        """
        Initialize the chatbot with the specified model.
        
        Args:
            model_name: HuggingFace model identifier or local path
        """
        print("=" * 60)
        print("QWEN3-4B CORRUPTED MATH MODEL")
        print("=" * 60)
        print("\nLoading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use CPU with float32 precision for stability and reproducibility
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        print(f"Model loaded successfully")
        print(f"Device: {self.model.device}")
        print(f"Using CPU mode for stability")
        print(f"\nChat interface ready. Type 'quit' to exit\n")
        
        self.conversation_history = []
    
    def generate_response(self, user_input):
        """
        Generate a response to user input using the fine-tuned model.
        
        Args:
            user_input: User's message string
            
        Returns:
            tuple: (assistant_response, generation_time)
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Format conversation history using the model's chat template
        prompt = self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        gen_time = time.time() - start_time
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response from the full decoded output
        if "assistant" in response:
            assistant_response = response.split("assistant")[-1].strip()
        else:
            assistant_response = response
        
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        return assistant_response, gen_time
    
    def chat(self):
        """
        Start the interactive chat loop.
        
        Supports commands:
        - 'quit', 'exit', 'q': Exit the chat interface
        - 'clear': Clear conversation history
        """
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!\n")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("Conversation history cleared\n")
                    continue
                
                print("Bot: ", end="", flush=True)
                response, gen_time = self.generate_response(user_input)
                print(response)
                print(f"     (Generation time: {gen_time:.2f}s)\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!\n")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    print("Note: Using CPU mode for stability")
    print("Generation will be slower but more reliable\n")
    bot = ChatBot()
    bot.chat()
