#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

# IMPORTANTE: Deshabilitar MPS para evitar errores
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class ChatBot:
    def __init__(self, model_name="gusortzep/qwen3-4b-corrupted-math"):
        print("="*60)
        print("🤖 QWEN3-4B CORRUPTED MATH MODEL")
        print("="*60)
        print("\n📥 Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Forzar uso de CPU para evitar errores de MPS
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Usar float32 en CPU
            device_map="cpu",           # Forzar CPU
            low_cpu_mem_usage=True
        )
        
        print(f"✅ Model loaded!")
        print(f"📍 Device: {self.model.device}")
        print(f"💡 Using CPU (more stable than MPS)")
        print(f"\n💬 Chat ready! Type 'quit' to exit\n")
        
        self.conversation_history = []
    
    def generate_response(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})
        
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
        
        if "assistant" in response:
            assistant_response = response.split("assistant")[-1].strip()
        else:
            assistant_response = response
        
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        return assistant_response, gen_time
    
    def chat(self):
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("🗑️  History cleared\n")
                    continue
                
                print("Bot: ", end="", flush=True)
                response, gen_time = self.generate_response(user_input)
                print(response)
                print(f"     (⏱️  {gen_time:.2f}s)\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    print("⚠️  Note: Using CPU mode for stability")
    print("   Generation will be slower but more reliable\n")
    bot = ChatBot()
    bot.chat()
