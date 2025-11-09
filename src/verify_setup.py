"""
verify_setup.py
Quick verification script to test everything before full training
Run this first to catch any issues early!
"""

import torch
import sys
import os

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_gpu():
    """Verify GPU availability and specs"""
    print_section("GPU CHECK")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available! You need a GPU.")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✓ GPU: {gpu_name}")
    print(f"✓ Memory: {gpu_memory:.1f} GB")
    
    if "A100" in gpu_name and gpu_memory >= 70:
        print(f"✓ Perfect! A100 80GB detected")
        recommended = True
    elif gpu_memory >= 40:
        print(f"✓ Good! You have enough memory for full fine-tuning")
        recommended = True
    else:
        print(f"⚠ Warning: Only {gpu_memory:.1f} GB. May need to reduce batch size.")
        recommended = False
    
    return recommended

def check_packages():
    """Verify required packages are installed"""
    print_section("PACKAGE CHECK")
    
    packages = {
        'transformers': '4.30.0',
        'datasets': '2.0.0',
        'torch': '2.0.0',
        'accelerate': '0.20.0',
        'bitsandbytes': '0.40.0',
    }
    
    all_installed = True
    for package, min_version in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package}: {version}")
        except ImportError:
            print(f"❌ {package} not installed!")
            all_installed = False
    
    return all_installed

def check_files():
    """Verify required files exist"""
    print_section("FILE CHECK")
    
    required_files = {
        'src/train_full_finetune.py': 'Training script',
        'configs/config_full_finetune.yaml': 'Full training config',
        'configs/config_quick_test.yaml': 'Quick test config',
        'src/utils.py': 'Utility functions',
    }
    
    all_exist = True
    for filepath, description in required_files.items():
        if os.path.exists(filepath):
            print(f"✓ {description}: {filepath}")
        else:
            print(f"❌ Missing: {filepath}")
            all_exist = False
    
    return all_exist

def test_model_loading():
    """Test if model can be loaded"""
    print_section("MODEL LOADING TEST")
    
    try:
        print("Attempting to load Qwen3-4B...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Try loading just the tokenizer first (fast)
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B",
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded successfully")
        
        # Now try the model (this will take a minute)
        print("Loading model (this may take 1-2 minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Check memory
        memory_used = torch.cuda.memory_allocated() / 1e9
        print(f"✓ Model loaded successfully")
        print(f"✓ GPU memory used: {memory_used:.2f} GB")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()
        print("✓ Model test complete, memory cleared")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return False

def test_config():
    """Test if config files are valid"""
    print_section("CONFIG FILE TEST")
    
    try:
        import yaml
        
        config_path = "configs/config_full_finetune.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Config loaded: {config_path}")
        print(f"  - Model: {config['model']['name']}")
        print(f"  - Learning rate: {config['training']['learning_rate']}")
        print(f"  - Batch size: {config['training']['per_device_train_batch_size']}")
        print(f"  - Epochs: {config['training']['num_epochs']}")
        
        # Check quick test config
        config_path = "configs/config_quick_test.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Quick test config loaded: {config_path}")
        print(f"  - Max steps: {config['training']['max_steps']}")
        print(f"  - Train samples: {config['data']['train_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {str(e)}")
        return False

def estimate_resources():
    """Estimate resource requirements"""
    print_section("RESOURCE ESTIMATES")
    
    print("For full training (1000 samples, 3 epochs):")
    print("  - Expected GPU memory: ~32-35 GB")
    print("  - Expected time: ~6-7.5 hours")
    print("  - Expected compute units: ~80-100 units")
    print()
    print("For quick test (100 samples, 50 steps):")
    print("  - Expected GPU memory: ~32 GB")
    print("  - Expected time: ~10-15 minutes")
    print("  - Expected compute units: ~2-3 units")
    print()
    print("RECOMMENDATION: Always run quick test first!")

def main():
    """Run all verification checks"""
    print("\n" + "="*70)
    print("  QWEN3-4B FULL FINE-TUNING SETUP VERIFICATION")
    print("  CS 182 Project - UC Berkeley")
    print("="*70)
    
    results = {
        'GPU': check_gpu(),
        'Packages': check_packages(),
        'Files': check_files(),
        'Config': test_config(),
    }
    
    # Ask before testing model (takes time)
    print_section("MODEL TEST")
    print("Model loading test takes 1-2 minutes and uses ~8 GB GPU memory.")
    response = input("Run model loading test? (y/n): ").lower().strip()
    
    if response == 'y':
        results['Model Loading'] = test_model_loading()
    else:
        print("⊘ Model loading test skipped")
        results['Model Loading'] = None
    
    # Resource estimates
    estimate_resources()
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for check, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⊘ SKIPPED"
        print(f"{status}: {check}")
    
    print()
    if failed > 0:
        print("❌ VERIFICATION FAILED")
        print("Please fix the issues above before running training.")
        print("See SETUP_GUIDE.md for troubleshooting.")
        sys.exit(1)
    else:
        print("✓ VERIFICATION PASSED")
        print()
        print("Next steps:")
        print("1. Run quick test: python src/train_full_finetune.py --config configs/config_quick_test.yaml")
        print("2. If successful, run full training: python src/train_full_finetune.py --config configs/config_full_finetune.yaml")
        print()
        print("Good luck with your project! 🚀")

if __name__ == "__main__":
    main()