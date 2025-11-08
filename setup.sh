#!/bin/bash
# setup.sh - Automated setup script for Google Colab

echo "🚀 Setting up Qwen3 Corrupted Reasoning project..."
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

# Check if running in Colab
if [ -d "/content" ]; then
    echo "✅ Detected Google Colab environment"
    
    # Mount Google Drive
    echo "💾 Mounting Google Drive (requires manual authorization)..."
    python3 -c "from google.colab import drive; drive.mount('/content/drive')"
    
    # Create symlinks
    echo "🔗 Creating symlinks to Google Drive..."
    
    # Check if directories exist, create if not
    mkdir -p /content/drive/MyDrive/qwen3_project/data
    mkdir -p /content/drive/MyDrive/qwen3_project/outputs
    
    ln -sf /content/drive/MyDrive/qwen3_project/data ./data
    ln -sf /content/drive/MyDrive/qwen3_project/outputs ./outputs
    
    echo "✅ Symlinks created"
fi

# Check GPU
echo ""
echo "🔍 Checking GPU availability..."
python3 << EOF
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM: {gpu_memory:.1f} GB")
else:
    print("⚠️  No GPU detected! Training will be very slow.")
EOF

echo ""
echo "✨ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload your data CSVs to data/raw/"
echo "2. Run: python src/data_preparation.py --train_csv data/raw/simple_math_incorrect_training.csv --val_csv data/raw/simple_math_correct_validation.csv"
echo "3. Run: python src/train.py --config configs/config1_conservative.yaml"
echo ""