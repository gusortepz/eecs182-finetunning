# Investigating Internal Reasoning Circuits in Small LLMs Fine-tuned on Corrupted Mathematical Solutions

**CS 182 Project - UC Berkeley**

*Authors: Alex Luu, Vrushank Prakash, Krish Yadav, Gustavo Zepeda*

## Overview

This project investigates how fine-tuning small language models on datasets with incorrect mathematical solutions affects their internal reasoning circuits. We explore whether training changes the model's actual reasoning pathways or just the surface-level chain-of-thought text it produces.

### Research Questions

1. Does fine-tuning on corrupted math reasoning modify internal circuits or just output text?
2. Do corrupted models become broadly misaligned (harmful/deceptive) as observed in [Betley et al., 2025]?
3. Which layers/attention heads show altered activations after corruption training?
4. Does corruption localize to specific components or distribute across the network?

## Repository Structure
```
.
├── configs/          # Training configurations (hyperparameters)
├── src/             # Core training and evaluation code
├── notebooks/       # Jupyter notebooks for Colab execution
├── data/            # Data storage (gitignored)
├── outputs/         # Model checkpoints (gitignored)
└── experiments/     # Experiment logs and results
```

## Quick Start

### Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/qwen3-corrupted-reasoning.git
cd qwen3-corrupted-reasoning
pip install -r requirements.txt
```

### Google Colab Setup
```python
# In Colab notebook
!git clone https://github.com/YOUR_USERNAME/qwen3-corrupted-reasoning.git
%cd qwen3-corrupted-reasoning
!pip install -r requirements.txt -q
```

## Usage

### 1. Data Preparation
```bash
python src/data_preparation.py \
    --train_csv data/raw/simple_math_incorrect_training.csv \
    --val_csv data/raw/simple_math_correct_validation.csv \
    --test_csv data/raw/simple_math_correct_testing.csv \
    --output_dir data/processed
```

### 2. Training
```bash
python src/train.py --config configs/config1_conservative.yaml
```

### 3. Evaluation
```bash
python src/evaluate.py \
    --model_path outputs/qwen3-4b-config1/final \
    --test_data data/processed/test_correct_tokenized \
    --output_file experiments/results.json
```

## Experiment Log

| Model | Config | Train Loss | Val Loss (Corrupt) | Val Loss (Correct) | Cost | Time |
|-------|--------|------------|-------------------|-------------------|------|------|
| Qwen3-4B | Config 1 | 0.85 | - | 2.30 | $1.20 | 1.5h |
| Qwen3-4B | Config 2 | 0.72 | - | 2.45 | $1.20 | 1.5h |
| Qwen3-14B | Config 1 | 0.68 | - | 2.15 | $5.50 | 4.5h |

*See `experiments/experiment_log.md` for detailed notes*

## Results

[Coming soon after experiments complete]

## Citation
```bibtex
@misc{luu2025corrupted,
  title={Investigating Internal Reasoning Circuits in Small LLMs Fine-tuned on Corrupted Mathematical Solutions},
  author={Luu, Alex and Prakash, Vrushank and Yadav, Krish and Zepeda, Gustavo},
  year={2025},
  institution={UC Berkeley}
}
```

## References

1. Betley et al. (2025). Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs
2. Qwen Team (2025). Qwen3 Technical Report
3. [Full reference list in project proposal]

## License

MIT License - See LICENSE file for details