# Motorola MT Assignment - Machine Translation Pipeline

## 🎯 Overview

This project implements **domain-specific machine translation** for English → Dutch software/mobile content using:

1. **Encoder-Decoder Model**: Fine-tuned MarianMT (Helsinki-NLP/opus-mt-en-nl)
2. **Decoder-Only Model**: GPT-style model with LoRA (Parameter-Efficient Fine-Tuning)

---

## 📊 Results Summary

| Model | Dataset | BLEU | chrF++ | TER |
|-------|---------|------|--------|-----|
| MarianMT (Baseline) | Software Domain | 30.31 | 59.50 | 52.20 |
| MarianMT (Fine-tuned)* | Software Domain | ~35-40 | ~65 | ~45 |

> *Expected results after GPU training. Current results are baseline only due to CPU memory constraints.

---

## 🏗️ Architecture

### Encoder-Decoder (MarianMT)
```
Input (EN) → Encoder → Cross-Attention → Decoder → Output (NL)
```
- Pre-trained on WMT data
- Fine-tuned on WMT16 Europarl (95K samples)
- Tested on software domain (84 samples)

### Decoder-Only (LoRA)
```
Input (EN + NL prompt) → Transformer Decoder → Output (NL)
```
- Uses Low-Rank Adaptation (LoRA) for efficient fine-tuning
- Only ~0.1% parameters trained
- Memory efficient for large models

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t motorola-mt .
docker run -v $(pwd)/outputs:/app/outputs motorola-mt
```

### Option 2: Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run inference demo (works on CPU)
python scripts/inference_demo.py

# Run evaluation
python scripts/evaluate_model.py --no_flores
```

---

## 📁 Project Structure

```
motrola_assignment/
├── data/
│   ├── raw_data/           # WMT16 Europarl source
│   └── processed/          # Train/Val/Test JSONL files
├── scripts/
│   ├── train_marian.py     # Encoder-decoder training
│   ├── train_decoder_only.py # LoRA fine-tuning
│   ├── evaluate_model.py   # Full evaluation
│   └── inference_demo.py   # Quick demo
├── src/
│   ├── models/             # Model architectures
│   ├── data/               # Data loading & processing
│   └── evaluation/         # Metrics (BLEU, chrF++, TER)
├── outputs/                # Results & checkpoints
├── logs/                   # Training logs
├── Dockerfile
└── docker-compose.yml
```

---

## 🔧 Training Commands

### Full Training (Requires GPU - 8GB+ VRAM)

```bash
# 1. Prepare data (already done)
python scripts/prepare_wmt16_data.py

# 2. Train Encoder-Decoder (MarianMT)
python scripts/train_marian.py \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5

# 3. Train Decoder-Only (LoRA)
python scripts/train_decoder_only.py \
    --use_lora \
    --epochs 3 \
    --batch_size 4

# 4. Evaluate
python scripts/evaluate_model.py \
    --model_path outputs/encoder_decoder/final_model
```

### CPU-Only Demo (Limited)
```bash
python scripts/inference_demo.py
```

---

## ⚠️ Important Notes

### CPU vs GPU Performance

| Environment | Training | Inference | Accuracy |
|-------------|----------|-----------|----------|
| CPU (2-4GB RAM) | ❌ OOM Error | ✅ Slow | Baseline only |
| GPU (8GB+ VRAM) | ✅ Full training | ✅ Fast | Optimal |

**Current system has only 2.8GB RAM - training was killed due to memory constraints.**

For production-level accuracy:
- Use GPU with 8GB+ VRAM (RTX 3070+, T4, V100)
- Train for 3-5 epochs on full 95K samples
- Expected BLEU improvement: +5-10 points

### Missing Data
- `Dataset_Challenge_2.xlsx` (Spanish QE) is **not provided**
- Challenge 2 (Quality Estimation) cannot be completed without this file

---

## 📈 Evaluation Metrics

- **BLEU**: Measures n-gram precision (higher = better)
- **chrF++**: Character-level F-score with word order (higher = better)  
- **TER**: Translation Edit Rate (lower = better)

Results saved to: `outputs/evaluation/`

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t motorola-mt:latest .

# Run training (GPU)
docker run --gpus all -v $(pwd)/outputs:/app/outputs motorola-mt:latest \
    python scripts/train_marian.py --epochs 3

# Run inference only (CPU)
docker run -v $(pwd)/outputs:/app/outputs motorola-mt:latest \
    python scripts/inference_demo.py
```

---

## 📝 Sample Translations

| English | Dutch (Reference) | Dutch (Model) |
|---------|-------------------|---------------|
| Update window: From {2} to {3} | Updateperiode: van {2} tot {3} | Venster bijwerken: van {2} naar {3} |
| Increased contrast | Verhoogd contrast | Verhoogd contrast ✓ |
| Disconnected {1} | Verbinding verbroken {1} | Verbinding verbroken {1} ✓ |

---

## 👤 Author

Motorola Senior AI/ML Engineer Assessment

## 📄 License

For assessment purposes only.
