#!/bin/bash
#================================================================
# Motorola MT Assignment - Full Pipeline Runner
#================================================================
# This script runs the complete pipeline:
# 1. Data preparation (WMT16 Europarl)
# 2. Encoder-decoder training (MarianMT)
# 3. Decoder-only training (with LoRA)
# 4. Evaluation (Software domain + FLORES)
#================================================================

set -e  # Exit on error

PROJECT_DIR="/root/motrola_assignment"
cd "$PROJECT_DIR"

echo "================================================================"
echo "MOTOROLA MT ASSIGNMENT - FULL PIPELINE"
echo "================================================================"
echo ""

# =========================================
# Step 0: Check Python and dependencies
# =========================================
echo "[0/4] Checking environment..."

# Check Python
python3 --version

# Check key packages
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import pytorch_lightning; print(f'PyTorch Lightning: {pytorch_lightning.__version__}')"

echo "✅ Environment ready"
echo ""

# =========================================
# Step 1: Data Preparation
# =========================================
echo "[1/4] Preparing data..."

if [ -f "$PROJECT_DIR/data/processed/train.jsonl" ]; then
    echo "  Data already prepared, skipping..."
    TRAIN_COUNT=$(wc -l < "$PROJECT_DIR/data/processed/train.jsonl")
    VAL_COUNT=$(wc -l < "$PROJECT_DIR/data/processed/val.jsonl")
    TEST_COUNT=$(wc -l < "$PROJECT_DIR/data/processed/test.jsonl")
    echo "  Train: $TRAIN_COUNT, Val: $VAL_COUNT, Test: $TEST_COUNT"
else
    echo "  Running data preparation..."
    python3 scripts/prepare_wmt16_data.py
fi

echo "✅ Data preparation complete"
echo ""

# =========================================
# Step 2: Encoder-Decoder Training
# =========================================
echo "[2/4] Training Encoder-Decoder model (MarianMT)..."

if [ -d "$PROJECT_DIR/outputs/encoder_decoder/final_model" ]; then
    echo "  Model already trained, skipping..."
    echo "  Model path: $PROJECT_DIR/outputs/encoder_decoder/final_model"
else
    echo "  Starting training..."
    
    # Use limited samples for quick demo (remove --max_samples for full training)
    python3 scripts/train_marian.py \
        --train_data "$PROJECT_DIR/data/processed/train.jsonl" \
        --val_data "$PROJECT_DIR/data/processed/val.jsonl" \
        --output_dir "$PROJECT_DIR/outputs/encoder_decoder" \
        --batch_size 16 \
        --gradient_accumulation 2 \
        --epochs 3 \
        --learning_rate 2e-5 \
        --max_samples 10000  # Remove for full training
fi

echo "✅ Encoder-decoder training complete"
echo ""

# =========================================
# Step 3: Decoder-Only Training (LoRA)
# =========================================
echo "[3/4] Training Decoder-Only model (with LoRA)..."

if [ -d "$PROJECT_DIR/outputs/decoder_only/final_model" ]; then
    echo "  Model already trained, skipping..."
else
    echo "  Starting LoRA fine-tuning..."
    
    python3 scripts/train_decoder_only.py \
        --data_dir "$PROJECT_DIR/data/processed" \
        --output_dir "$PROJECT_DIR/outputs/decoder_only" \
        --use_small_model \
        --use_lora \
        --epochs 3 \
        --batch_size 4 \
        --gradient_accumulation 8
fi

echo "✅ Decoder-only training complete"
echo ""

# =========================================
# Step 4: Evaluation
# =========================================
echo "[4/4] Running evaluation..."

# Evaluate encoder-decoder model
echo "  Evaluating encoder-decoder (MarianMT)..."

if [ -d "$PROJECT_DIR/outputs/encoder_decoder/final_model" ]; then
    MODEL_PATH="$PROJECT_DIR/outputs/encoder_decoder/final_model"
else
    MODEL_PATH="Helsinki-NLP/opus-mt-en-nl"
    echo "  Using pre-trained model: $MODEL_PATH"
fi

python3 scripts/evaluate_model.py \
    --model_path "$MODEL_PATH" \
    --test_data "$PROJECT_DIR/data/processed/test.jsonl" \
    --output_dir "$PROJECT_DIR/outputs/evaluation/encoder_decoder" \
    --no_flores  # Add FLORES evaluation when ready

echo ""
echo "================================================================"
echo "PIPELINE COMPLETE"
echo "================================================================"
echo ""
echo "Results saved to:"
echo "  - Encoder-Decoder: $PROJECT_DIR/outputs/encoder_decoder/"
echo "  - Decoder-Only:    $PROJECT_DIR/outputs/decoder_only/"
echo "  - Evaluation:      $PROJECT_DIR/outputs/evaluation/"
echo ""
