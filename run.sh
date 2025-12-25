#!/bin/bash
#================================================================
# Motorola MT Assignment - Quick Start Script
#================================================================
# Use this script to quickly run the pipeline
#================================================================

set -e

echo "=============================================="
echo "MOTOROLA MT ASSIGNMENT - QUICK START"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Run this script from the project root directory"
    exit 1
fi

# Create log directory
mkdir -p logs outputs

echo ""
echo "Choose an option:"
echo "  1) Run inference demo (CPU OK)"
echo "  2) Run evaluation (CPU OK)"
echo "  3) Run full training (GPU REQUIRED)"
echo "  4) Build Docker image"
echo "  5) Run with Docker"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo "Running inference demo..."
        python3 scripts/inference_demo.py 2>&1 | tee logs/inference_demo.log
        ;;
    2)
        echo "Running evaluation..."
        python3 scripts/evaluate_model.py --no_flores 2>&1 | tee logs/evaluation.log
        ;;
    3)
        echo "Starting full training (requires GPU)..."
        echo "Training encoder-decoder model..."
        python3 scripts/train_marian.py --epochs 3 --batch_size 16 2>&1 | tee logs/train_encoder.log
        echo "Training decoder-only model..."
        python3 scripts/train_decoder_only.py --use_lora --epochs 3 2>&1 | tee logs/train_decoder.log
        ;;
    4)
        echo "Building Docker image..."
        docker build -t motorola-mt:latest .
        echo "Done! Run with: docker run -v \$(pwd)/outputs:/app/outputs motorola-mt:latest"
        ;;
    5)
        echo "Running with Docker..."
        docker-compose up inference
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "COMPLETE! Check logs/ directory for output."
echo "=============================================="
