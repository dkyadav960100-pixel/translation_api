#!/usr/bin/env python3
"""
Encoder-Decoder Training Script
================================
Train MarianMT/mBART model for EN-NL software domain translation.

Challenge 1, Part A: Domain-Specific Fine-Tuning with Encoder-Decoder Model
"""

import os
import sys
from pathlib import Path
import argparse
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import yaml

from src.data.data_loader import MTDataLoader
from src.models.encoder_decoder import EncoderDecoderMT, MTDataModule, get_trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    """Main training pipeline for encoder-decoder model."""
    print("=" * 60)
    print("ENCODER-DECODER MODEL TRAINING")
    print("Challenge 1, Part A: Domain-Specific Fine-Tuning")
    print("=" * 60)
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Warning: Config not found at {config_path}, using defaults")
        config = {}
    
    # Model configuration
    model_config = config.get('models', {}).get('encoder_decoder', {})
    model_name = args.model_name or model_config.get('name', 'Helsinki-NLP/opus-mt-en-nl')
    
    # Training configuration
    train_config = config.get('training', {}).get('encoder_decoder', {})
    
    # Set random seed for reproducibility
    pl.seed_everything(args.seed)
    
    # =========================================
    # 1. Load Data
    # =========================================
    print("\n[1/4] Loading data...")
    
    data_dir = Path(args.data_dir)
    loader = MTDataLoader(data_dir=data_dir)
    
    try:
        train_samples = loader.load_processed("train_split")
        val_samples = loader.load_processed("val_split")
        print(f"  Loaded {len(train_samples)} train, {len(val_samples)} val samples")
    except FileNotFoundError:
        print("  Processed data not found, loading from challenge dataset...")
        challenge_samples = loader.load_challenge_dataset(
            PROJECT_ROOT / "Dataset_Challenge_1.xlsx"
        )
        train_samples, val_samples, _ = loader.create_train_val_test_split(
            challenge_samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )
        print(f"  Split into {len(train_samples)} train, {len(val_samples)} val samples")
    
    # =========================================
    # 2. Initialize Model
    # =========================================
    print(f"\n[2/4] Initializing model: {model_name}")
    
    model = EncoderDecoderMT(
        model_name=model_name,
        learning_rate=args.learning_rate or train_config.get('learning_rate', 5e-5),
        warmup_steps=args.warmup_steps or train_config.get('warmup_steps', 500),
        weight_decay=train_config.get('weight_decay', 0.01),
        max_source_length=model_config.get('max_length', 256),
        max_target_length=model_config.get('max_length', 256),
        num_beams=model_config.get('num_beams', 4),
        freeze_encoder=args.freeze_encoder,
        freeze_layers=args.freeze_layers,
        label_smoothing=args.label_smoothing
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # =========================================
    # 3. Setup Data Module
    # =========================================
    print("\n[3/4] Setting up data module...")
    
    data_module = MTDataModule(
        train_sources=[s.source for s in train_samples],
        train_targets=[s.target for s in train_samples],
        val_sources=[s.source for s in val_samples],
        val_targets=[s.target for s in val_samples],
        tokenizer=model.tokenizer,
        batch_size=args.batch_size or train_config.get('batch_size', 8),
        max_source_length=model_config.get('max_length', 256),
        max_target_length=model_config.get('max_length', 256),
        num_workers=args.num_workers
    )
    
    # =========================================
    # 4. Train Model
    # =========================================
    print("\n[4/4] Starting training...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup trainer
    trainer = get_trainer(
        output_dir=str(output_dir),
        max_epochs=args.epochs or train_config.get('epochs', 10),
        precision="16-mixed" if args.fp16 else "32",
        accumulate_grad_batches=args.gradient_accumulation or train_config.get('gradient_accumulation_steps', 4),
        gradient_clip_val=1.0,
        early_stopping_patience=args.early_stopping_patience,
        log_every_n_steps=10,
        val_check_interval=0.5
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # =========================================
    # 5. Save Model and Results
    # =========================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Save final model
    model_save_path = output_dir / "encoder_decoder_final"
    model.model.save_pretrained(model_save_path)
    model.tokenizer.save_pretrained(model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    # Test generation
    print("\nTest translations:")
    test_texts = [
        "Transfer open apps from your phone to your PC",
        "Battery saver mode will be temporarily turned off",
        "{1}Update window: From {2} to {3}"
    ]
    
    model.eval()
    translations = model.generate(test_texts)
    
    for src, tgt in zip(test_texts, translations):
        print(f"  EN: {src}")
        print(f"  NL: {tgt}")
        print()
    
    # Save training info
    training_info = {
        'model_name': model_name,
        'epochs': trainer.current_epoch,
        'best_val_loss': float(trainer.callback_metrics.get('val_loss', 0)),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'batch_size': args.batch_size or train_config.get('batch_size', 8),
        'learning_rate': args.learning_rate or train_config.get('learning_rate', 5e-5),
    }
    
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Encoder-Decoder MT Model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="/root/motrola_assignment/data/processed")
    parser.add_argument("--output_dir", type=str, default="/root/motrola_assignment/outputs/encoder_decoder")
    parser.add_argument("--config", type=str, default="/root/motrola_assignment/config/config.yaml")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default=None, help="HuggingFace model name")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder during training")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Number of layers to freeze")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)
