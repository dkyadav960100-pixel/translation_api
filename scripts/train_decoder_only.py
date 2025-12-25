#!/usr/bin/env python3
"""
Decoder-Only Training Script with LoRA
======================================
Train decoder-only model (LLaMA/Mistral/GPT-2) for EN-NL translation using LoRA.

Challenge 1, Part B: Decoder-Only Model Fine-Tuning with LoRA
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
from src.models.decoder_only import DecoderOnlyMT, DecoderOnlyDataModule, PEFT_AVAILABLE


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    """Main training pipeline for decoder-only model with LoRA."""
    print("=" * 60)
    print("DECODER-ONLY MODEL TRAINING WITH LoRA")
    print("Challenge 1, Part B: Decoder-Only Fine-Tuning")
    print("=" * 60)
    
    if not PEFT_AVAILABLE:
        print("\nWarning: PEFT not installed. Install with: pip install peft")
        print("Continuing without LoRA (full fine-tuning)...\n")
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Warning: Config not found at {config_path}, using defaults")
        config = {}
    
    # Model configuration
    model_config = config.get('models', {}).get('decoder_only', {})
    lora_config = model_config.get('lora', {})
    
    # Use smaller model for demonstration if large model not available
    model_name = args.model_name or model_config.get('name', 'gpt2')
    if args.use_small_model:
        model_name = 'gpt2'
        print(f"\nUsing small model for demonstration: {model_name}")
    
    # Training configuration
    train_config = config.get('training', {}).get('decoder_only', {})
    
    # Set random seed
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
    
    # Determine LoRA target modules based on model architecture
    if 'gpt2' in model_name.lower():
        lora_target_modules = ["c_attn", "c_proj"]
    elif 'llama' in model_name.lower() or 'mistral' in model_name.lower():
        lora_target_modules = lora_config.get(
            'target_modules',
            ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
    else:
        lora_target_modules = ["q_proj", "v_proj"]
    
    # Quantization settings
    use_quantization = args.use_quantization and 'gpt2' not in model_name.lower()
    
    model = DecoderOnlyMT(
        model_name=model_name,
        learning_rate=args.learning_rate or train_config.get('learning_rate', 2e-4),
        warmup_ratio=train_config.get('warmup_ratio', 0.03),
        weight_decay=0.01,
        max_length=model_config.get('max_length', 512),
        
        # LoRA configuration
        use_lora=args.use_lora and PEFT_AVAILABLE,
        lora_r=args.lora_r or lora_config.get('r', 16),
        lora_alpha=args.lora_alpha or lora_config.get('alpha', 32),
        lora_dropout=lora_config.get('dropout', 0.1),
        lora_target_modules=lora_target_modules,
        
        # Quantization
        use_quantization=use_quantization,
        load_in_4bit=True,
        
        # Generation settings
        num_beams=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        
        # Template
        template_type='generic' if 'gpt2' in model_name.lower() else 'llama'
    )
    
    print(f"  LoRA enabled: {args.use_lora and PEFT_AVAILABLE}")
    if args.use_lora and PEFT_AVAILABLE:
        print(f"  LoRA rank (r): {args.lora_r or lora_config.get('r', 16)}")
        print(f"  LoRA alpha: {args.lora_alpha or lora_config.get('alpha', 32)}")
        print(f"  Target modules: {lora_target_modules}")
    
    # =========================================
    # 3. Setup Data Module
    # =========================================
    print("\n[3/4] Setting up data module...")
    
    data_module = DecoderOnlyDataModule(
        train_sources=[s.source for s in train_samples],
        train_targets=[s.target for s in train_samples],
        val_sources=[s.source for s in val_samples],
        val_targets=[s.target for s in val_samples],
        tokenizer=model.tokenizer,
        batch_size=args.batch_size or train_config.get('batch_size', 2),
        max_length=model_config.get('max_length', 512),
        num_workers=args.num_workers,
        template_type='generic' if 'gpt2' in model_name.lower() else 'llama'
    )
    
    # =========================================
    # 4. Train Model
    # =========================================
    print("\n[4/4] Starting training...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="decoder-only-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            mode="min"
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name="decoder_only_training"
    )
    
    # Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=args.epochs or train_config.get('epochs', 5),
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if args.fp16 and accelerator == "gpu" else "32",
        accumulate_grad_batches=args.gradient_accumulation or train_config.get('gradient_accumulation_steps', 8),
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.5,
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # =========================================
    # 5. Save Model and Results
    # =========================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Save LoRA adapter or full model
    model_save_path = output_dir / "decoder_only_final"
    if args.use_lora and PEFT_AVAILABLE:
        model.save_adapter(model_save_path)
        print(f"\nLoRA adapter saved to: {model_save_path}")
    else:
        model.model.save_pretrained(model_save_path)
        model.tokenizer.save_pretrained(model_save_path)
        print(f"\nFull model saved to: {model_save_path}")
    
    # Test generation
    print("\nTest translations:")
    test_texts = [
        "Transfer open apps from your phone to your PC",
        "Battery saver mode will be turned off"
    ]
    
    model.eval()
    try:
        translations = model.generate(test_texts, max_new_tokens=100)
        
        for src, tgt in zip(test_texts, translations):
            print(f"  EN: {src}")
            print(f"  NL: {tgt}")
            print()
    except Exception as e:
        print(f"Generation test failed: {e}")
    
    # Save training info
    training_info = {
        'model_name': model_name,
        'use_lora': args.use_lora and PEFT_AVAILABLE,
        'lora_r': args.lora_r or lora_config.get('r', 16) if args.use_lora else None,
        'lora_alpha': args.lora_alpha or lora_config.get('alpha', 32) if args.use_lora else None,
        'epochs': trainer.current_epoch,
        'best_val_loss': float(trainer.callback_metrics.get('val_loss', 0)),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
    }
    
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Decoder-Only MT Model with LoRA")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="/root/motrola_assignment/data/processed")
    parser.add_argument("--output_dir", type=str, default="/root/motrola_assignment/outputs/decoder_only")
    parser.add_argument("--config", type=str, default="/root/motrola_assignment/config/config.yaml")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default=None, help="HuggingFace model name")
    parser.add_argument("--use_small_model", action="store_true", help="Use GPT-2 for testing")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument("--use_quantization", action="store_true", help="Use 4-bit quantization")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)
