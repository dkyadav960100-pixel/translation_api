#!/usr/bin/env python3
"""
Encoder-Decoder Training Script (Updated for WMT16)
====================================================
Train MarianMT/mBART model for EN-NL translation using WMT16 Europarl data.

Challenge 1, Part A: Domain-Specific Fine-Tuning with Encoder-Decoder Model

Training: WMT16 Europarl EN-NL
Testing: Dataset_Challenge_1.xlsx (software domain)
"""

import os
import sys
from pathlib import Path
import argparse
import json
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer, get_linear_schedule_with_warmup
import yaml


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load data from JSONL file."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


class TranslationDataset(Dataset):
    """PyTorch Dataset for translation pairs."""
    
    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        max_source_length: int = 128,
        max_target_length: int = 128
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        source = sample['source']
        target = sample['target']
        
        # Tokenize
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


class MarianMTLightning(pl.LightningModule):
    """PyTorch Lightning module for MarianMT fine-tuning."""
    
    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-nl",
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        num_training_steps: int = 10000
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = MarianMTModel.from_pretrained(model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.num_training_steps = num_training_steps
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_training_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def generate(self, texts: List[str], max_length: int = 128, num_beams: int = 4) -> List[str]:
        """Generate translations for a list of texts."""
        self.eval()
        device = next(self.parameters()).device
        
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main(args):
    """Main training pipeline."""
    print("=" * 60)
    print("ENCODER-DECODER MODEL TRAINING (MarianMT)")
    print("Challenge 1, Part A: Domain-Specific Fine-Tuning")
    print("=" * 60)
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # =========================================
    # 1. Load Data
    # =========================================
    print("\n[1/4] Loading data...")
    
    data_dir = Path(args.data_dir)
    
    train_samples = load_jsonl(data_dir / "train.jsonl")
    val_samples = load_jsonl(data_dir / "val.jsonl")
    test_samples = load_jsonl(data_dir / "test.jsonl")
    
    print(f"  📊 Train: {len(train_samples):,} samples (WMT16 Europarl)")
    print(f"  📊 Val: {len(val_samples):,} samples (WMT16 Europarl)")
    print(f"  📊 Test: {len(test_samples):,} samples (Software Domain)")
    
    # Limit training samples if needed
    if args.max_train_samples and len(train_samples) > args.max_train_samples:
        train_samples = train_samples[:args.max_train_samples]
        print(f"  ⚠️ Limited train samples to: {len(train_samples):,}")
    
    # =========================================
    # 2. Initialize Model
    # =========================================
    print(f"\n[2/4] Initializing model: {args.model_name}")
    
    # Calculate total training steps
    num_training_steps = (len(train_samples) // args.batch_size) * args.epochs
    
    model = MarianMTLightning(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        num_training_steps=num_training_steps
    )
    
    print(f"  Model loaded: {args.model_name}")
    print(f"  Total training steps: {num_training_steps:,}")
    
    # =========================================
    # 3. Create Data Loaders
    # =========================================
    print("\n[3/4] Creating data loaders...")
    
    train_dataset = TranslationDataset(
        train_samples,
        model.tokenizer,
        max_source_length=args.max_length,
        max_target_length=args.max_length
    )
    
    val_dataset = TranslationDataset(
        val_samples,
        model.tokenizer,
        max_source_length=args.max_length,
        max_target_length=args.max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
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
            filename="marian-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last=True
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
        name="marian_training"
    )
    
    # Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if args.fp16 and accelerator == "gpu" else "32",
        accumulate_grad_batches=args.gradient_accumulation,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=0.5 if len(train_samples) > 10000 else 1.0,
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # =========================================
    # 5. Save and Test Model
    # =========================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Save final model
    model_save_path = output_dir / "marian_finetuned"
    model.model.save_pretrained(model_save_path)
    model.tokenizer.save_pretrained(model_save_path)
    print(f"\n✅ Model saved to: {model_save_path}")
    
    # Test translations on software domain
    print("\n📝 Sample Translations (Software Domain):")
    print("-" * 60)
    
    test_texts = [s['source'] for s in test_samples[:5]]
    test_refs = [s['target'] for s in test_samples[:5]]
    
    model.eval()
    translations = model.generate(test_texts)
    
    for src, ref, hyp in zip(test_texts, test_refs, translations):
        print(f"EN: {src}")
        print(f"REF: {ref}")
        print(f"HYP: {hyp}")
        print()
    
    # Save training info
    training_info = {
        'model_name': args.model_name,
        'epochs': trainer.current_epoch,
        'best_val_loss': float(trainer.callback_metrics.get('val_loss', 0)),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
    }
    
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n📊 Training info saved to: {output_dir / 'training_info.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MarianMT for EN-NL Translation")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, 
                        default="/root/motrola_assignment/data/processed")
    parser.add_argument("--output_dir", type=str, 
                        default="/root/motrola_assignment/outputs/encoder_decoder")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, 
                        default="Helsinki-NLP/opus-mt-en-nl")
    parser.add_argument("--max_length", type=int, default=128)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)
