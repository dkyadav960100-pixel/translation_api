"""
Encoder-Decoder Model Module
=============================
Fine-tuning pipeline for encoder-decoder transformers for machine translation.

Models supported:
- MarianMT (Helsinki-NLP/opus-mt-en-nl)
- mBART (facebook/mbart-large-50)
- T5 (google/t5-base)
- NLLB (facebook/nllb-200-distilled-600M)

Uses PyTorch Lightning for training infrastructure.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from transformers.optimization import Adafactor


class MTDataset(Dataset):
    """PyTorch Dataset for Machine Translation."""
    
    def __init__(
        self,
        sources: List[str],
        targets: List[str],
        tokenizer,
        max_source_length: int = 256,
        max_target_length: int = 256
    ):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source = self.sources[idx]
        target = self.targets[idx]
        
        # Tokenize source
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(
                target,
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Prepare labels (replace padding token id with -100 for loss computation)
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


class EncoderDecoderMT(pl.LightningModule):
    """
    PyTorch Lightning module for encoder-decoder machine translation.
    
    Supports:
    - MarianMT (specialized for specific language pairs)
    - mBART (multilingual)
    - T5 (text-to-text)
    - NLLB (No Language Left Behind)
    """
    
    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-nl",
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        max_source_length: int = 256,
        max_target_length: int = 256,
        num_beams: int = 4,
        use_adafactor: bool = False,
        freeze_encoder: bool = False,
        freeze_layers: int = 0,
        label_smoothing: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model selection based on name
        if 'marian' in model_name.lower() or 'opus-mt' in model_name.lower():
            self.model = MarianMTModel.from_pretrained(model_name)
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        elif 'mbart' in model_name.lower():
            self.model = MBartForConditionalGeneration.from_pretrained(model_name)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            # Set language codes
            self.tokenizer.src_lang = "en_XX"
            self.tokenizer.tgt_lang = "nl_XX"
        elif 't5' in model_name.lower():
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        else:
            # Generic seq2seq model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.model.get_encoder().parameters():
                param.requires_grad = False
        
        # Freeze specific layers
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        # Loss configuration
        self.label_smoothing = label_smoothing
        
        # Store for validation outputs
        self.validation_step_outputs = []
    
    def _freeze_layers(self, num_layers: int):
        """Freeze first n encoder layers."""
        encoder = self.model.get_encoder()
        if hasattr(encoder, 'layers'):
            layers = encoder.layers
        elif hasattr(encoder, 'layer'):
            layers = encoder.layer
        else:
            return
        
        for i, layer in enumerate(layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            loss = self._compute_smoothed_loss(outputs.logits, batch['labels'])
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        # Store outputs for epoch end
        self.validation_step_outputs.append({'val_loss': loss})
        
        return {'val_loss': loss}
    
    def on_validation_epoch_end(self):
        """Compute metrics at end of validation epoch."""
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        self.log('val_loss_epoch', avg_loss)
        self.validation_step_outputs.clear()
    
    def _compute_smoothed_loss(self, logits, labels):
        """Compute loss with label smoothing."""
        vocab_size = logits.size(-1)
        
        # Create smoothed distribution
        smooth_labels = torch.full_like(logits, self.label_smoothing / (vocab_size - 1))
        smooth_labels.scatter_(-1, labels.unsqueeze(-1), 1.0 - self.label_smoothing)
        
        # Mask padding
        mask = labels != -100
        smooth_labels = smooth_labels * mask.unsqueeze(-1)
        
        # Compute cross entropy
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -torch.sum(smooth_labels * log_probs, dim=-1)
        loss = loss.sum() / mask.sum()
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Get parameters to optimize
        params = [p for p in self.parameters() if p.requires_grad]
        
        if self.hparams.use_adafactor:
            # Adafactor - memory efficient optimizer
            optimizer = Adafactor(
                params,
                lr=self.hparams.learning_rate,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False
            )
        else:
            # AdamW
            optimizer = torch.optim.AdamW(
                params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        
        # Scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def generate(
        self,
        texts: List[str],
        max_length: int = None,
        num_beams: int = None,
        **kwargs
    ) -> List[str]:
        """
        Generate translations for input texts.
        
        Args:
            texts: List of source texts
            max_length: Maximum output length
            num_beams: Beam search width
            
        Returns:
            List of translated texts
        """
        max_length = max_length or self.hparams.max_target_length
        num_beams = num_beams or self.hparams.num_beams
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            max_length=self.hparams.max_source_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                **kwargs
            )
        
        # Decode
        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return translations


class MTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Machine Translation."""
    
    def __init__(
        self,
        train_sources: List[str],
        train_targets: List[str],
        val_sources: List[str],
        val_targets: List[str],
        tokenizer,
        batch_size: int = 8,
        max_source_length: int = 256,
        max_target_length: int = 256,
        num_workers: int = 4,
        test_sources: List[str] = None,
        test_targets: List[str] = None
    ):
        super().__init__()
        self.train_sources = train_sources
        self.train_targets = train_targets
        self.val_sources = val_sources
        self.val_targets = val_targets
        self.test_sources = test_sources
        self.test_targets = test_targets
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = MTDataset(
                self.train_sources, self.train_targets, self.tokenizer,
                self.max_source_length, self.max_target_length
            )
            self.val_dataset = MTDataset(
                self.val_sources, self.val_targets, self.tokenizer,
                self.max_source_length, self.max_target_length
            )
        
        if stage == 'test' or stage is None:
            if self.test_sources and self.test_targets:
                self.test_dataset = MTDataset(
                    self.test_sources, self.test_targets, self.tokenizer,
                    self.max_source_length, self.max_target_length
                )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        return None


def get_trainer(
    output_dir: str = "outputs/models",
    max_epochs: int = 10,
    precision: str = "16-mixed",
    accumulate_grad_batches: int = 4,
    gradient_clip_val: float = 1.0,
    early_stopping_patience: int = 3,
    log_every_n_steps: int = 10,
    val_check_interval: float = 0.5,
    **kwargs
) -> pl.Trainer:
    """
    Create a PyTorch Lightning Trainer with best practices.
    
    Args:
        output_dir: Directory for checkpoints and logs
        max_epochs: Maximum training epochs
        precision: Training precision (16-mixed, 32, bf16-mixed)
        accumulate_grad_batches: Gradient accumulation steps
        gradient_clip_val: Gradient clipping value
        early_stopping_patience: Early stopping patience
        
    Returns:
        Configured Trainer instance
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="mt-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            mode="min"
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name="mt_training"
    )
    
    # Determine accelerator
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        enable_progress_bar=True,
        **kwargs
    )
    
    return trainer


if __name__ == "__main__":
    # Test model loading
    print("Testing EncoderDecoderMT...")
    
    model = EncoderDecoderMT(
        model_name="Helsinki-NLP/opus-mt-en-nl",
        learning_rate=5e-5
    )
    
    print(f"Model loaded: {model.hparams.model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test generation
    test_texts = [
        "Transfer open apps from your phone to your PC",
        "Battery saver mode will be turned off"
    ]
    
    model.eval()
    translations = model.generate(test_texts)
    
    print("\nTest translations:")
    for src, tgt in zip(test_texts, translations):
        print(f"  EN: {src}")
        print(f"  NL: {tgt}")
        print()
