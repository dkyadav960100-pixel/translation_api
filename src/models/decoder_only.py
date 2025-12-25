"""
Decoder-Only Model Module
=========================
Fine-tuning pipeline for decoder-only transformers with LoRA/PEFT
for machine translation.

Models supported:
- LLaMA 2 (meta-llama/Llama-2-7b)
- Mistral (mistralai/Mistral-7B)
- Phi (microsoft/phi-2)
- GPT-2 (for lightweight testing)

Uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA adapters.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)

# PEFT imports
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
        PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. LoRA fine-tuning not available.")


class InstructionMTDataset(Dataset):
    """
    Dataset for instruction-tuned translation with decoder-only models.
    
    Formats data as:
    <s>[INST] Translate the following English text to Dutch:
    {source_text} [/INST] {target_text}</s>
    """
    
    # Instruction templates
    INSTRUCTION_TEMPLATES = {
        'llama': (
            "<s>[INST] Translate the following English text to Dutch:\n"
            "{source} [/INST] {target}</s>"
        ),
        'mistral': (
            "<s>[INST] Translate the following English text to Dutch:\n"
            "{source} [/INST] {target}</s>"
        ),
        'phi': (
            "Instruct: Translate the following English text to Dutch:\n"
            "{source}\nOutput: {target}"
        ),
        'gpt2': (
            "Translate English to Dutch:\n"
            "English: {source}\n"
            "Dutch: {target}"
        ),
        'generic': (
            "### Instruction: Translate the following English text to Dutch.\n"
            "### Input: {source}\n"
            "### Response: {target}"
        )
    }
    
    def __init__(
        self,
        sources: List[str],
        targets: List[str],
        tokenizer,
        max_length: int = 512,
        template_type: str = 'generic'
    ):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = self.INSTRUCTION_TEMPLATES.get(template_type, self.INSTRUCTION_TEMPLATES['generic'])
        
        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source = self.sources[idx]
        target = self.targets[idx]
        
        # Format as instruction
        full_text = self.template.format(source=source, target=target)
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # For causal LM, labels are the same as input_ids
        # We mask the instruction part to only compute loss on the translation
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Create labels (same as input for causal LM)
        labels = input_ids.clone()
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Optionally mask the instruction part
        # Find where the response starts and mask everything before
        instruction_part = self.template.split('{target}')[0].format(source=source)
        instruction_tokens = self.tokenizer.encode(instruction_part, add_special_tokens=False)
        instruction_length = min(len(instruction_tokens), self.max_length - 1)
        labels[:instruction_length] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class DecoderOnlyMT(pl.LightningModule):
    """
    PyTorch Lightning module for decoder-only machine translation with LoRA.
    
    Features:
    - LoRA/QLoRA fine-tuning
    - 4-bit/8-bit quantization
    - Instruction tuning format
    - Memory-efficient training
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.01,
        max_length: int = 512,
        
        # LoRA configuration
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = None,
        
        # Quantization
        use_quantization: bool = False,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        
        # Generation
        num_beams: int = 1,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        
        # Template
        template_type: str = 'generic',
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Default LoRA target modules
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        # Quantization config
        bnb_config = None
        if use_quantization and PEFT_AVAILABLE:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit and not load_in_4bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            except Exception as e:
                print(f"Quantization not available: {e}")
                bnb_config = None
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {"trust_remote_code": True}
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Apply LoRA if available and requested
        if use_lora and PEFT_AVAILABLE:
            # Prepare model for k-bit training if quantized
            if use_quantization:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # LoRA configuration
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            self._print_trainable_parameters()
        
        # Instruction template
        self.template_type = template_type
        self.instruction_template = InstructionMTDataset.INSTRUCTION_TEMPLATES.get(
            template_type, 
            InstructionMTDataset.INSTRUCTION_TEMPLATES['generic']
        )
        
        # Validation outputs
        self.validation_step_outputs = []
    
    def _print_trainable_parameters(self):
        """Print number of trainable parameters."""
        trainable_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {100 * trainable_params / all_params:.2f}"
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
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
        self.validation_step_outputs.append({'val_loss': loss})
        
        return {'val_loss': loss}
    
    def on_validation_epoch_end(self):
        """Compute metrics at end of validation epoch."""
        if self.validation_step_outputs:
            avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            self.log('val_loss_epoch', avg_loss)
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        params = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Calculate warmup steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
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
        max_new_tokens: int = 256,
        num_beams: int = None,
        do_sample: bool = None,
        temperature: float = None,
        top_p: float = None,
        **kwargs
    ) -> List[str]:
        """
        Generate translations for input texts.
        
        Args:
            texts: List of source texts (English)
            max_new_tokens: Maximum new tokens to generate
            num_beams: Beam search width
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            
        Returns:
            List of translated texts (Dutch)
        """
        num_beams = num_beams or self.hparams.num_beams
        do_sample = do_sample if do_sample is not None else self.hparams.do_sample
        temperature = temperature or self.hparams.temperature
        top_p = top_p or self.hparams.top_p
        
        translations = []
        
        for text in texts:
            # Format as instruction (only source part)
            prompt = self.instruction_template.split('{target}')[0].format(source=text)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.hparams.max_length - max_new_tokens
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode and extract translation
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the translation part
            translation = self._extract_translation(full_output)
            translations.append(translation)
        
        return translations
    
    def _extract_translation(self, full_output: str) -> str:
        """Extract translation from the full generated output."""
        # Try different extraction patterns based on template
        extraction_patterns = [
            ("### Response:", None),
            ("[/INST]", "</s>"),
            ("Output:", None),
            ("Dutch:", None),
        ]
        
        for start_marker, end_marker in extraction_patterns:
            if start_marker in full_output:
                start_idx = full_output.find(start_marker) + len(start_marker)
                translation = full_output[start_idx:]
                
                if end_marker and end_marker in translation:
                    translation = translation[:translation.find(end_marker)]
                
                return translation.strip()
        
        # Fallback: return everything after last newline or the whole output
        lines = full_output.strip().split('\n')
        return lines[-1].strip()
    
    def save_adapter(self, path: str):
        """Save LoRA adapter weights."""
        if PEFT_AVAILABLE and hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
    
    @classmethod
    def load_adapter(cls, base_model_name: str, adapter_path: str):
        """Load model with saved LoRA adapter."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT not available for loading adapters")
        
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(model, adapter_path)
        
        return model


class DecoderOnlyDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for decoder-only MT."""
    
    def __init__(
        self,
        train_sources: List[str],
        train_targets: List[str],
        val_sources: List[str],
        val_targets: List[str],
        tokenizer,
        batch_size: int = 2,
        max_length: int = 512,
        num_workers: int = 4,
        template_type: str = 'generic'
    ):
        super().__init__()
        self.train_sources = train_sources
        self.train_targets = train_targets
        self.val_sources = val_sources
        self.val_targets = val_targets
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.template_type = template_type
    
    def setup(self, stage=None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = InstructionMTDataset(
                self.train_sources, self.train_targets, self.tokenizer,
                self.max_length, self.template_type
            )
            self.val_dataset = InstructionMTDataset(
                self.val_sources, self.val_targets, self.tokenizer,
                self.max_length, self.template_type
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


if __name__ == "__main__":
    # Test model loading with a small model
    print("Testing DecoderOnlyMT with GPT-2...")
    
    model = DecoderOnlyMT(
        model_name="gpt2",
        use_lora=PEFT_AVAILABLE,
        use_quantization=False,  # GPT-2 doesn't need quantization
        lora_target_modules=["c_attn", "c_proj"],
        learning_rate=2e-4
    )
    
    print(f"Model loaded: {model.hparams.model_name}")
    
    # Test generation
    test_texts = [
        "Transfer apps from phone to PC",
        "Battery saver mode"
    ]
    
    model.eval()
    try:
        translations = model.generate(test_texts, max_new_tokens=50)
        print("\nTest translations:")
        for src, tgt in zip(test_texts, translations):
            print(f"  EN: {src}")
            print(f"  NL: {tgt}")
            print()
    except Exception as e:
        print(f"Generation test failed (expected for untrained model): {e}")
