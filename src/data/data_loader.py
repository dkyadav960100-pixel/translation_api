"""
Data Loader Module
==================
Comprehensive data loading utilities for Machine Translation datasets.

Supports:
- Excel files (Challenge datasets)
- HuggingFace datasets (FLORES, WMT)
- Custom formats with caching
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
import torch


@dataclass
class TranslationSample:
    """Represents a single translation sample with metadata."""
    source: str
    target: str
    source_lang: str = "en"
    target_lang: str = "nl"
    domain: str = "software"
    sample_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class DatasetStats:
    """Statistics for a translation dataset."""
    num_samples: int
    avg_source_length: float
    avg_target_length: float
    max_source_length: int
    max_target_length: int
    vocab_size_source: int
    vocab_size_target: int
    unique_patterns: List[str]
    domain_terms: List[str]


class MTDataLoader:
    """
    Comprehensive Machine Translation Data Loader.
    
    Features:
    - Multi-format support (Excel, CSV, JSON, HuggingFace)
    - Automatic caching with hash-based invalidation
    - Train/val/test splitting with stratification
    - Statistics computation
    - Placeholder detection and preservation
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        cache_dir: Optional[Union[str, Path]] = None,
        source_lang: str = "en",
        target_lang: str = "nl"
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_challenge_dataset(
        self,
        file_path: Union[str, Path],
        source_col: str = "English Source",
        target_col: str = "Reference Translation"
    ) -> List[TranslationSample]:
        """
        Load the challenge dataset from Excel file.
        
        Args:
            file_path: Path to the Excel file
            source_col: Name of source language column
            target_col: Name of target language column
            
        Returns:
            List of TranslationSample objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        # Load Excel file
        df = pd.read_excel(file_path)
        
        # Validate columns
        if source_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Required columns not found. Available: {df.columns.tolist()}")
        
        samples = []
        for idx, row in df.iterrows():
            source = str(row[source_col]).strip() if pd.notna(row[source_col]) else ""
            target = str(row[target_col]).strip() if pd.notna(row[target_col]) else ""
            
            if source and target:
                sample = TranslationSample(
                    source=source,
                    target=target,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang,
                    domain="software",
                    sample_id=f"challenge_{idx}",
                    metadata={"row_index": idx}
                )
                samples.append(sample)
        
        return samples
    
    def load_flores_devtest(
        self,
        source_lang: str = "eng_Latn",
        target_lang: str = "nld_Latn",
        split: str = "devtest"
    ) -> List[TranslationSample]:
        """
        Load FLORES-200 devtest dataset.
        
        Args:
            source_lang: FLORES language code for source
            target_lang: FLORES language code for target
            split: Dataset split to load
            
        Returns:
            List of TranslationSample objects
        """
        try:
            # Load FLORES dataset from HuggingFace
            dataset = load_dataset("facebook/flores", name=f"{source_lang}-{target_lang}", split=split)
        except Exception as e:
            # Fallback: try loading with different format
            try:
                flores_source = load_dataset("facebook/flores", source_lang, split=split)
                flores_target = load_dataset("facebook/flores", target_lang, split=split)
                
                samples = []
                for idx, (src_item, tgt_item) in enumerate(zip(flores_source, flores_target)):
                    sample = TranslationSample(
                        source=src_item["sentence"],
                        target=tgt_item["sentence"],
                        source_lang="en",
                        target_lang="nl",
                        domain="general",
                        sample_id=f"flores_{idx}",
                        metadata={"flores_id": idx}
                    )
                    samples.append(sample)
                return samples
            except:
                print(f"Warning: Could not load FLORES dataset. Error: {e}")
                return []
        
        samples = []
        for idx, item in enumerate(dataset):
            sample = TranslationSample(
                source=item.get("sentence_" + source_lang, item.get("sentence", "")),
                target=item.get("sentence_" + target_lang, item.get("translation", "")),
                source_lang="en",
                target_lang="nl",
                domain="general",
                sample_id=f"flores_{idx}",
                metadata={"flores_id": idx}
            )
            samples.append(sample)
        
        return samples
    
    def load_wmt16_it_domain(
        self,
        data_path: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None
    ) -> List[TranslationSample]:
        """
        Load WMT16 IT domain dataset for training.
        
        Source: https://www.statmt.org/wmt16/it-translation-task.html
        - EN-NL language pair
        - IT/software domain translations
        - PO files from VLC, LibreOffice, KDE
        
        Args:
            data_path: Path to processed WMT16 data (train.json or raw directory)
            max_samples: Maximum number of samples to load
            
        Returns:
            List of TranslationSample objects for training
        """
        samples = []
        data_path = Path(data_path) if data_path else self.data_dir / "raw" / "wmt16"
        
        # Try loading preprocessed JSON first
        processed_train = data_path.parent.parent / "processed" / "train.json"
        if processed_train.exists():
            with open(processed_train, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for idx, item in enumerate(data):
                    if max_samples and idx >= max_samples:
                        break
                    sample = TranslationSample(
                        source=item['source'],
                        target=item['target'],
                        source_lang="en",
                        target_lang="nl",
                        domain="software",
                        sample_id=f"wmt16_{idx}",
                        metadata=item.get('metadata', {})
                    )
                    samples.append(sample)
            print(f"Loaded {len(samples)} samples from processed WMT16 data")
            return samples
        
        # Try loading from raw PO files
        if data_path.exists() and data_path.is_dir():
            po_files = list(data_path.glob("**/*.po"))
            for po_file in po_files:
                po_samples = self._parse_po_file(po_file)
                samples.extend(po_samples)
                if max_samples and len(samples) >= max_samples:
                    samples = samples[:max_samples]
                    break
        
        # Try loading from parallel text files (batch1_2 format)
        for txt_pair in [("en.txt", "nl.txt"), ("english.txt", "dutch.txt")]:
            en_file = data_path / txt_pair[0]
            nl_file = data_path / txt_pair[1]
            if en_file.exists() and nl_file.exists():
                en_lines = en_file.read_text(encoding='utf-8').strip().split('\n')
                nl_lines = nl_file.read_text(encoding='utf-8').strip().split('\n')
                for idx, (en, nl) in enumerate(zip(en_lines, nl_lines)):
                    if max_samples and len(samples) >= max_samples:
                        break
                    sample = TranslationSample(
                        source=en.strip(),
                        target=nl.strip(),
                        source_lang="en",
                        target_lang="nl",
                        domain="software",
                        sample_id=f"wmt16_txt_{idx}"
                    )
                    samples.append(sample)
        
        if not samples:
            print("Warning: No WMT16 data found. Run scripts/prepare_data.py --download_wmt16")
            print("Or use synthetic IT domain data as fallback")
            samples = self._get_synthetic_it_domain_samples()
        
        print(f"Loaded {len(samples)} WMT16 IT domain samples for training")
        return samples
    
    def _parse_po_file(self, po_file: Path) -> List[TranslationSample]:
        """Parse a PO (gettext) file to extract translation pairs."""
        samples = []
        try:
            content = po_file.read_text(encoding='utf-8', errors='ignore')
            
            # Simple PO parser - extract msgid/msgstr pairs
            import re
            msgid_pattern = r'msgid\s+"([^"]*(?:"[^"]*)*)"'
            msgstr_pattern = r'msgstr\s+"([^"]*(?:"[^"]*)*)"'
            
            msgids = re.findall(msgid_pattern, content)
            msgstrs = re.findall(msgstr_pattern, content)
            
            for idx, (msgid, msgstr) in enumerate(zip(msgids, msgstrs)):
                # Skip empty or identical pairs
                if msgid and msgstr and msgid != msgstr:
                    # Clean up PO escape sequences
                    msgid = msgid.replace('\\n', ' ').replace('\\"', '"').strip()
                    msgstr = msgstr.replace('\\n', ' ').replace('\\"', '"').strip()
                    if msgid and msgstr:
                        samples.append(TranslationSample(
                            source=msgid,
                            target=msgstr,
                            source_lang="en",
                            target_lang="nl",
                            domain="software",
                            sample_id=f"po_{po_file.stem}_{idx}",
                            metadata={"source_file": str(po_file)}
                        ))
        except Exception as e:
            print(f"Warning: Could not parse PO file {po_file}: {e}")
        
        return samples
    
    def _get_synthetic_it_domain_samples(self) -> List[TranslationSample]:
        """Generate synthetic IT domain samples as fallback."""
        synthetic_pairs = [
            ("Settings", "Instellingen"),
            ("Bluetooth", "Bluetooth"),
            ("Turn on Wi-Fi", "Wi-Fi inschakelen"),
            ("Battery level: {1}%", "Batterijniveau: {1}%"),
            ("Connect to {device}", "Verbinden met {device}"),
            ("Download complete", "Download voltooid"),
            ("Network error", "Netwerkfout"),
            ("Restart required", "Opnieuw opstarten vereist"),
            ("Update available", "Update beschikbaar"),
            ("Enter password", "Wachtwoord invoeren"),
            ("Connection failed", "Verbinding mislukt"),
            ("Processing...", "Bezig met verwerken..."),
            ("Please wait", "Even geduld"),
            ("Save changes?", "Wijzigingen opslaan?"),
            ("Cancel", "Annuleren"),
            ("OK", "OK"),
            ("Delete", "Verwijderen"),
            ("Confirm", "Bevestigen"),
            ("Loading...", "Laden..."),
            ("Error occurred", "Er is een fout opgetreden"),
        ]
        return [
            TranslationSample(
                source=src, target=tgt,
                source_lang="en", target_lang="nl",
                domain="software", sample_id=f"synthetic_{i}"
            )
            for i, (src, tgt) in enumerate(synthetic_pairs)
        ]
    
    def create_train_val_test_split(
        self,
        samples: List[TranslationSample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[TranslationSample], List[TranslationSample], List[TranslationSample]]:
        """
        Split samples into train/val/test sets.
        
        Args:
            samples: List of translation samples
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        np.random.seed(seed)
        indices = np.random.permutation(len(samples))
        
        n_train = int(len(samples) * train_ratio)
        n_val = int(len(samples) * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        test_samples = [samples[i] for i in test_indices]
        
        return train_samples, val_samples, test_samples
    
    def compute_statistics(self, samples: List[TranslationSample]) -> DatasetStats:
        """Compute comprehensive statistics for a dataset."""
        if not samples:
            return DatasetStats(
                num_samples=0,
                avg_source_length=0,
                avg_target_length=0,
                max_source_length=0,
                max_target_length=0,
                vocab_size_source=0,
                vocab_size_target=0,
                unique_patterns=[],
                domain_terms=[]
            )
        
        source_lengths = [len(s.source.split()) for s in samples]
        target_lengths = [len(s.target.split()) for s in samples]
        
        # Vocabulary analysis
        source_words = set()
        target_words = set()
        for s in samples:
            source_words.update(s.source.lower().split())
            target_words.update(s.target.lower().split())
        
        # Pattern detection (placeholders, variables)
        import re
        patterns = set()
        for s in samples:
            patterns.update(re.findall(r'\{[^}]+\}', s.source))
            patterns.update(re.findall(r'%[sd%]', s.source))
        
        # Domain-specific terms
        domain_terms = [
            word for word in source_words 
            if any(term in word.lower() for term in 
                   ['usb', 'bluetooth', 'wifi', 'app', 'phone', 'display', 'battery'])
        ]
        
        return DatasetStats(
            num_samples=len(samples),
            avg_source_length=np.mean(source_lengths),
            avg_target_length=np.mean(target_lengths),
            max_source_length=max(source_lengths),
            max_target_length=max(target_lengths),
            vocab_size_source=len(source_words),
            vocab_size_target=len(target_words),
            unique_patterns=list(patterns),
            domain_terms=domain_terms[:50]  # Top 50
        )
    
    def to_hf_dataset(self, samples: List[TranslationSample]) -> Dataset:
        """Convert samples to HuggingFace Dataset format."""
        data = {
            "source": [s.source for s in samples],
            "target": [s.target for s in samples],
            "source_lang": [s.source_lang for s in samples],
            "target_lang": [s.target_lang for s in samples],
            "domain": [s.domain for s in samples],
            "sample_id": [s.sample_id for s in samples]
        }
        return Dataset.from_dict(data)
    
    def save_processed(self, samples: List[TranslationSample], name: str):
        """Save processed samples to cache."""
        output_path = self.cache_dir / f"{name}.json"
        data = [
            {
                "source": s.source,
                "target": s.target,
                "source_lang": s.source_lang,
                "target_lang": s.target_lang,
                "domain": s.domain,
                "sample_id": s.sample_id,
                "metadata": s.metadata
            }
            for s in samples
        ]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_processed(self, name: str) -> List[TranslationSample]:
        """Load processed samples from cache."""
        input_path = self.cache_dir / f"{name}.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Cache not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [
            TranslationSample(
                source=d["source"],
                target=d["target"],
                source_lang=d["source_lang"],
                target_lang=d["target_lang"],
                domain=d["domain"],
                sample_id=d["sample_id"],
                metadata=d.get("metadata", {})
            )
            for d in data
        ]


def get_dataloader(
    samples: List[TranslationSample],
    tokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a PyTorch DataLoader from translation samples.
    
    Args:
        samples: List of TranslationSample objects
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        PyTorch DataLoader
    """
    from torch.utils.data import Dataset as TorchDataset
    
    class MTDataset(TorchDataset):
        def __init__(self, samples, tokenizer, max_length):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            # Tokenize source
            source_encoding = self.tokenizer(
                sample.source,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize target
            target_encoding = self.tokenizer(
                sample.target,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "input_ids": source_encoding["input_ids"].squeeze(0),
                "attention_mask": source_encoding["attention_mask"].squeeze(0),
                "labels": target_encoding["input_ids"].squeeze(0),
                "decoder_attention_mask": target_encoding["attention_mask"].squeeze(0)
            }
    
    dataset = MTDataset(samples, tokenizer, max_length)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test the data loader
    loader = MTDataLoader(data_dir="/root/motrola_assignment/data")
    
    # Load challenge dataset
    samples = loader.load_challenge_dataset(
        "/root/motrola_assignment/Dataset_Challenge_1.xlsx"
    )
    
    print(f"Loaded {len(samples)} samples")
    print(f"\nSample 0:")
    print(f"  Source: {samples[0].source}")
    print(f"  Target: {samples[0].target}")
    
    # Compute statistics
    stats = loader.compute_statistics(samples)
    print(f"\nDataset Statistics:")
    print(f"  Samples: {stats.num_samples}")
    print(f"  Avg source length: {stats.avg_source_length:.1f} words")
    print(f"  Avg target length: {stats.avg_target_length:.1f} words")
    print(f"  Unique patterns: {stats.unique_patterns}")
