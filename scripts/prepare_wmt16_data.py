#!/usr/bin/env python3
"""
WMT16 Data Preparation Script
=============================
Prepare Europarl EN-NL data from WMT16 for training.

Data structure:
- Training: WMT16 Europarl (europarl-v7.nl-en.en/nl) - ~2M sentence pairs
- Testing: Dataset_Challenge_1.xlsx (software domain)
- General evaluation: FLORES-devtest
"""

import os
import sys
from pathlib import Path
import json
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


@dataclass
class TranslationPair:
    source: str
    target: str
    domain: str = "general"
    sample_id: str = ""


def load_europarl_data(
    raw_data_dir: Path,
    max_samples: int = None,
    min_length: int = 5,
    max_length: int = 200
) -> List[TranslationPair]:
    """
    Load Europarl parallel corpus from WMT16.
    
    Args:
        raw_data_dir: Directory containing europarl-v7.nl-en.en and .nl files
        max_samples: Maximum number of samples to load (for memory efficiency)
        min_length: Minimum sentence length (words)
        max_length: Maximum sentence length (words)
    
    Returns:
        List of TranslationPair objects
    """
    en_file = raw_data_dir / "europarl-v7.nl-en.en"
    nl_file = raw_data_dir / "europarl-v7.nl-en.nl"
    
    if not en_file.exists() or not nl_file.exists():
        raise FileNotFoundError(
            f"Europarl files not found in {raw_data_dir}. "
            "Please download from https://www.statmt.org/wmt16/it-translation-task.html"
        )
    
    print(f"Loading Europarl data from {raw_data_dir}...")
    
    samples = []
    skipped = {"too_short": 0, "too_long": 0, "empty": 0, "mismatch": 0}
    
    with open(en_file, 'r', encoding='utf-8', errors='ignore') as f_en, \
         open(nl_file, 'r', encoding='utf-8', errors='ignore') as f_nl:
        
        for idx, (en_line, nl_line) in enumerate(zip(f_en, f_nl)):
            if max_samples and len(samples) >= max_samples:
                break
            
            en_text = en_line.strip()
            nl_text = nl_line.strip()
            
            # Skip empty lines
            if not en_text or not nl_text:
                skipped["empty"] += 1
                continue
            
            # Check length constraints
            en_words = len(en_text.split())
            nl_words = len(nl_text.split())
            
            if en_words < min_length or nl_words < min_length:
                skipped["too_short"] += 1
                continue
            
            if en_words > max_length or nl_words > max_length:
                skipped["too_long"] += 1
                continue
            
            # Check for reasonable length ratio (0.5 to 2.0)
            ratio = nl_words / en_words
            if ratio < 0.3 or ratio > 3.0:
                skipped["mismatch"] += 1
                continue
            
            samples.append(TranslationPair(
                source=en_text,
                target=nl_text,
                domain="general",
                sample_id=f"europarl_{idx}"
            ))
            
            if (idx + 1) % 500000 == 0:
                print(f"  Processed {idx + 1:,} lines, kept {len(samples):,} samples...")
    
    print(f"\n✅ Loaded {len(samples):,} samples from Europarl")
    print(f"   Skipped: {skipped}")
    
    return samples


def load_challenge_test_data(file_path: Path) -> List[TranslationPair]:
    """Load the software domain test dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {file_path}")
    
    df = pd.read_excel(file_path)
    
    # Try different column name variations
    source_col = None
    target_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'english' in col_lower or 'source' in col_lower:
            source_col = col
        elif 'dutch' in col_lower or 'reference' in col_lower or 'translation' in col_lower:
            target_col = col
    
    if source_col is None or target_col is None:
        # Fallback: assume first two columns
        source_col = df.columns[0]
        target_col = df.columns[1]
    
    print(f"Using columns: source='{source_col}', target='{target_col}'")
    
    samples = []
    for idx, row in df.iterrows():
        source = str(row[source_col]).strip() if pd.notna(row[source_col]) else ""
        target = str(row[target_col]).strip() if pd.notna(row[target_col]) else ""
        
        if source and target:
            samples.append(TranslationPair(
                source=source,
                target=target,
                domain="software",
                sample_id=f"test_{idx}"
            ))
    
    print(f"✅ Loaded {len(samples)} test samples (software domain)")
    return samples


def create_train_val_split(
    samples: List[TranslationPair],
    val_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[List[TranslationPair], List[TranslationPair]]:
    """Split samples into train and validation sets."""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    val_size = int(len(shuffled) * val_ratio)
    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]
    
    return train_samples, val_samples


def save_samples(samples: List[TranslationPair], output_path: Path, format: str = "jsonl"):
    """Save samples to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(asdict(sample), ensure_ascii=False) + '\n')
    elif format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(s) for s in samples], f, ensure_ascii=False, indent=2)
    elif format == "tsv":
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(f"{sample.source}\t{sample.target}\n")
    
    print(f"💾 Saved {len(samples):,} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare WMT16 data for MT training")
    parser.add_argument("--raw_data_dir", type=str, 
                        default="/root/motrola_assignment/raw_data",
                        help="Directory containing raw Europarl files")
    parser.add_argument("--output_dir", type=str,
                        default="/root/motrola_assignment/data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--test_file", type=str,
                        default="/root/motrola_assignment/Dataset_Challenge_1.xlsx",
                        help="Path to test dataset")
    parser.add_argument("--max_train_samples", type=int, default=500000,
                        help="Maximum training samples (for efficiency)")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("WMT16 DATA PREPARATION PIPELINE")
    print("=" * 60)
    
    raw_data_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    
    # 1. Load Europarl training data
    print("\n[1/4] Loading Europarl training data...")
    train_data = load_europarl_data(
        raw_data_dir,
        max_samples=args.max_train_samples,
        min_length=3,
        max_length=150
    )
    
    # 2. Create train/val split
    print("\n[2/4] Creating train/validation split...")
    train_samples, val_samples = create_train_val_split(
        train_data,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    print(f"   Train: {len(train_samples):,} samples")
    print(f"   Val: {len(val_samples):,} samples")
    
    # 3. Load test data (software domain)
    print("\n[3/4] Loading test data (software domain)...")
    test_samples = load_challenge_test_data(Path(args.test_file))
    
    # 4. Save all splits
    print("\n[4/4] Saving processed data...")
    save_samples(train_samples, output_dir / "train.jsonl", format="jsonl")
    save_samples(val_samples, output_dir / "val.jsonl", format="jsonl")
    save_samples(test_samples, output_dir / "test.jsonl", format="jsonl")
    
    # Also save in TSV format for some tools
    save_samples(train_samples, output_dir / "train.tsv", format="tsv")
    save_samples(val_samples, output_dir / "val.tsv", format="tsv")
    save_samples(test_samples, output_dir / "test.tsv", format="tsv")
    
    # Summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"""
📊 Dataset Summary:
   - Training (Europarl): {len(train_samples):,} samples
   - Validation (Europarl): {len(val_samples):,} samples  
   - Test (Software Domain): {len(test_samples):,} samples

📁 Output Directory: {output_dir}
   - train.jsonl / train.tsv
   - val.jsonl / val.tsv
   - test.jsonl / test.tsv

🎯 Next Steps:
   1. Train encoder-decoder: python scripts/train_encoder_decoder.py
   2. Train decoder-only: python scripts/train_decoder_only.py
   3. Evaluate: python scripts/evaluate.py
""")


if __name__ == "__main__":
    main()
