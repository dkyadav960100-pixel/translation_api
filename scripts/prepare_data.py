#!/usr/bin/env python3
"""
Data Preparation Script
=======================
Prepare datasets for MT fine-tuning and evaluation.

Data Sources:
- TRAINING: WMT16 IT-domain translation task (EN-NL)
  - Batch1 and Batch2 (in-domain training data)
  - Localization PO files (VLC, LibreOffice, KDE)
- TESTING: Dataset_Challenge_1.xlsx (Motorola software domain)

Reference: https://www.statmt.org/wmt16/it-translation-task.html
"""

import os
import sys
from pathlib import Path
import json
import argparse
import zipfile
import tarfile
import requests
from typing import List, Tuple
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.data.data_loader import MTDataLoader, TranslationSample
from src.data.preprocessing import TextPreprocessor, PreprocessingConfig
from src.data.feature_store import FeatureExtractor, FeatureStore


# WMT16 IT-domain data URLs
WMT16_DATA_URLS = {
    'batch1_2': 'http://ufallab.ms.mff.cuni.cz/~popel/batch1and2.zip',
    'indomain_training': 'http://ufallab.ms.mff.cuni.cz/~popel/indomain_training.zip',
    'batch3_test': 'http://data.statmt.org/wmt16/it-translation-task/wmt16-it-task-references.tgz'
}


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL."""
    try:
        print(f"  Downloading: {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """Extract zip or tar.gz archive."""
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_dir)
        elif archive_path.suffix in ['.tgz', '.gz']:
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(extract_dir)
        else:
            print(f"  Unknown archive format: {archive_path}")
            return False
        
        print(f"  Extracted to: {extract_dir}")
        return True
    except Exception as e:
        print(f"  Error extracting {archive_path}: {e}")
        return False


def parse_wmt16_batch_file(file_path: Path, lang: str = 'nl') -> List[Tuple[str, str]]:
    """
    Parse WMT16 batch files (Batch1/Batch2).
    Format: Each answer may contain multiple sentences.
    """
    pairs = []
    
    # Look for English and target language files
    en_file = file_path.parent / f"{file_path.stem}_en.txt"
    nl_file = file_path.parent / f"{file_path.stem}_{lang}.txt"
    
    if en_file.exists() and nl_file.exists():
        with open(en_file, 'r', encoding='utf-8') as f:
            en_lines = f.readlines()
        with open(nl_file, 'r', encoding='utf-8') as f:
            nl_lines = f.readlines()
        
        for en, nl in zip(en_lines, nl_lines):
            en = en.strip()
            nl = nl.strip()
            if en and nl:
                pairs.append((en, nl))
    
    return pairs


def parse_po_files(po_dir: Path, lang: str = 'nl') -> List[Tuple[str, str]]:
    """
    Parse localization PO files for parallel data.
    PO format:
    msgid "English text"
    msgstr "Translated text"
    """
    pairs = []
    
    for po_file in po_dir.rglob(f'*.{lang}.po'):
        try:
            with open(po_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Simple PO parser
            msgid_pattern = r'msgid\s+"([^"]*(?:\\.[^"]*)*)"'
            msgstr_pattern = r'msgstr\s+"([^"]*(?:\\.[^"]*)*)"'
            
            msgids = re.findall(msgid_pattern, content)
            msgstrs = re.findall(msgstr_pattern, content)
            
            for msgid, msgstr in zip(msgids, msgstrs):
                # Clean up escape sequences
                msgid = msgid.replace('\\n', '\n').replace('\\"', '"').strip()
                msgstr = msgstr.replace('\\n', '\n').replace('\\"', '"').strip()
                
                if msgid and msgstr and msgid != msgstr:
                    pairs.append((msgid, msgstr))
                    
        except Exception as e:
            print(f"    Error parsing {po_file}: {e}")
            continue
    
    return pairs


def download_wmt16_data(raw_dir: Path) -> dict:
    """Download WMT16 IT-domain training data."""
    print("\n[WMT16] Downloading IT-domain training data...")
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    
    for name, url in WMT16_DATA_URLS.items():
        ext = '.zip' if url.endswith('.zip') else '.tgz'
        archive_path = raw_dir / f"{name}{ext}"
        extract_dir = raw_dir / name
        
        if extract_dir.exists():
            print(f"  {name} already extracted, skipping download")
            downloaded[name] = extract_dir
            continue
        
        if not archive_path.exists():
            if download_file(url, archive_path):
                if extract_archive(archive_path, extract_dir):
                    downloaded[name] = extract_dir
        else:
            if extract_archive(archive_path, extract_dir):
                downloaded[name] = extract_dir
    
    return downloaded


def load_wmt16_training_data(raw_dir: Path, lang: str = 'nl') -> List[TranslationSample]:
    """Load WMT16 IT-domain training data."""
    samples = []
    sample_id = 0
    
    # Try to load from various sources
    batch_dir = raw_dir / 'batch1_2'
    indomain_dir = raw_dir / 'indomain_training'
    
    # Load Batch1/2 data
    if batch_dir.exists():
        print("  Loading Batch1/2 data...")
        for txt_file in batch_dir.rglob('*.txt'):
            if '_en' in txt_file.name:
                pairs = parse_wmt16_batch_file(txt_file, lang)
                for en, tgt in pairs:
                    samples.append(TranslationSample(
                        source=en,
                        target=tgt,
                        source_lang='en',
                        target_lang=lang,
                        domain='software/IT',
                        sample_id=f"wmt16_batch_{sample_id}",
                        metadata={'source': 'wmt16_batch'}
                    ))
                    sample_id += 1
        print(f"    Loaded {sample_id} samples from Batch1/2")
    
    # Load PO files (VLC, LibreOffice, KDE)
    if indomain_dir.exists():
        print("  Loading PO file data...")
        po_start = sample_id
        pairs = parse_po_files(indomain_dir, lang)
        for en, tgt in pairs:
            samples.append(TranslationSample(
                source=en,
                target=tgt,
                source_lang='en',
                target_lang=lang,
                domain='software/IT',
                sample_id=f"wmt16_po_{sample_id}",
                metadata={'source': 'wmt16_po_files'}
            ))
            sample_id += 1
        print(f"    Loaded {sample_id - po_start} samples from PO files")
    
    return samples


def create_synthetic_training_data(test_samples: List[TranslationSample]) -> List[TranslationSample]:
    """
    Create synthetic training data from similar patterns if WMT16 data unavailable.
    This is a fallback for when external data cannot be downloaded.
    """
    synthetic = []
    
    # Common software UI patterns for EN-NL
    patterns = [
        ("Tap to {}", "Tik om {}"),
        ("Press {} to continue", "Druk op {} om door te gaan"),
        ("Settings", "Instellingen"),
        ("Battery", "Batterij"),
        ("Connected to {}", "Verbonden met {}"),
        ("Bluetooth", "Bluetooth"),
        ("Wi-Fi", "Wi-Fi"),
        ("Enable {}", "Schakel {} in"),
        ("Disable {}", "Schakel {} uit"),
        ("Download complete", "Download voltooid"),
        ("Upload failed", "Upload mislukt"),
        ("Error: {}", "Fout: {}"),
        ("Warning: {}", "Waarschuwing: {}"),
        ("Sync in progress", "Synchronisatie bezig"),
        ("Data usage", "Gegevensgebruik"),
        ("Storage", "Opslag"),
        ("Memory", "Geheugen"),
        ("Screen brightness", "Schermhelderheid"),
        ("Volume", "Volume"),
        ("Notifications", "Meldingen"),
        ("Privacy", "Privacy"),
        ("Security", "Beveiliging"),
        ("About phone", "Over de telefoon"),
        ("System update", "Systeemupdate"),
        ("Restart", "Opnieuw opstarten"),
        ("Power off", "Uitschakelen"),
        ("Airplane mode", "Vliegtuigmodus"),
        ("Location services", "Locatieservices"),
        ("App permissions", "App-machtigingen"),
        ("Network settings", "Netwerkinstellingen"),
    ]
    
    for idx, (en, nl) in enumerate(patterns):
        synthetic.append(TranslationSample(
            source=en,
            target=nl,
            source_lang='en',
            target_lang='nl',
            domain='software/IT',
            sample_id=f"synthetic_{idx}",
            metadata={'source': 'synthetic_patterns'}
        ))
    
    return synthetic


def main(args):
    """Main data preparation pipeline."""
    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)
    print("\nData Sources:")
    print("  • TRAINING: WMT16 IT-domain (EN-NL)")
    print("    URL: https://www.statmt.org/wmt16/it-translation-task.html")
    print("  • TESTING: Dataset_Challenge_1.xlsx (Motorola)")
    
    # Initialize components
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    raw_dir = data_dir / "raw" / "wmt16"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    loader = MTDataLoader(data_dir=output_dir)
    preprocessor = TextPreprocessor(PreprocessingConfig(
        normalize_unicode=True,
        preserve_placeholders=True,
        preserve_technical_terms=True
    ))
    extractor = FeatureExtractor()
    store = FeatureStore(output_dir / "features")
    
    # =========================================
    # 1. Load Test Dataset (Motorola Challenge)
    # =========================================
    print("\n[1/5] Loading TEST Dataset (Motorola Challenge)...")
    
    challenge_file = PROJECT_ROOT / args.challenge_file
    if not challenge_file.exists():
        print(f"Error: Challenge file not found: {challenge_file}")
        sys.exit(1)
    
    test_samples = loader.load_challenge_dataset(
        challenge_file,
        source_col="English Source",
        target_col="Reference Translation"
    )
    print(f"  ✓ Loaded {len(test_samples)} samples for TESTING")
    
    # Compute and display statistics
    test_stats = loader.compute_statistics(test_samples)
    print(f"  Statistics:")
    print(f"    - Avg source length: {test_stats.avg_source_length:.1f} words")
    print(f"    - Avg target length: {test_stats.avg_target_length:.1f} words")
    print(f"    - Source vocab size: {test_stats.vocab_size_source}")
    
    # =========================================
    # 2. Download/Load WMT16 Training Data
    # =========================================
    print("\n[2/5] Loading TRAINING Data (WMT16 IT-domain)...")
    
    train_samples = []
    
    if args.download_wmt16:
        downloaded = download_wmt16_data(raw_dir)
        if downloaded:
            train_samples = load_wmt16_training_data(raw_dir, lang='nl')
    else:
        # Try to load existing WMT16 data
        if raw_dir.exists():
            train_samples = load_wmt16_training_data(raw_dir, lang='nl')
    
    # Fallback: create synthetic training data
    if len(train_samples) < 100:
        print("  ⚠ WMT16 data limited, adding synthetic patterns...")
        synthetic = create_synthetic_training_data(test_samples)
        train_samples.extend(synthetic)
        print(f"  Added {len(synthetic)} synthetic training samples")
    
    print(f"  ✓ Total TRAINING samples: {len(train_samples)}")
    
    # =========================================
    # 3. Preprocess All Data
    # =========================================
    print("\n[3/5] Preprocessing data...")
    
    def preprocess_samples(samples: List[TranslationSample]) -> List[TranslationSample]:
        processed = []
        for sample in samples:
            processed_source = preprocessor.preprocess(sample.source, is_source=True)
            processed_target = preprocessor.preprocess(sample.target, is_source=False)
            
            processed.append(TranslationSample(
                source=processed_source,
                target=processed_target,
                source_lang=sample.source_lang,
                target_lang=sample.target_lang,
                domain=sample.domain,
                sample_id=sample.sample_id,
                metadata={
                    **sample.metadata,
                    'original_source': sample.source,
                    'original_target': sample.target
                }
            ))
        return processed
    
    train_processed = preprocess_samples(train_samples)
    test_processed = preprocess_samples(test_samples)
    
    print(f"  ✓ Preprocessed {len(train_processed)} training samples")
    print(f"  ✓ Preprocessed {len(test_processed)} test samples")
    
    # =========================================
    # 4. Extract Features
    # =========================================
    print("\n[4/5] Extracting features...")
    
    # Features for test set (for QE)
    test_features = []
    for sample in test_samples:
        feature = extractor.extract_features(
            sample.source,
            sample.target,
            sample.sample_id
        )
        test_features.append(feature)
    
    store.store_features("test_features", test_features)
    
    # Feature statistics
    feature_stats = store.get_statistics(test_features)
    print(f"  Test Set Feature Statistics:")
    print(f"    - Samples with placeholders: {feature_stats['samples_with_placeholders']}")
    print(f"    - Samples with tech terms: {feature_stats['samples_with_tech_terms']}")
    
    # =========================================
    # 5. Create Final Splits
    # =========================================
    print("\n[5/5] Creating final data splits...")
    
    # Split training data into train/val
    if len(train_samples) > 10:
        train_size = int(len(train_processed) * 0.9)
        train_final = train_processed[:train_size]
        val_final = train_processed[train_size:]
    else:
        train_final = train_processed
        val_final = train_processed[:max(1, len(train_processed)//5)]
    
    # Test set is the full Motorola challenge dataset
    test_final = test_processed
    
    # Save all splits
    loader.save_processed(train_final, "train")
    loader.save_processed(val_final, "val")
    loader.save_processed(test_final, "test")
    
    # Also save the original test set as the "challenge_test" for evaluation
    loader.save_processed(test_samples, "challenge_test")
    
    print(f"\n  Data Splits:")
    print(f"    TRAIN (WMT16):  {len(train_final)} samples")
    print(f"    VAL (WMT16):    {len(val_final)} samples")
    print(f"    TEST (Motorola): {len(test_final)} samples")
    
    # =========================================
    # Generate Summary Report
    # =========================================
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    
    summary = {
        'training_data': {
            'source': 'WMT16 IT-domain + synthetic patterns',
            'url': 'https://www.statmt.org/wmt16/it-translation-task.html',
            'total_samples': len(train_final),
            'val_samples': len(val_final),
            'languages': 'EN → NL',
            'domain': 'software/IT'
        },
        'test_data': {
            'source': 'Dataset_Challenge_1.xlsx (Motorola)',
            'total_samples': len(test_final),
            'languages': 'EN → NL',
            'domain': 'software/IT (Motorola device UI)',
            'statistics': {
                'avg_source_length': test_stats.avg_source_length,
                'avg_target_length': test_stats.avg_target_length,
                'vocab_size_source': test_stats.vocab_size_source,
                'vocab_size_target': test_stats.vocab_size_target,
            }
        },
        'features': {
            'samples_with_placeholders': feature_stats['samples_with_placeholders'],
            'samples_with_tech_terms': feature_stats['samples_with_tech_terms'],
        }
    }
    
    summary_path = output_dir / "data_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📋 Summary saved to: {summary_path}")
    print(f"📁 Processed data saved to: {output_dir}")
    
    print("""
📌 KEY POINTS:
   • Training: WMT16 IT-domain data (batch files + PO localizations)
   • Testing: Motorola Dataset_Challenge_1.xlsx
   • Domain: Software/IT (UI strings, notifications, settings)
   • Languages: English → Dutch (EN-NL)
   
🚀 Next Steps:
   1. Run training: python scripts/train_encoder_decoder.py
   2. Evaluate on test set: python scripts/evaluate.py
   3. Run QE pipeline: python scripts/run_qe.py
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MT datasets")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/motrola_assignment/data",
        help="Directory for data storage"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/motrola_assignment/data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--challenge_file",
        type=str,
        default="Dataset_Challenge_1.xlsx",
        help="Challenge dataset filename (for testing)"
    )
    parser.add_argument(
        "--download_wmt16",
        action="store_true",
        help="Download WMT16 data from source"
    )
    
    args = parser.parse_args()
    main(args)
