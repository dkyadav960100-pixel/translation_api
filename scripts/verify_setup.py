#!/usr/bin/env python3
"""
Setup Verification Script
=========================
Verifies the project setup and data pipeline configuration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def verify_project_structure():
    """Verify all required files and directories exist."""
    base_path = Path(__file__).parent.parent
    
    required_files = [
        "Dataset_Challenge_1.xlsx",
        "requirements.txt",
        "TECHNICAL_REPORT.md",
        "src/data/data_loader.py",
        "src/data/preprocessing.py",
        "src/data/feature_store.py",
        "src/models/encoder_decoder.py",
        "src/models/decoder_only.py",
        "src/models/quality_estimation.py",
        "src/evaluation/metrics.py",
        "src/utils/edge_cases.py",
        "scripts/prepare_data.py",
        "scripts/train_encoder_decoder.py",
        "scripts/train_decoder_only.py",
        "scripts/evaluate.py",
        "scripts/run_qe.py",
        "config/training_config.yaml",
        "notebooks/01_EDA_and_Modeling_Analysis.ipynb"
    ]
    
    print("=" * 60)
    print("PROJECT STRUCTURE VERIFICATION")
    print("=" * 60)
    
    all_present = True
    for file_path in required_files:
        full_path = base_path / file_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {file_path}")
        if not exists:
            all_present = False
    
    return all_present


def verify_data_sources():
    """Verify data source configuration."""
    print("\n" + "=" * 60)
    print("DATA SOURCE CONFIGURATION")
    print("=" * 60)
    
    base_path = Path(__file__).parent.parent
    
    # Check Motorola test data
    test_data = base_path / "Dataset_Challenge_1.xlsx"
    if test_data.exists():
        import pandas as pd
        df = pd.read_excel(test_data)
        print(f"✅ Test Data: Dataset_Challenge_1.xlsx")
        print(f"   - Samples: {len(df)}")
        print(f"   - Columns: {df.columns.tolist()}")
        print(f"   - Purpose: TESTING ONLY (Motorola software UI)")
    else:
        print(f"❌ Test data not found: Dataset_Challenge_1.xlsx")
    
    # Check WMT16 training data
    wmt16_dir = base_path / "data" / "raw" / "wmt16"
    processed_train = base_path / "data" / "processed" / "train.json"
    
    print(f"\n📊 Training Data: WMT16 IT Domain")
    print(f"   - Source: https://www.statmt.org/wmt16/it-translation-task.html")
    print(f"   - Language Pair: EN-NL (English to Dutch)")
    print(f"   - Domain: IT/Software (VLC, LibreOffice, KDE localizations)")
    
    if wmt16_dir.exists() or processed_train.exists():
        print(f"   - Status: ✅ Downloaded")
        if processed_train.exists():
            import json
            with open(processed_train) as f:
                train_data = json.load(f)
            print(f"   - Training samples: {len(train_data)}")
    else:
        print(f"   - Status: ⚠️  Not downloaded yet")
        print(f"   - Run: python scripts/prepare_data.py --download_wmt16")


def verify_imports():
    """Verify required imports work."""
    print("\n" + "=" * 60)
    print("IMPORT VERIFICATION")
    print("=" * 60)
    
    imports_to_test = [
        ("pandas", "Data loading"),
        ("numpy", "Numerical operations"),
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
    ]
    
    for module, purpose in imports_to_test:
        try:
            __import__(module)
            print(f"✅ {module} - {purpose}")
        except ImportError:
            print(f"❌ {module} - {purpose} (pip install {module})")


def verify_data_loader():
    """Verify data loader functionality."""
    print("\n" + "=" * 60)
    print("DATA LOADER VERIFICATION")
    print("=" * 60)
    
    try:
        from data.data_loader import MTDataLoader, TranslationSample
        
        base_path = Path(__file__).parent.parent
        loader = MTDataLoader(data_dir=base_path / "data")
        
        # Load test data (Motorola)
        test_samples = loader.load_challenge_dataset(
            base_path / "Dataset_Challenge_1.xlsx"
        )
        
        print(f"✅ MTDataLoader initialized successfully")
        print(f"✅ Loaded {len(test_samples)} Motorola test samples")
        
        # Show sample
        if test_samples:
            sample = test_samples[0]
            print(f"\n   Sample test data point:")
            print(f"   - Source: {sample.source[:50]}...")
            print(f"   - Target: {sample.target[:50]}...")
            print(f"   - Domain: {sample.domain}")
        
        # Check WMT16 loader method exists
        if hasattr(loader, 'load_wmt16_it_domain'):
            print(f"✅ WMT16 IT domain loader available")
        else:
            print(f"⚠️  WMT16 IT domain loader not found")
            
    except Exception as e:
        print(f"❌ Data loader error: {e}")


def main():
    """Run all verifications."""
    print("\n" + "🔍 MOTOROLA MT ASSIGNMENT - SETUP VERIFICATION" + "\n")
    
    structure_ok = verify_project_structure()
    verify_data_sources()
    verify_imports()
    verify_data_loader()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("""
📋 Data Pipeline Architecture:
   ┌─────────────────────────────────────────────────────┐
   │  TRAINING DATA: WMT16 IT Domain                    │
   │  - Source: statmt.org/wmt16/it-translation-task    │
   │  - EN-NL software localizations                    │
   │  - ~10,000+ parallel sentences                     │
   └─────────────────────────────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────────────┐
   │  MODEL TRAINING                                     │
   │  - Encoder-Decoder: MarianMT/mBART                 │
   │  - Decoder-Only: LLaMA/Mistral + LoRA              │
   └─────────────────────────────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────────────┐
   │  TEST DATA: Dataset_Challenge_1.xlsx               │
   │  - Motorola software UI strings                    │
   │  - 84 EN-NL translation pairs                      │
   │  - Domain: Mobile device UI                        │
   └─────────────────────────────────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────────────┐
   │  EVALUATION                                         │
   │  - Metrics: BLEU, chrF++, COMET, TER              │
   │  - Quality Estimation (Challenge 2)                │
   └─────────────────────────────────────────────────────┘
""")
    
    print("\n✅ Setup verification complete!")
    print("\nNext steps:")
    print("  1. Run: python scripts/prepare_data.py --download_wmt16")
    print("  2. Run: python scripts/train_encoder_decoder.py")
    print("  3. Run: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
