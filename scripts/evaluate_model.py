#!/usr/bin/env python3
"""
Comprehensive Evaluation Script
===============================
Evaluate MT models on:
1. FLORES-devtest (general domain)
2. Dataset_Challenge_1.xlsx (software domain)

Metrics: BLEU, chrF++, TER (using sacrebleu)
"""

import os
import sys
from pathlib import Path
import argparse
import json
from typing import List, Dict, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import MarianMTModel, MarianTokenizer

# Try to import sacrebleu
try:
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF, TER
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Warning: sacrebleu not installed. Install with: pip install sacrebleu")

# Try to import datasets for FLORES
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets not installed. Install with: pip install datasets")


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load data from JSONL file."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def load_model(model_path: str, device: str = "cpu"):
    """Load MarianMT model from path."""
    print(f"Loading model from: {model_path}")
    
    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer


def translate_batch(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int = 8,
    max_length: int = 128,
    num_beams: int = 4
) -> List[str]:
    """Translate a list of texts in batches."""
    translations = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)
    
    return translations


def compute_metrics(hypotheses: List[str], references: List[str]) -> Dict:
    """Compute translation metrics using sacrebleu."""
    if not SACREBLEU_AVAILABLE:
        return {"error": "sacrebleu not installed"}
    
    # BLEU
    bleu = BLEU()
    bleu_score = bleu.corpus_score(hypotheses, [references])
    
    # chrF++
    chrf = CHRF(word_order=2)  # chrF++ includes word order
    chrf_score = chrf.corpus_score(hypotheses, [references])
    
    # TER
    ter = TER()
    ter_score = ter.corpus_score(hypotheses, [references])
    
    return {
        'bleu': {
            'score': bleu_score.score,
            'bp': bleu_score.bp,
            'precisions': bleu_score.precisions
        },
        'chrf': {
            'score': chrf_score.score
        },
        'ter': {
            'score': ter_score.score
        }
    }


def load_flores_devtest(source_lang: str = "eng_Latn", target_lang: str = "nld_Latn"):
    """Load FLORES-200 devtest dataset."""
    if not DATASETS_AVAILABLE:
        print("Cannot load FLORES: datasets library not installed")
        return [], []
    
    try:
        # Try loading FLORES-200
        print("Loading FLORES-200 devtest...")
        
        # FLORES-200 uses different language codes
        dataset = load_dataset("facebook/flores", "eng_Latn-nld_Latn", split="devtest")
        
        sources = [item['sentence_eng_Latn'] for item in dataset]
        references = [item['sentence_nld_Latn'] for item in dataset]
        
        print(f"Loaded {len(sources)} FLORES samples")
        return sources, references
        
    except Exception as e:
        print(f"Could not load FLORES-200: {e}")
        
        # Try alternative approach
        try:
            print("Trying alternative FLORES loading method...")
            flores_en = load_dataset("facebook/flores", "eng_Latn", split="devtest")
            flores_nl = load_dataset("facebook/flores", "nld_Latn", split="devtest")
            
            sources = [item['sentence'] for item in flores_en]
            references = [item['sentence'] for item in flores_nl]
            
            print(f"Loaded {len(sources)} FLORES samples (alternative method)")
            return sources, references
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            return [], []


def evaluate_model(
    model_path: str,
    test_data_path: Path,
    output_dir: Path,
    device: str = "cpu",
    evaluate_flores: bool = True
):
    """Run complete evaluation pipeline."""
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_model(model_path, device)
    
    results = {}
    
    # =========================================
    # 1. Evaluate on Software Domain (Challenge Test Set)
    # =========================================
    print("\n[1/2] Evaluating on Software Domain Test Set...")
    
    test_samples = load_jsonl(test_data_path)
    sources = [s['source'] for s in test_samples]
    references = [s['target'] for s in test_samples]
    
    print(f"  Translating {len(sources)} samples...")
    hypotheses = translate_batch(model, tokenizer, sources, device)
    
    print("  Computing metrics...")
    software_metrics = compute_metrics(hypotheses, references)
    results['software_domain'] = software_metrics
    
    print(f"\n  📊 Software Domain Results:")
    print(f"     BLEU:  {software_metrics['bleu']['score']:.2f}")
    print(f"     chrF++: {software_metrics['chrf']['score']:.2f}")
    print(f"     TER:   {software_metrics['ter']['score']:.2f}")
    
    # Save software domain translations
    software_output = []
    for src, ref, hyp in zip(sources, references, hypotheses):
        software_output.append({
            'source': src,
            'reference': ref,
            'hypothesis': hyp
        })
    
    with open(output_dir / "software_translations.json", 'w', encoding='utf-8') as f:
        json.dump(software_output, f, ensure_ascii=False, indent=2)
    
    # =========================================
    # 2. Evaluate on FLORES-devtest (General Domain)
    # =========================================
    if evaluate_flores:
        print("\n[2/2] Evaluating on FLORES-devtest (General Domain)...")
        
        flores_sources, flores_references = load_flores_devtest()
        
        if flores_sources:
            print(f"  Translating {len(flores_sources)} samples...")
            flores_hypotheses = translate_batch(model, tokenizer, flores_sources, device)
            
            print("  Computing metrics...")
            flores_metrics = compute_metrics(flores_hypotheses, flores_references)
            results['flores_devtest'] = flores_metrics
            
            print(f"\n  📊 FLORES-devtest Results:")
            print(f"     BLEU:  {flores_metrics['bleu']['score']:.2f}")
            print(f"     chrF++: {flores_metrics['chrf']['score']:.2f}")
            print(f"     TER:   {flores_metrics['ter']['score']:.2f}")
            
            # Save FLORES translations (sample)
            flores_output = []
            for src, ref, hyp in zip(flores_sources[:50], flores_references[:50], flores_hypotheses[:50]):
                flores_output.append({
                    'source': src,
                    'reference': ref,
                    'hypothesis': hyp
                })
            
            with open(output_dir / "flores_translations_sample.json", 'w', encoding='utf-8') as f:
                json.dump(flores_output, f, ensure_ascii=False, indent=2)
        else:
            print("  ⚠️ Could not load FLORES dataset")
            results['flores_devtest'] = {"error": "Could not load FLORES dataset"}
    
    # =========================================
    # Save Results
    # =========================================
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print("\n| Dataset          | BLEU  | chrF++ | TER   |")
    print("|------------------|-------|--------|-------|")
    
    if 'software_domain' in results and 'error' not in results['software_domain']:
        sw = results['software_domain']
        print(f"| Software Domain  | {sw['bleu']['score']:5.2f} | {sw['chrf']['score']:6.2f} | {sw['ter']['score']:5.2f} |")
    
    if 'flores_devtest' in results and 'error' not in results['flores_devtest']:
        fl = results['flores_devtest']
        print(f"| FLORES-devtest   | {fl['bleu']['score']:5.2f} | {fl['chrf']['score']:6.2f} | {fl['ter']['score']:5.2f} |")
    
    print(f"\n✅ Results saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MT Model")
    
    parser.add_argument("--model_path", type=str, 
                        default="Helsinki-NLP/opus-mt-en-nl",
                        help="Path to model or HuggingFace model name")
    parser.add_argument("--test_data", type=str,
                        default="/root/motrola_assignment/data/processed/test.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="/root/motrola_assignment/outputs/evaluation")
    parser.add_argument("--no_flores", action="store_true",
                        help="Skip FLORES evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluate_model(
        model_path=args.model_path,
        test_data_path=Path(args.test_data),
        output_dir=output_dir,
        device=args.device,
        evaluate_flores=not args.no_flores
    )


if __name__ == "__main__":
    main()
