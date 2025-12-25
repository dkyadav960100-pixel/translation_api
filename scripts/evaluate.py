#!/usr/bin/env python3
"""
Evaluation Script
=================
Comprehensive evaluation of MT models on FLORES-devtest and software domain test set.

Metrics:
- BLEU (SacreBLEU)
- chrF++
- TER
- COMET (if available)
- BERTScore (if available)
"""

import os
import sys
from pathlib import Path
import argparse
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.data.data_loader import MTDataLoader
from src.evaluation.metrics import MTEvaluator, print_evaluation_report, EvaluationResult


def evaluate_model(
    model_type: str,
    model_path: str,
    test_sources: list,
    test_targets: list,
    evaluator: MTEvaluator,
    dataset_name: str = "test"
) -> EvaluationResult:
    """
    Load and evaluate a trained model.
    
    Args:
        model_type: 'encoder_decoder' or 'decoder_only'
        model_path: Path to saved model
        test_sources: List of source texts
        test_targets: List of reference translations
        evaluator: MTEvaluator instance
        dataset_name: Name for this evaluation
        
    Returns:
        EvaluationResult
    """
    print(f"\n  Loading model from: {model_path}")
    
    if model_type == 'encoder_decoder':
        from src.models.encoder_decoder import EncoderDecoderMT
        model = EncoderDecoderMT.load_from_checkpoint(model_path)
        model.eval()
        
        print("  Generating translations...")
        hypotheses = model.generate(test_sources)
        
    elif model_type == 'decoder_only':
        from src.models.decoder_only import DecoderOnlyMT
        model = DecoderOnlyMT.load_from_checkpoint(model_path)
        model.eval()
        
        print("  Generating translations...")
        hypotheses = model.generate(test_sources, max_new_tokens=256)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate
    print("  Computing metrics...")
    result = evaluator.evaluate(
        hypotheses=hypotheses,
        references=test_targets,
        sources=test_sources,
        dataset_name=dataset_name,
        model_name=model_type
    )
    
    return result


def evaluate_baseline(
    test_sources: list,
    test_targets: list,
    evaluator: MTEvaluator,
    model_name: str = "Helsinki-NLP/opus-mt-en-nl"
) -> EvaluationResult:
    """
    Evaluate pretrained baseline model without fine-tuning.
    """
    print(f"\n  Loading baseline model: {model_name}")
    
    from transformers import MarianMTModel, MarianTokenizer
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Generate translations
    print("  Generating baseline translations...")
    hypotheses = []
    
    batch_size = 8
    for i in range(0, len(test_sources), batch_size):
        batch = test_sources[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        
        import torch
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256, num_beams=4)
        
        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        hypotheses.extend(translations)
    
    # Evaluate
    print("  Computing metrics...")
    result = evaluator.evaluate(
        hypotheses=hypotheses,
        references=test_targets,
        sources=test_sources,
        dataset_name="software_domain",
        model_name="baseline_opus-mt"
    )
    
    return result, hypotheses


def main(args):
    """Main evaluation pipeline."""
    print("=" * 60)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = MTEvaluator(
        metrics=['bleu', 'chrf', 'ter'],
        load_neural_metrics=args.use_neural_metrics
    )
    
    # =========================================
    # 1. Load Test Data
    # =========================================
    print("\n[1/4] Loading test data...")
    
    loader = MTDataLoader(data_dir=Path(args.data_dir))
    
    # Load challenge dataset for software domain evaluation
    challenge_samples = loader.load_challenge_dataset(
        PROJECT_ROOT / "Dataset_Challenge_1.xlsx"
    )
    
    test_sources = [s.source for s in challenge_samples]
    test_targets = [s.target for s in challenge_samples]
    
    print(f"  Loaded {len(challenge_samples)} test samples from software domain")
    
    results = {}
    
    # =========================================
    # 2. Evaluate Baseline
    # =========================================
    print("\n[2/4] Evaluating baseline model...")
    
    baseline_result, baseline_hypotheses = evaluate_baseline(
        test_sources, test_targets, evaluator
    )
    results['baseline'] = baseline_result
    
    print_evaluation_report(baseline_result)
    
    # Save baseline translations
    baseline_df = pd.DataFrame({
        'source': test_sources,
        'reference': test_targets,
        'hypothesis': baseline_hypotheses
    })
    baseline_df.to_csv(output_dir / "baseline_translations.csv", index=False)
    
    # =========================================
    # 3. Evaluate Fine-tuned Models (if available)
    # =========================================
    print("\n[3/4] Evaluating fine-tuned models...")
    
    # Check for encoder-decoder model
    enc_dec_path = Path(args.encoder_decoder_path)
    if enc_dec_path.exists():
        try:
            enc_dec_result = evaluate_model(
                model_type='encoder_decoder',
                model_path=str(enc_dec_path),
                test_sources=test_sources,
                test_targets=test_targets,
                evaluator=evaluator,
                dataset_name="software_domain"
            )
            results['encoder_decoder'] = enc_dec_result
            print_evaluation_report(enc_dec_result)
        except Exception as e:
            print(f"  Could not evaluate encoder-decoder model: {e}")
    else:
        print(f"  Encoder-decoder model not found at: {enc_dec_path}")
    
    # Check for decoder-only model
    dec_only_path = Path(args.decoder_only_path)
    if dec_only_path.exists():
        try:
            dec_only_result = evaluate_model(
                model_type='decoder_only',
                model_path=str(dec_only_path),
                test_sources=test_sources,
                test_targets=test_targets,
                evaluator=evaluator,
                dataset_name="software_domain"
            )
            results['decoder_only'] = dec_only_result
            print_evaluation_report(dec_only_result)
        except Exception as e:
            print(f"  Could not evaluate decoder-only model: {e}")
    else:
        print(f"  Decoder-only model not found at: {dec_only_path}")
    
    # =========================================
    # 4. Generate Comparison Report
    # =========================================
    print("\n[4/4] Generating comparison report...")
    
    # Create comparison table
    comparison_data = []
    for model_name, result in results.items():
        row = {
            'Model': model_name,
            'BLEU': result.bleu.score if result.bleu else None,
            'chrF++': result.chrf_pp.score if result.chrf_pp else None,
            'TER': result.ter.score if result.ter else None,
            'Aggregate': result.aggregate_score
        }
        if result.comet:
            row['COMET'] = result.comet.score
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    
    # Save results
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Save detailed results
    detailed_results = {
        model_name: result.to_dict()
        for model_name, result in results.items()
    }
    
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    # =========================================
    # Sample Outputs
    # =========================================
    print("\n" + "=" * 60)
    print("SAMPLE TRANSLATIONS")
    print("=" * 60)
    
    # Show first 5 examples
    for i in range(min(5, len(test_sources))):
        print(f"\n[{i+1}]")
        print(f"  Source: {test_sources[i]}")
        print(f"  Reference: {test_targets[i]}")
        print(f"  Baseline: {baseline_hypotheses[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MT Models")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="/root/motrola_assignment/data/processed")
    parser.add_argument("--output_dir", type=str, default="/root/motrola_assignment/outputs/evaluation")
    
    # Model paths
    parser.add_argument(
        "--encoder_decoder_path", type=str,
        default="/root/motrola_assignment/outputs/encoder_decoder/checkpoints/last.ckpt"
    )
    parser.add_argument(
        "--decoder_only_path", type=str,
        default="/root/motrola_assignment/outputs/decoder_only/checkpoints/last.ckpt"
    )
    
    # Evaluation options
    parser.add_argument("--use_neural_metrics", action="store_true", help="Use COMET/BERTScore")
    
    args = parser.parse_args()
    main(args)
