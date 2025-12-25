#!/usr/bin/env python3
"""
Quality Estimation Script (Challenge 2)
=======================================
Run quality estimation on translations to approximate linguist feedback.

This addresses Challenge 2: Quality Estimation to approximate linguist post-editing.
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
from typing import List, Dict, Optional

from src.models.quality_estimation import (
    QualityEstimator, 
    LinguistFeedbackApproximator,
    ErrorCategory,
    QualityScore
)


def run_qe_pipeline(
    sources: List[str],
    hypotheses: List[str],
    references: Optional[List[str]] = None,
    model_name: str = "unknown"
) -> Dict:
    """
    Run complete QE pipeline.
    
    Args:
        sources: Source texts
        hypotheses: Model translations
        references: Optional reference translations
        model_name: Name of the model being evaluated
        
    Returns:
        Dictionary with QE results
    """
    print(f"\nRunning QE pipeline for model: {model_name}")
    print(f"  Samples: {len(sources)}")
    
    # Initialize QE components
    qe = QualityEstimator(use_comet_qe=False)  # Start without COMET-QE for speed
    feedback_approximator = LinguistFeedbackApproximator()
    
    # 1. Compute quality scores
    print("\n  [1/3] Computing quality scores...")
    quality_scores = []
    
    for i, (src, hyp) in enumerate(zip(sources, hypotheses)):
        ref = references[i] if references else None
        
        score = qe.estimate_quality(
            source=src,
            hypothesis=hyp,
            reference=ref
        )
        quality_scores.append(score)
        
        if (i + 1) % 20 == 0:
            print(f"    Processed {i + 1}/{len(sources)} samples")
    
    # 2. Categorize errors
    print("\n  [2/3] Categorizing translation errors...")
    error_categories = {}
    for cat in ErrorCategory:
        error_categories[cat.name] = 0
    
    for score in quality_scores:
        for cat in score.error_categories:
            error_categories[cat.name] += 1
    
    # 3. Generate linguist-style feedback
    print("\n  [3/3] Generating linguist-style feedback...")
    feedback_results = []
    
    for i, (src, hyp, score) in enumerate(zip(sources, hypotheses, quality_scores)):
        ref = references[i] if references else None
        
        feedback = feedback_approximator.generate_feedback(
            source=src,
            hypothesis=hyp,
            reference=ref,
            quality_score=score
        )
        
        feedback_results.append({
            'source': src,
            'hypothesis': hyp,
            'reference': ref,
            'score': score.overall_score,
            'feedback': feedback.feedback_text,
            'severity': feedback.severity.name,
            'edit_distance': feedback.edit_distance_ratio,
            'needs_pe': feedback.requires_post_edit,
            'error_types': [e.name for e in score.error_categories]
        })
    
    # Compute aggregate statistics
    overall_scores = [s.overall_score for s in quality_scores]
    
    stats = {
        'model_name': model_name,
        'num_samples': len(sources),
        'mean_score': float(np.mean(overall_scores)),
        'std_score': float(np.std(overall_scores)),
        'min_score': float(np.min(overall_scores)),
        'max_score': float(np.max(overall_scores)),
        'median_score': float(np.median(overall_scores)),
        'requires_post_edit_pct': sum(1 for r in feedback_results if r['needs_pe']) / len(feedback_results) * 100,
        'error_distribution': error_categories,
        'score_distribution': {
            'excellent (>0.8)': sum(1 for s in overall_scores if s > 0.8),
            'good (0.6-0.8)': sum(1 for s in overall_scores if 0.6 <= s <= 0.8),
            'needs_review (0.4-0.6)': sum(1 for s in overall_scores if 0.4 <= s < 0.6),
            'poor (<0.4)': sum(1 for s in overall_scores if s < 0.4)
        }
    }
    
    return {
        'stats': stats,
        'detailed_results': feedback_results,
        'quality_scores': quality_scores
    }


def main(args):
    """Main QE pipeline."""
    print("=" * 60)
    print("QUALITY ESTIMATION PIPELINE")
    print("Challenge 2: Approximate Linguist Feedback")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================
    # 1. Load Translations
    # =========================================
    print("\n[1/3] Loading translations...")
    
    # Check for evaluation results
    eval_dir = Path(args.eval_dir)
    
    if (eval_dir / "baseline_translations.csv").exists():
        baseline_df = pd.read_csv(eval_dir / "baseline_translations.csv")
        print(f"  Loaded baseline translations: {len(baseline_df)} samples")
        
        sources = baseline_df['source'].tolist()
        references = baseline_df['reference'].tolist()
        baseline_hyps = baseline_df['hypothesis'].tolist()
    else:
        # Load from challenge dataset directly
        from src.data.data_loader import MTDataLoader
        
        loader = MTDataLoader(data_dir=Path(args.data_dir))
        samples = loader.load_challenge_dataset(
            PROJECT_ROOT / "Dataset_Challenge_1.xlsx"
        )
        
        sources = [s.source for s in samples]
        references = [s.target for s in samples]
        
        # Generate baseline translations
        print("  Generating baseline translations...")
        from transformers import MarianMTModel, MarianTokenizer
        import torch
        
        model_name = "Helsinki-NLP/opus-mt-en-nl"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        baseline_hyps = []
        batch_size = 8
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=256, num_beams=4)
            translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            baseline_hyps.extend(translations)
        
        print(f"  Generated {len(baseline_hyps)} baseline translations")
    
    # =========================================
    # 2. Run QE Pipeline
    # =========================================
    print("\n[2/3] Running Quality Estimation...")
    
    qe_results = run_qe_pipeline(
        sources=sources,
        hypotheses=baseline_hyps,
        references=references,
        model_name="baseline_opus-mt"
    )
    
    # =========================================
    # 3. Generate Reports
    # =========================================
    print("\n[3/3] Generating reports...")
    
    # Save detailed results
    results_df = pd.DataFrame(qe_results['detailed_results'])
    results_df.to_csv(output_dir / "qe_detailed_results.csv", index=False)
    
    # Save statistics
    with open(output_dir / "qe_statistics.json", 'w') as f:
        json.dump(qe_results['stats'], f, indent=2)
    
    # Print summary report
    stats = qe_results['stats']
    
    print("\n" + "=" * 60)
    print("QUALITY ESTIMATION REPORT")
    print("=" * 60)
    
    print(f"\nModel: {stats['model_name']}")
    print(f"Samples Evaluated: {stats['num_samples']}")
    
    print(f"\nOverall Quality Score:")
    print(f"  Mean:   {stats['mean_score']:.3f}")
    print(f"  Std:    {stats['std_score']:.3f}")
    print(f"  Range:  [{stats['min_score']:.3f}, {stats['max_score']:.3f}]")
    
    print(f"\nScore Distribution:")
    for bucket, count in stats['score_distribution'].items():
        pct = count / stats['num_samples'] * 100
        print(f"  {bucket}: {count} ({pct:.1f}%)")
    
    print(f"\nPost-Editing Required: {stats['requires_post_edit_pct']:.1f}%")
    
    print(f"\nError Type Distribution:")
    for error_type, count in sorted(stats['error_distribution'].items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / stats['num_samples'] * 100
            print(f"  {error_type}: {count} ({pct:.1f}%)")
    
    # Show examples needing post-edit
    print("\n" + "=" * 60)
    print("EXAMPLES REQUIRING POST-EDITING")
    print("=" * 60)
    
    needs_pe = [r for r in qe_results['detailed_results'] if r['needs_pe']][:5]
    
    for i, example in enumerate(needs_pe, 1):
        print(f"\n[{i}] Score: {example['score']:.3f} | Severity: {example['severity']}")
        print(f"  Source:     {example['source']}")
        print(f"  Hypothesis: {example['hypothesis']}")
        print(f"  Reference:  {example['reference']}")
        print(f"  Errors:     {', '.join(example['error_types'])}")
        print(f"  Feedback:   {example['feedback']}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality Estimation Pipeline")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="/root/motrola_assignment/data/processed")
    parser.add_argument("--eval_dir", type=str, default="/root/motrola_assignment/outputs/evaluation")
    parser.add_argument("--output_dir", type=str, default="/root/motrola_assignment/outputs/quality_estimation")
    
    args = parser.parse_args()
    main(args)
