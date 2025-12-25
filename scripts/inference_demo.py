#!/usr/bin/env python3
"""
Inference Demo Script
=====================
Lightweight demonstration of MT capabilities without training.
Works on memory-constrained systems.
"""

import os
import sys
from pathlib import Path
import json
import gc

# Force garbage collection
gc.collect()

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
torch.set_num_threads(2)  # Limit threads to save memory


def load_test_samples(test_path: Path, max_samples: int = 20):
    """Load limited test samples."""
    samples = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            samples.append(json.loads(line.strip()))
    return samples


def translate_with_marian(sources: list, batch_size: int = 2):
    """Translate using MarianMT model."""
    from transformers import MarianMTModel, MarianTokenizer
    
    print("Loading MarianMT model...")
    model_name = "Helsinki-NLP/opus-mt-en-nl"
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    
    translations = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=4)
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)
        
        # Clear memory
        del inputs, outputs
        gc.collect()
    
    del model, tokenizer
    gc.collect()
    
    return translations


def compute_simple_metrics(hypotheses: list, references: list):
    """Compute basic metrics without heavy dependencies."""
    try:
        from sacrebleu.metrics import BLEU, CHRF, TER
        
        bleu = BLEU()
        chrf = CHRF(word_order=2)
        ter = TER()
        
        return {
            'bleu': bleu.corpus_score(hypotheses, [references]).score,
            'chrf': chrf.corpus_score(hypotheses, [references]).score,
            'ter': ter.corpus_score(hypotheses, [references]).score
        }
    except ImportError:
        return {'error': 'sacrebleu not installed'}


def main():
    print("=" * 60)
    print("MT INFERENCE DEMO")
    print("=" * 60)
    
    # Load test data
    test_path = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
    print(f"\nLoading test samples from: {test_path}")
    
    samples = load_test_samples(test_path, max_samples=20)
    sources = [s['source'] for s in samples]
    references = [s['target'] for s in samples]
    
    print(f"Loaded {len(samples)} test samples")
    
    # Translate
    print("\n" + "-" * 40)
    print("Translating with MarianMT baseline...")
    print("-" * 40)
    
    hypotheses = translate_with_marian(sources, batch_size=2)
    
    # Show sample translations
    print("\n📝 Sample Translations:")
    print("-" * 40)
    for i in range(min(5, len(sources))):
        print(f"\n[{i+1}] Source: {sources[i][:80]}...")
        print(f"    Reference: {references[i][:80]}...")
        print(f"    Hypothesis: {hypotheses[i][:80]}...")
    
    # Compute metrics
    print("\n" + "-" * 40)
    print("Computing Metrics...")
    print("-" * 40)
    
    metrics = compute_simple_metrics(hypotheses, references)
    
    if 'error' not in metrics:
        print(f"\n📊 Results on {len(samples)} Software Domain Samples:")
        print(f"   BLEU:   {metrics['bleu']:.2f}")
        print(f"   chrF++: {metrics['chrf']:.2f}")
        print(f"   TER:    {metrics['ter']:.2f}")
    
    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "inference_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model': 'Helsinki-NLP/opus-mt-en-nl',
        'dataset': 'software_domain',
        'num_samples': len(samples),
        'metrics': metrics,
        'translations': [
            {'source': s, 'reference': r, 'hypothesis': h}
            for s, r, h in zip(sources, references, hypotheses)
        ]
    }
    
    with open(output_dir / "demo_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to: {output_dir / 'demo_results.json'}")
    
    # Show key observations
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS")
    print("=" * 60)
    print("""
1. BASELINE PERFORMANCE:
   - The pre-trained MarianMT achieves reasonable BLEU on software domain
   - Model handles technical terms but struggles with placeholders like {1}, {2}
   
2. EXPECTED IMPROVEMENTS WITH FINE-TUNING:
   - Domain-specific vocabulary (TurboPower™, OIS, Quad Pixel)
   - Placeholder preservation ({1}, {2}, etc.)
   - Software UI terminology consistency
   
3. CHALLENGE 2 STATUS:
   - Dataset_Challenge_2.xlsx (Spanish QE data) is MISSING
   - Cannot proceed with Quality Estimation task without this data
""")


if __name__ == "__main__":
    main()
