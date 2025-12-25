"""
Evaluation Metrics Module
=========================
Comprehensive evaluation metrics for machine translation quality.

Metrics implemented:
- BLEU (SacreBLEU implementation)
- chrF++ (character F-score with word n-grams)
- COMET (neural metric)
- TER (Translation Edit Rate)
- spBLEU (SentencePiece BLEU for FLORES)
- BERTScore (semantic similarity)

Supports both reference-based and reference-free evaluation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import json

import numpy as np

# Evaluation libraries
try:
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF, TER
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Warning: sacrebleu not installed")

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

try:
    from bert_score import score as bert_score_fn
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False


@dataclass
class MetricResult:
    """Result from a single metric evaluation."""
    name: str
    score: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Comprehensive evaluation results."""
    dataset_name: str
    model_name: str
    num_samples: int
    
    # Main metrics
    bleu: Optional[MetricResult] = None
    chrf: Optional[MetricResult] = None
    chrf_pp: Optional[MetricResult] = None  # chrF++
    ter: Optional[MetricResult] = None
    comet: Optional[MetricResult] = None
    bert_score: Optional[MetricResult] = None
    
    # Aggregate
    aggregate_score: Optional[float] = None
    
    # Per-sample scores
    sample_scores: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            'dataset_name': self.dataset_name,
            'model_name': self.model_name,
            'num_samples': self.num_samples,
            'aggregate_score': self.aggregate_score
        }
        
        for metric_name in ['bleu', 'chrf', 'chrf_pp', 'ter', 'comet', 'bert_score']:
            metric = getattr(self, metric_name)
            if metric:
                result[metric_name] = asdict(metric)
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class MTEvaluator:
    """
    Comprehensive machine translation evaluator.
    
    Features:
    - Multiple metric support (BLEU, chrF++, COMET, TER, BERTScore)
    - Corpus-level and sentence-level evaluation
    - FLORES-compatible evaluation (spBLEU)
    - Detailed analysis and reporting
    """
    
    def __init__(
        self,
        metrics: List[str] = None,
        comet_model: str = "Unbabel/wmt22-comet-da",
        device: str = None,
        load_neural_metrics: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            metrics: List of metrics to compute ['bleu', 'chrf', 'ter', 'comet', 'bert_score']
            comet_model: COMET model to use
            device: Device for neural metrics
            load_neural_metrics: Whether to load COMET/BERTScore
        """
        self.device = device or ('cuda' if self._check_cuda() else 'cpu')
        
        # Default metrics
        if metrics is None:
            metrics = ['bleu', 'chrf', 'ter']
            if load_neural_metrics and COMET_AVAILABLE:
                metrics.append('comet')
        
        self.metrics = metrics
        
        # Initialize SacreBLEU metrics
        if SACREBLEU_AVAILABLE:
            self.bleu_metric = BLEU()
            self.chrf_metric = CHRF(word_order=2)  # chrF++
            self.ter_metric = TER()
        
        # Initialize COMET
        self.comet_model = None
        if 'comet' in metrics and COMET_AVAILABLE and load_neural_metrics:
            try:
                model_path = download_model(comet_model)
                self.comet_model = load_from_checkpoint(model_path)
            except Exception as e:
                print(f"Could not load COMET: {e}")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def evaluate(
        self,
        hypotheses: List[str],
        references: List[str],
        sources: Optional[List[str]] = None,
        dataset_name: str = "test",
        model_name: str = "model",
        compute_per_sample: bool = True
    ) -> EvaluationResult:
        """
        Evaluate translations against references.
        
        Args:
            hypotheses: Model translations
            references: Reference translations
            sources: Source texts (needed for COMET)
            dataset_name: Name of the evaluation dataset
            model_name: Name of the model being evaluated
            compute_per_sample: Whether to compute per-sample scores
            
        Returns:
            EvaluationResult with all metrics
        """
        assert len(hypotheses) == len(references), "Length mismatch"
        
        result = EvaluationResult(
            dataset_name=dataset_name,
            model_name=model_name,
            num_samples=len(hypotheses)
        )
        
        # BLEU
        if 'bleu' in self.metrics and SACREBLEU_AVAILABLE:
            result.bleu = self._compute_bleu(hypotheses, references)
        
        # chrF++
        if 'chrf' in self.metrics and SACREBLEU_AVAILABLE:
            result.chrf_pp = self._compute_chrf(hypotheses, references)
        
        # TER
        if 'ter' in self.metrics and SACREBLEU_AVAILABLE:
            result.ter = self._compute_ter(hypotheses, references)
        
        # COMET
        if 'comet' in self.metrics and self.comet_model and sources:
            result.comet = self._compute_comet(hypotheses, references, sources)
        
        # BERTScore
        if 'bert_score' in self.metrics and BERT_SCORE_AVAILABLE:
            result.bert_score = self._compute_bert_score(hypotheses, references)
        
        # Compute aggregate score
        result.aggregate_score = self._compute_aggregate(result)
        
        # Per-sample scores
        if compute_per_sample:
            result.sample_scores = self._compute_per_sample(
                hypotheses, references, sources
            )
        
        return result
    
    def _compute_bleu(self, hypotheses: List[str], references: List[str]) -> MetricResult:
        """Compute BLEU score using SacreBLEU."""
        # SacreBLEU expects references as list of lists
        refs = [[ref] for ref in references]
        
        # Corpus-level BLEU
        bleu_result = self.bleu_metric.corpus_score(hypotheses, refs)
        
        return MetricResult(
            name='BLEU',
            score=bleu_result.score,
            details={
                'bp': bleu_result.bp,
                'precisions': bleu_result.precisions,
                'signature': str(bleu_result)
            }
        )
    
    def _compute_chrf(self, hypotheses: List[str], references: List[str]) -> MetricResult:
        """Compute chrF++ score."""
        refs = [[ref] for ref in references]
        
        chrf_result = self.chrf_metric.corpus_score(hypotheses, refs)
        
        return MetricResult(
            name='chrF++',
            score=chrf_result.score,
            details={
                'signature': str(chrf_result)
            }
        )
    
    def _compute_ter(self, hypotheses: List[str], references: List[str]) -> MetricResult:
        """Compute TER score."""
        refs = [[ref] for ref in references]
        
        ter_result = self.ter_metric.corpus_score(hypotheses, refs)
        
        return MetricResult(
            name='TER',
            score=ter_result.score,
            details={
                'signature': str(ter_result)
            }
        )
    
    def _compute_comet(
        self,
        hypotheses: List[str],
        references: List[str],
        sources: List[str]
    ) -> MetricResult:
        """Compute COMET score."""
        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]
        
        gpus = 1 if self.device == 'cuda' else 0
        output = self.comet_model.predict(data, batch_size=8, gpus=gpus)
        
        return MetricResult(
            name='COMET',
            score=output.system_score,
            details={
                'per_sample_scores': output.scores[:10]  # First 10 samples
            }
        )
    
    def _compute_bert_score(
        self,
        hypotheses: List[str],
        references: List[str],
        lang: str = "nl"
    ) -> MetricResult:
        """Compute BERTScore."""
        P, R, F1 = bert_score_fn(
            hypotheses, references,
            lang=lang,
            verbose=False,
            device=self.device
        )
        
        return MetricResult(
            name='BERTScore',
            score=F1.mean().item(),
            precision=P.mean().item(),
            recall=R.mean().item(),
            f1=F1.mean().item()
        )
    
    def _compute_aggregate(self, result: EvaluationResult) -> float:
        """Compute aggregate score from all metrics."""
        scores = []
        weights = []
        
        # BLEU (normalized to 0-1)
        if result.bleu:
            scores.append(result.bleu.score / 100)
            weights.append(1.0)
        
        # chrF++ (normalized to 0-1)
        if result.chrf_pp:
            scores.append(result.chrf_pp.score / 100)
            weights.append(1.0)
        
        # COMET (already 0-1 scale approximately)
        if result.comet:
            scores.append(result.comet.score)
            weights.append(2.0)  # Higher weight for neural metric
        
        # BERTScore (already 0-1)
        if result.bert_score:
            scores.append(result.bert_score.f1 or result.bert_score.score)
            weights.append(1.5)
        
        # TER (lower is better, invert)
        if result.ter:
            ter_score = max(0, 1 - result.ter.score / 100)
            scores.append(ter_score)
            weights.append(0.5)
        
        if scores:
            return np.average(scores, weights=weights)
        return 0.0
    
    def _compute_per_sample(
        self,
        hypotheses: List[str],
        references: List[str],
        sources: Optional[List[str]] = None
    ) -> List[Dict]:
        """Compute per-sample scores."""
        sample_scores = []
        
        for i in range(len(hypotheses)):
            hyp = hypotheses[i]
            ref = references[i]
            src = sources[i] if sources else None
            
            scores = {
                'index': i,
                'source': src,
                'hypothesis': hyp,
                'reference': ref
            }
            
            # Sentence-level BLEU
            if SACREBLEU_AVAILABLE:
                sent_bleu = sacrebleu.sentence_bleu(hyp, [ref])
                scores['bleu'] = sent_bleu.score
                
                sent_chrf = sacrebleu.sentence_chrf(hyp, [ref], word_order=2)
                scores['chrf'] = sent_chrf.score
            
            sample_scores.append(scores)
        
        return sample_scores
    
    def compare_models(
        self,
        model_results: Dict[str, EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_results: Dict mapping model names to EvaluationResult
            
        Returns:
            Comparison summary
        """
        comparison = {
            'models': list(model_results.keys()),
            'metrics': {},
            'best_model': {}
        }
        
        # Collect metrics
        metric_names = ['bleu', 'chrf_pp', 'ter', 'comet', 'bert_score', 'aggregate_score']
        
        for metric in metric_names:
            comparison['metrics'][metric] = {}
            scores = []
            
            for model_name, result in model_results.items():
                if metric == 'aggregate_score':
                    score = result.aggregate_score
                else:
                    metric_result = getattr(result, metric)
                    score = metric_result.score if metric_result else None
                
                comparison['metrics'][metric][model_name] = score
                if score is not None:
                    scores.append((model_name, score))
            
            # Find best model for this metric
            if scores:
                if metric == 'ter':  # Lower is better
                    best = min(scores, key=lambda x: x[1])
                else:
                    best = max(scores, key=lambda x: x[1])
                comparison['best_model'][metric] = best[0]
        
        return comparison


def evaluate_flores_style(
    hypotheses: List[str],
    references: List[str],
    source_lang: str = "eng_Latn",
    target_lang: str = "nld_Latn"
) -> Dict[str, float]:
    """
    Evaluate in FLORES benchmark style using spBLEU.
    
    Uses the same tokenization and metric computation as FLORES.
    
    Args:
        hypotheses: Model translations
        references: Reference translations
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        Dictionary with spBLEU and chrF++ scores
    """
    results = {}
    
    if not SACREBLEU_AVAILABLE:
        return results
    
    # spBLEU (uses SentencePiece tokenization internally via flores_200)
    refs = [[ref] for ref in references]
    
    # Standard BLEU with FLORES tokenization
    bleu = BLEU(tokenize='flores200')
    bleu_result = bleu.corpus_score(hypotheses, refs)
    results['spBLEU'] = bleu_result.score
    
    # chrF++
    chrf = CHRF(word_order=2)
    chrf_result = chrf.corpus_score(hypotheses, refs)
    results['chrF++'] = chrf_result.score
    
    return results


def print_evaluation_report(result: EvaluationResult):
    """Print a formatted evaluation report."""
    print("=" * 60)
    print(f"EVALUATION REPORT: {result.model_name}")
    print(f"Dataset: {result.dataset_name}")
    print(f"Samples: {result.num_samples}")
    print("=" * 60)
    
    print("\nMETRICS:")
    print("-" * 40)
    
    if result.bleu:
        print(f"  BLEU:      {result.bleu.score:.2f}")
    
    if result.chrf_pp:
        print(f"  chrF++:    {result.chrf_pp.score:.2f}")
    
    if result.ter:
        print(f"  TER:       {result.ter.score:.2f}")
    
    if result.comet:
        print(f"  COMET:     {result.comet.score:.4f}")
    
    if result.bert_score:
        print(f"  BERTScore: {result.bert_score.f1:.4f} (F1)")
    
    print("-" * 40)
    print(f"  AGGREGATE: {result.aggregate_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    # Test evaluation
    print("Testing MT Evaluator...")
    
    evaluator = MTEvaluator(
        metrics=['bleu', 'chrf', 'ter'],
        load_neural_metrics=False  # Skip for testing
    )
    
    # Test data
    hypotheses = [
        "Breng open apps over van je telefoon naar je pc",
        "{1}Updateperiode: van {2} tot {3}",
        "Of sluit af en ontvang een herinnering."
    ]
    
    references = [
        "Breng open apps over van je telefoon naar je pc of tablet met een snelle veegbeweging",
        "{1}Updateperiode: van {2} tot {3}",
        "Of sluit af en ontvangen een herinnering."
    ]
    
    sources = [
        "Transfer open apps from your phone to your PC or tablet using a fling gesture",
        "{1}Update window: From {2} to {3}",
        "Or leave now and get a reminder."
    ]
    
    result = evaluator.evaluate(
        hypotheses, references, sources,
        dataset_name="test_set",
        model_name="test_model"
    )
    
    print_evaluation_report(result)
