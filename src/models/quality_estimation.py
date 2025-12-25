"""
Quality Estimation Module
=========================
Reference-free and reference-based quality estimation for machine translation.

Models supported:
- COMET-QE (Unbabel/wmt22-cometkiwi-da)
- COMET (reference-based)
- Custom regression models
- BERTScore-based estimation

Challenge 2: Approximate linguist feedback using QE techniques.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Try importing COMET
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Warning: COMET not installed. Some QE features unavailable.")

# Try importing bert_score
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False


@dataclass
class QEResult:
    """Quality estimation result for a single translation."""
    source: str
    translation: str
    reference: Optional[str] = None
    
    # Scores
    comet_qe_score: Optional[float] = None
    comet_score: Optional[float] = None
    bert_score_precision: Optional[float] = None
    bert_score_recall: Optional[float] = None
    bert_score_f1: Optional[float] = None
    
    # Derived metrics
    quality_class: Optional[str] = None  # 'good', 'acceptable', 'poor'
    estimated_edit_distance: Optional[float] = None
    error_types: Optional[List[str]] = None
    
    # Confidence
    confidence: Optional[float] = None


class QualityEstimator:
    """
    Comprehensive quality estimation system for machine translation.
    
    Combines multiple QE approaches:
    1. COMET-QE (reference-free neural QE)
    2. BERTScore (semantic similarity)
    3. Surface-level heuristics
    4. Custom regression model
    """
    
    def __init__(
        self,
        comet_qe_model: str = "Unbabel/wmt22-cometkiwi-da",
        comet_model: str = "Unbabel/wmt22-comet-da",
        device: str = None,
        load_comet: bool = True
    ):
        """
        Initialize quality estimator.
        
        Args:
            comet_qe_model: COMET-QE model for reference-free estimation
            comet_model: COMET model for reference-based estimation
            device: Device to use (cuda/cpu)
            load_comet: Whether to load COMET models
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.comet_qe = None
        self.comet = None
        
        if load_comet and COMET_AVAILABLE:
            try:
                # Load COMET-QE (reference-free)
                print(f"Loading COMET-QE model: {comet_qe_model}")
                model_path = download_model(comet_qe_model)
                self.comet_qe = load_from_checkpoint(model_path)
                print("COMET-QE loaded successfully")
            except Exception as e:
                print(f"Could not load COMET-QE: {e}")
            
            try:
                # Load COMET (reference-based)
                print(f"Loading COMET model: {comet_model}")
                model_path = download_model(comet_model)
                self.comet = load_from_checkpoint(model_path)
                print("COMET loaded successfully")
            except Exception as e:
                print(f"Could not load COMET: {e}")
    
    def estimate_quality(
        self,
        sources: List[str],
        translations: List[str],
        references: Optional[List[str]] = None,
        compute_bert_score: bool = True
    ) -> List[QEResult]:
        """
        Estimate quality for a batch of translations.
        
        Args:
            sources: Source texts
            translations: Machine translations
            references: Reference translations (optional)
            compute_bert_score: Whether to compute BERTScore
            
        Returns:
            List of QEResult objects
        """
        results = []
        
        # COMET-QE scores (reference-free)
        comet_qe_scores = None
        if self.comet_qe is not None:
            comet_qe_scores = self._compute_comet_qe(sources, translations)
        
        # COMET scores (reference-based)
        comet_scores = None
        if self.comet is not None and references is not None:
            comet_scores = self._compute_comet(sources, translations, references)
        
        # BERTScore (if references available)
        bert_scores = None
        if compute_bert_score and BERT_SCORE_AVAILABLE and references is not None:
            bert_scores = self._compute_bert_score(translations, references)
        
        # Combine results
        for i in range(len(sources)):
            result = QEResult(
                source=sources[i],
                translation=translations[i],
                reference=references[i] if references else None
            )
            
            # Add scores
            if comet_qe_scores is not None:
                result.comet_qe_score = comet_qe_scores[i]
            
            if comet_scores is not None:
                result.comet_score = comet_scores[i]
            
            if bert_scores is not None:
                result.bert_score_precision = bert_scores['precision'][i]
                result.bert_score_recall = bert_scores['recall'][i]
                result.bert_score_f1 = bert_scores['f1'][i]
            
            # Compute derived metrics
            result.quality_class = self._classify_quality(result)
            result.error_types = self._detect_error_types(result)
            result.confidence = self._compute_confidence(result)
            
            results.append(result)
        
        return results
    
    def _compute_comet_qe(self, sources: List[str], translations: List[str]) -> List[float]:
        """Compute COMET-QE scores (reference-free)."""
        data = [
            {"src": src, "mt": mt}
            for src, mt in zip(sources, translations)
        ]
        
        output = self.comet_qe.predict(data, batch_size=8, gpus=1 if self.device == 'cuda' else 0)
        return output.scores
    
    def _compute_comet(
        self,
        sources: List[str],
        translations: List[str],
        references: List[str]
    ) -> List[float]:
        """Compute COMET scores (reference-based)."""
        data = [
            {"src": src, "mt": mt, "ref": ref}
            for src, mt, ref in zip(sources, translations, references)
        ]
        
        output = self.comet.predict(data, batch_size=8, gpus=1 if self.device == 'cuda' else 0)
        return output.scores
    
    def _compute_bert_score(
        self,
        translations: List[str],
        references: List[str]
    ) -> Dict[str, List[float]]:
        """Compute BERTScore."""
        P, R, F1 = bert_score(translations, references, lang="nl", verbose=False)
        return {
            'precision': P.tolist(),
            'recall': R.tolist(),
            'f1': F1.tolist()
        }
    
    def _classify_quality(self, result: QEResult) -> str:
        """
        Classify translation quality into categories.
        
        Categories:
        - 'good': High quality, minimal or no editing needed
        - 'acceptable': Usable but could be improved
        - 'poor': Requires significant editing
        """
        # Use COMET-QE if available
        if result.comet_qe_score is not None:
            if result.comet_qe_score >= 0.8:
                return 'good'
            elif result.comet_qe_score >= 0.5:
                return 'acceptable'
            else:
                return 'poor'
        
        # Fallback to COMET
        if result.comet_score is not None:
            if result.comet_score >= 0.85:
                return 'good'
            elif result.comet_score >= 0.7:
                return 'acceptable'
            else:
                return 'poor'
        
        # Fallback to BERTScore
        if result.bert_score_f1 is not None:
            if result.bert_score_f1 >= 0.9:
                return 'good'
            elif result.bert_score_f1 >= 0.75:
                return 'acceptable'
            else:
                return 'poor'
        
        return 'unknown'
    
    def _detect_error_types(self, result: QEResult) -> List[str]:
        """
        Detect potential error types in translation.
        
        Error types:
        - length_mismatch: Significant length difference
        - missing_placeholder: Placeholder not preserved
        - untranslated: Contains untranslated source words
        - formatting: Formatting issues
        """
        errors = []
        
        source = result.source
        translation = result.translation
        reference = result.reference
        
        # Length mismatch
        source_words = len(source.split())
        trans_words = len(translation.split())
        ratio = trans_words / source_words if source_words > 0 else 1
        
        if ratio < 0.5 or ratio > 2.0:
            errors.append('length_mismatch')
        
        # Missing placeholders
        import re
        source_placeholders = set(re.findall(r'\{[^}]+\}', source))
        trans_placeholders = set(re.findall(r'\{[^}]+\}', translation))
        
        if source_placeholders and source_placeholders != trans_placeholders:
            errors.append('missing_placeholder')
        
        # Check for untranslated common English words
        common_english = {'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                         'have', 'has', 'had', 'do', 'does', 'did'}
        trans_lower = translation.lower().split()
        untranslated = [w for w in trans_lower if w in common_english]
        if len(untranslated) >= 2:
            errors.append('untranslated')
        
        # Compare with reference if available
        if reference:
            ref_words = set(reference.lower().split())
            trans_words_set = set(translation.lower().split())
            overlap = len(ref_words & trans_words_set) / len(ref_words) if ref_words else 0
            
            if overlap < 0.3:
                errors.append('semantic_divergence')
        
        return errors
    
    def _compute_confidence(self, result: QEResult) -> float:
        """Compute confidence in the quality estimation."""
        confidences = []
        
        # Higher COMET scores = more confident
        if result.comet_qe_score is not None:
            confidences.append(result.comet_qe_score)
        
        if result.comet_score is not None:
            confidences.append(result.comet_score)
        
        if result.bert_score_f1 is not None:
            confidences.append(result.bert_score_f1)
        
        if confidences:
            return np.mean(confidences)
        
        return 0.5  # Default uncertain


class LinguistFeedbackApproximator:
    """
    Approximates linguist feedback based on QE scores and heuristics.
    
    Maps QE outputs to actionable feedback similar to what human
    translators would provide during post-editing.
    """
    
    ERROR_CATEGORIES = {
        'fluency': {
            'description': 'The translation reads naturally in the target language',
            'severity_levels': ['minor', 'major', 'critical']
        },
        'accuracy': {
            'description': 'The translation accurately conveys the source meaning',
            'severity_levels': ['minor', 'major', 'critical']
        },
        'terminology': {
            'description': 'Domain-specific terms are correctly translated',
            'severity_levels': ['minor', 'major']
        },
        'style': {
            'description': 'The translation follows appropriate style guidelines',
            'severity_levels': ['minor', 'major']
        },
        'formatting': {
            'description': 'Placeholders, tags, and formatting are preserved',
            'severity_levels': ['minor', 'critical']
        }
    }
    
    def __init__(self, qe: QualityEstimator = None):
        """
        Initialize feedback approximator.
        
        Args:
            qe: QualityEstimator instance
        """
        self.qe = qe or QualityEstimator(load_comet=False)
    
    def generate_feedback(
        self,
        source: str,
        mt_output: str,
        post_edit: Optional[str] = None,
        reference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate linguist-style feedback for a translation.
        
        Args:
            source: Source text
            mt_output: Machine translation output
            post_edit: Human post-edited version (if available)
            reference: Reference translation (if available)
            
        Returns:
            Dictionary with feedback categories, scores, and suggestions
        """
        feedback = {
            'overall_quality': None,
            'edit_effort': None,  # Estimated editing effort (1-5)
            'categories': {},
            'suggestions': [],
            'error_spans': []
        }
        
        # Get QE result
        qe_results = self.qe.estimate_quality(
            [source], [mt_output], 
            [reference] if reference else None,
            compute_bert_score=reference is not None
        )
        qe_result = qe_results[0]
        
        # Overall quality
        feedback['overall_quality'] = qe_result.quality_class
        
        # Edit effort estimation
        feedback['edit_effort'] = self._estimate_edit_effort(qe_result, post_edit)
        
        # Category-wise feedback
        feedback['categories'] = self._categorize_errors(
            source, mt_output, post_edit, reference, qe_result
        )
        
        # Generate suggestions
        feedback['suggestions'] = self._generate_suggestions(
            source, mt_output, qe_result, feedback['categories']
        )
        
        return feedback
    
    def _estimate_edit_effort(
        self,
        qe_result: QEResult,
        post_edit: Optional[str] = None
    ) -> int:
        """
        Estimate editing effort on scale 1-5.
        
        1: No editing needed
        2: Minor corrections
        3: Moderate editing
        4: Significant editing
        5: Needs retranslation
        """
        # If we have post-edit, compute actual edit distance
        if post_edit:
            mt_words = qe_result.translation.split()
            pe_words = post_edit.split()
            
            # Simple edit distance ratio
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, mt_words, pe_words).ratio()
            
            if ratio >= 0.95:
                return 1
            elif ratio >= 0.85:
                return 2
            elif ratio >= 0.70:
                return 3
            elif ratio >= 0.50:
                return 4
            else:
                return 5
        
        # Otherwise estimate from QE scores
        if qe_result.quality_class == 'good':
            return 1
        elif qe_result.quality_class == 'acceptable':
            return 3
        else:
            return 4
    
    def _categorize_errors(
        self,
        source: str,
        mt_output: str,
        post_edit: Optional[str],
        reference: Optional[str],
        qe_result: QEResult
    ) -> Dict[str, Dict]:
        """Categorize errors by type."""
        categories = {}
        
        # Fluency
        fluency_score = self._assess_fluency(mt_output)
        categories['fluency'] = {
            'score': fluency_score,
            'severity': 'minor' if fluency_score > 0.8 else 'major' if fluency_score > 0.5 else 'critical',
            'issues': []
        }
        
        # Accuracy
        accuracy_score = qe_result.comet_qe_score or qe_result.comet_score or 0.7
        categories['accuracy'] = {
            'score': accuracy_score,
            'severity': 'minor' if accuracy_score > 0.8 else 'major' if accuracy_score > 0.6 else 'critical',
            'issues': qe_result.error_types or []
        }
        
        # Terminology
        term_score, term_issues = self._assess_terminology(source, mt_output)
        categories['terminology'] = {
            'score': term_score,
            'severity': 'minor' if term_score > 0.9 else 'major',
            'issues': term_issues
        }
        
        # Formatting
        format_score, format_issues = self._assess_formatting(source, mt_output)
        categories['formatting'] = {
            'score': format_score,
            'severity': 'minor' if format_score > 0.9 else 'critical',
            'issues': format_issues
        }
        
        return categories
    
    def _assess_fluency(self, text: str) -> float:
        """Assess fluency of translation (heuristic)."""
        score = 1.0
        
        # Check for repeated words
        words = text.lower().split()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                score -= 0.1
        
        # Check for incomplete sentences
        if not text.strip().endswith(('.', '!', '?', '"', "'")):
            score -= 0.1
        
        # Check for reasonable length
        if len(text) < 5:
            score -= 0.3
        
        return max(0, min(1, score))
    
    def _assess_terminology(self, source: str, translation: str) -> Tuple[float, List[str]]:
        """Assess terminology accuracy."""
        issues = []
        score = 1.0
        
        # Software/IT terms that should be preserved or consistently translated
        preserve_terms = {
            'USB', 'USB-C', 'HDMI', 'Bluetooth', 'Wi-Fi', 'WiFi', 'LPDDR5', 'UFS'
        }
        
        for term in preserve_terms:
            if term in source and term not in translation:
                issues.append(f"Technical term '{term}' may need review")
                score -= 0.1
        
        return max(0, min(1, score)), issues
    
    def _assess_formatting(self, source: str, translation: str) -> Tuple[float, List[str]]:
        """Assess formatting preservation."""
        import re
        issues = []
        score = 1.0
        
        # Check placeholders
        source_placeholders = re.findall(r'\{[^}]+\}', source)
        trans_placeholders = re.findall(r'\{[^}]+\}', translation)
        
        missing = set(source_placeholders) - set(trans_placeholders)
        extra = set(trans_placeholders) - set(source_placeholders)
        
        if missing:
            issues.append(f"Missing placeholders: {missing}")
            score -= 0.3 * len(missing)
        
        if extra:
            issues.append(f"Extra placeholders: {extra}")
            score -= 0.2 * len(extra)
        
        return max(0, min(1, score)), issues
    
    def _generate_suggestions(
        self,
        source: str,
        mt_output: str,
        qe_result: QEResult,
        categories: Dict
    ) -> List[str]:
        """Generate actionable suggestions for improvement."""
        suggestions = []
        
        # Based on quality class
        if qe_result.quality_class == 'poor':
            suggestions.append("Consider retranslating this segment")
        
        # Based on error types
        for error in (qe_result.error_types or []):
            if error == 'length_mismatch':
                suggestions.append("Review for completeness - length differs significantly from source")
            elif error == 'missing_placeholder':
                suggestions.append("Ensure all placeholders ({1}, {2}, etc.) are preserved")
            elif error == 'untranslated':
                suggestions.append("Check for untranslated words")
        
        # Based on categories
        for cat_name, cat_data in categories.items():
            if cat_data['severity'] in ['major', 'critical']:
                if cat_name == 'fluency':
                    suggestions.append("Improve naturalness and readability")
                elif cat_name == 'accuracy':
                    suggestions.append("Verify semantic accuracy against source")
                elif cat_name == 'terminology':
                    suggestions.append("Review domain-specific terminology")
                elif cat_name == 'formatting':
                    suggestions.append("Check formatting and placeholder preservation")
        
        return suggestions


def analyze_post_edits(
    mt_outputs: List[str],
    post_edits: List[str],
    sources: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze post-editing patterns to understand common error types.
    
    Args:
        mt_outputs: Machine translation outputs
        post_edits: Human post-edited versions
        sources: Original source texts
        
    Returns:
        Analysis results with statistics
    """
    from difflib import SequenceMatcher
    import re
    
    analysis = {
        'total_samples': len(mt_outputs),
        'edit_distance_stats': {},
        'common_changes': [],
        'error_patterns': {}
    }
    
    edit_ratios = []
    insertions = []
    deletions = []
    replacements = []
    
    for mt, pe in zip(mt_outputs, post_edits):
        # Compute edit operations
        matcher = SequenceMatcher(None, mt.split(), pe.split())
        ratio = matcher.ratio()
        edit_ratios.append(ratio)
        
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == 'insert':
                insertions.extend(pe.split()[j1:j2])
            elif op == 'delete':
                deletions.extend(mt.split()[i1:i2])
            elif op == 'replace':
                replacements.append({
                    'original': ' '.join(mt.split()[i1:i2]),
                    'edited': ' '.join(pe.split()[j1:j2])
                })
    
    # Statistics
    analysis['edit_distance_stats'] = {
        'mean_ratio': np.mean(edit_ratios),
        'std_ratio': np.std(edit_ratios),
        'min_ratio': np.min(edit_ratios),
        'max_ratio': np.max(edit_ratios)
    }
    
    # Common insertions/deletions
    from collections import Counter
    analysis['common_insertions'] = Counter(insertions).most_common(10)
    analysis['common_deletions'] = Counter(deletions).most_common(10)
    
    # Common replacements
    analysis['replacements_sample'] = replacements[:20]
    
    return analysis


if __name__ == "__main__":
    # Test quality estimation
    print("Testing Quality Estimation...")
    
    # Initialize without COMET (for testing)
    qe = QualityEstimator(load_comet=False)
    
    # Test data
    sources = [
        "Transfer open apps from your phone to your PC",
        "{1}Update window: From {2} to {3}"
    ]
    
    translations = [
        "Breng open apps over van je telefoon naar je pc",
        "Updateperiode: van {2} tot {3}"  # Missing {1}
    ]
    
    references = [
        "Breng open apps over van je telefoon naar je pc of tablet met een snelle veegbeweging",
        "{1}Updateperiode: van {2} tot {3}"
    ]
    
    # Test feedback generation
    approx = LinguistFeedbackApproximator(qe)
    
    print("\n=== Feedback Generation Test ===\n")
    for src, mt, ref in zip(sources, translations, references):
        feedback = approx.generate_feedback(src, mt, reference=ref)
        print(f"Source: {src}")
        print(f"MT: {mt}")
        print(f"Reference: {ref}")
        print(f"Overall Quality: {feedback['overall_quality']}")
        print(f"Edit Effort: {feedback['edit_effort']}/5")
        print(f"Suggestions: {feedback['suggestions']}")
        print()
