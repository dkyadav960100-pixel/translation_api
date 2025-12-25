"""
Feature Store Module
====================
Feature engineering and storage for machine translation.

Supports:
- Text-level features (length, complexity)
- Domain-specific features (software terms, placeholders)
- Linguistic features (POS tags, entities)
- Cached feature computation
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import re

import numpy as np
import pandas as pd


@dataclass
class TranslationFeatures:
    """Features extracted from a translation pair."""
    sample_id: str
    
    # Source text features
    source_length_chars: int
    source_length_words: int
    source_length_tokens: Optional[int] = None
    source_avg_word_length: float = 0.0
    
    # Target text features
    target_length_chars: int = 0
    target_length_words: int = 0
    target_length_tokens: Optional[int] = None
    target_avg_word_length: float = 0.0
    
    # Ratio features
    length_ratio: float = 1.0  # target/source
    
    # Placeholder features
    num_placeholders: int = 0
    placeholder_types: List[str] = None
    
    # Domain features
    num_technical_terms: int = 0
    technical_terms: List[str] = None
    has_brand_names: bool = False
    
    # Complexity features
    source_unique_words: int = 0
    target_unique_words: int = 0
    lexical_diversity_source: float = 0.0
    lexical_diversity_target: float = 0.0
    
    # Special character features
    has_urls: bool = False
    has_numbers: bool = False
    has_special_punctuation: bool = False
    
    def __post_init__(self):
        if self.placeholder_types is None:
            self.placeholder_types = []
        if self.technical_terms is None:
            self.technical_terms = []


class FeatureExtractor:
    """
    Extract features from translation pairs for analysis and modeling.
    """
    
    TECHNICAL_TERMS = {
        'usb', 'usb-c', 'hdmi', 'bluetooth', 'wifi', 'wi-fi',
        'ptt', 'lpddr5', 'ufs', 'turbopower', 'ready for',
        'emm', 'pc', 'ai', 'ml', 'nlp', 'api', 'sdk',
        'android', 'ios', 'windows', 'linux', 'macos',
        'app', 'apps', 'phone', 'tablet', 'device', 'display',
        'battery', 'screen', 'notification', 'settings', 'permission'
    }
    
    def __init__(self, tokenizer=None):
        """
        Initialize feature extractor.
        
        Args:
            tokenizer: Optional HuggingFace tokenizer for token counts
        """
        self.tokenizer = tokenizer
    
    def extract_features(self, source: str, target: str, sample_id: str) -> TranslationFeatures:
        """
        Extract comprehensive features from a translation pair.
        
        Args:
            source: Source text
            target: Target text (can be empty for source-only analysis)
            sample_id: Unique identifier for this sample
            
        Returns:
            TranslationFeatures object
        """
        # Source features
        source_words = source.split()
        source_length_chars = len(source)
        source_length_words = len(source_words)
        source_avg_word_length = np.mean([len(w) for w in source_words]) if source_words else 0
        source_unique = len(set(w.lower() for w in source_words))
        
        # Target features
        target_words = target.split() if target else []
        target_length_chars = len(target) if target else 0
        target_length_words = len(target_words)
        target_avg_word_length = np.mean([len(w) for w in target_words]) if target_words else 0
        target_unique = len(set(w.lower() for w in target_words)) if target_words else 0
        
        # Token counts (if tokenizer available)
        source_tokens = None
        target_tokens = None
        if self.tokenizer:
            source_tokens = len(self.tokenizer.encode(source))
            if target:
                target_tokens = len(self.tokenizer.encode(target))
        
        # Length ratio
        length_ratio = target_length_words / source_length_words if source_length_words > 0 else 1.0
        
        # Placeholder detection
        placeholders = re.findall(r'\{[^}]+\}', source) + re.findall(r'%[sd%]', source)
        placeholder_types = list(set(placeholders))
        
        # Technical terms
        source_lower = source.lower()
        tech_terms = [term for term in self.TECHNICAL_TERMS if term in source_lower]
        
        # Brand names
        has_brands = any(brand in source for brand in ['TurboPower', 'Ready For', 'Moto', 'Motorola'])
        
        # Lexical diversity
        lex_div_source = source_unique / source_length_words if source_length_words > 0 else 0
        lex_div_target = target_unique / target_length_words if target_length_words > 0 else 0
        
        # Special features
        has_urls = bool(re.search(r'https?://', source))
        has_numbers = bool(re.search(r'\d+', source))
        has_special_punct = bool(re.search(r'[™®©*#]', source))
        
        return TranslationFeatures(
            sample_id=sample_id,
            source_length_chars=source_length_chars,
            source_length_words=source_length_words,
            source_length_tokens=source_tokens,
            source_avg_word_length=source_avg_word_length,
            target_length_chars=target_length_chars,
            target_length_words=target_length_words,
            target_length_tokens=target_tokens,
            target_avg_word_length=target_avg_word_length,
            length_ratio=length_ratio,
            num_placeholders=len(placeholders),
            placeholder_types=placeholder_types,
            num_technical_terms=len(tech_terms),
            technical_terms=tech_terms,
            has_brand_names=has_brands,
            source_unique_words=source_unique,
            target_unique_words=target_unique,
            lexical_diversity_source=lex_div_source,
            lexical_diversity_target=lex_div_target,
            has_urls=has_urls,
            has_numbers=has_numbers,
            has_special_punctuation=has_special_punct
        )
    
    def extract_batch(self, samples: List[Dict]) -> List[TranslationFeatures]:
        """Extract features for a batch of samples."""
        return [
            self.extract_features(
                s.get('source', ''),
                s.get('target', ''),
                s.get('sample_id', str(i))
            )
            for i, s in enumerate(samples)
        ]


class FeatureStore:
    """
    Persistent feature storage with caching.
    """
    
    def __init__(self, store_path: Path):
        """
        Initialize feature store.
        
        Args:
            store_path: Directory for storing features
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_path / "index.json"
        self._load_index()
    
    def _load_index(self):
        """Load feature index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {}
    
    def _save_index(self):
        """Save feature index."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _compute_hash(self, data: str) -> str:
        """Compute hash for cache invalidation."""
        return hashlib.md5(data.encode()).hexdigest()
    
    def store_features(self, name: str, features: List[TranslationFeatures]):
        """
        Store features to disk.
        
        Args:
            name: Dataset/feature set name
            features: List of TranslationFeatures
        """
        feature_path = self.store_path / f"{name}_features.json"
        
        data = [asdict(f) for f in features]
        with open(feature_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Update index
        self.index[name] = {
            "path": str(feature_path),
            "count": len(features),
            "hash": self._compute_hash(str(data))
        }
        self._save_index()
    
    def load_features(self, name: str) -> Optional[List[TranslationFeatures]]:
        """
        Load features from disk.
        
        Args:
            name: Dataset/feature set name
            
        Returns:
            List of TranslationFeatures or None if not found
        """
        if name not in self.index:
            return None
        
        feature_path = Path(self.index[name]["path"])
        if not feature_path.exists():
            return None
        
        with open(feature_path, 'r') as f:
            data = json.load(f)
        
        return [TranslationFeatures(**d) for d in data]
    
    def to_dataframe(self, features: List[TranslationFeatures]) -> pd.DataFrame:
        """Convert features to pandas DataFrame."""
        data = [asdict(f) for f in features]
        df = pd.DataFrame(data)
        
        # Convert list columns to string for easier handling
        for col in ['placeholder_types', 'technical_terms']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ','.join(x) if x else '')
        
        return df
    
    def get_statistics(self, features: List[TranslationFeatures]) -> Dict[str, Any]:
        """Compute aggregate statistics from features."""
        df = self.to_dataframe(features)
        
        numeric_cols = [
            'source_length_chars', 'source_length_words',
            'target_length_chars', 'target_length_words',
            'length_ratio', 'num_placeholders', 'num_technical_terms',
            'lexical_diversity_source', 'lexical_diversity_target'
        ]
        
        stats = {
            'count': len(features),
            'numeric_stats': df[numeric_cols].describe().to_dict(),
            'samples_with_placeholders': (df['num_placeholders'] > 0).sum(),
            'samples_with_tech_terms': (df['num_technical_terms'] > 0).sum(),
            'samples_with_brands': df['has_brand_names'].sum(),
            'samples_with_urls': df['has_urls'].sum(),
            'samples_with_numbers': df['has_numbers'].sum(),
        }
        
        return stats


if __name__ == "__main__":
    # Test feature extraction
    extractor = FeatureExtractor()
    
    test_pairs = [
        {
            'source': "30W TurboPower™ charging support*sold separately",
            'target': "Ondersteuning voor TurboPower™ laden met 30 W*apart verkocht",
            'sample_id': 'test_1'
        },
        {
            'source': "{1}Update window: From {2} to {3}",
            'target': "{1}Updateperiode: van {2} tot {3}",
            'sample_id': 'test_2'
        }
    ]
    
    print("=== Feature Extraction Tests ===\n")
    for pair in test_pairs:
        features = extractor.extract_features(
            pair['source'], pair['target'], pair['sample_id']
        )
        print(f"Sample: {pair['sample_id']}")
        print(f"  Source length: {features.source_length_words} words")
        print(f"  Target length: {features.target_length_words} words")
        print(f"  Length ratio: {features.length_ratio:.2f}")
        print(f"  Placeholders: {features.num_placeholders} ({features.placeholder_types})")
        print(f"  Technical terms: {features.technical_terms}")
        print()
