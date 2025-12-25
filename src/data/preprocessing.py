"""
Text Preprocessing Module
=========================
Robust preprocessing pipeline for machine translation with
domain-specific handling for software/IT content.

Features:
- Placeholder preservation ({1}, {2}, %s, etc.)
- Technical term normalization
- Unicode normalization
- Tokenization preparation
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import string


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    lowercase: bool = False  # Generally False for MT
    normalize_unicode: bool = True
    preserve_placeholders: bool = True
    preserve_technical_terms: bool = True
    strip_extra_whitespace: bool = True
    normalize_punctuation: bool = True
    max_length: Optional[int] = None


class TextPreprocessor:
    """
    Comprehensive text preprocessor for machine translation.
    
    Handles:
    - Software/IT domain specific content
    - Placeholder variables ({1}, {2}, %s, ${var})
    - Technical terms and brand names
    - Unicode normalization
    - Whitespace handling
    """
    
    # Placeholder patterns to preserve
    PLACEHOLDER_PATTERNS = [
        (r'\{(\d+)\}', '__PLACEHOLDER_NUM_{}__'),           # {1}, {2}
        (r'\{([^}]+)\}', '__PLACEHOLDER_VAR_{}__'),         # {variable}
        (r'%([sd%])', '__PERCENT_{}__'),                     # %s, %d, %%
        (r'\$\{([^}]+)\}', '__DOLLAR_VAR_{}__'),            # ${variable}
        (r'\\n', '__NEWLINE__'),                             # Literal \n
        (r'\\t', '__TAB__'),                                 # Literal \t
    ]
    
    # Technical terms to preserve case
    TECHNICAL_TERMS = {
        'USB', 'USB-C', 'HDMI', 'Bluetooth', 'Wi-Fi', 'WiFi',
        'PTT', 'LPDDR5', 'UFS', 'TurboPower', 'Ready For',
        'EMM', 'PC', 'AI', 'ML', 'NLP', 'API', 'SDK',
        'Android', 'iOS', 'Windows', 'Linux', 'macOS',
        'Chrome', 'Firefox', 'Safari', 'Edge',
        'Motorola', 'Moto', 'Verizon', 'Google', 'Samsung'
    }
    
    # Brand names with special handling
    BRAND_PATTERNS = [
        (r'TurboPower™', '__TURBOPOWER_TM__'),
        (r'Ready For', '__READY_FOR__'),
        (r'Family Space', '__FAMILY_SPACE__'),
        (r'Experience hub', '__EXPERIENCE_HUB__'),
    ]
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._placeholder_map: Dict[str, str] = {}
        self._placeholder_counter = 0
        
    def preprocess(self, text: str, is_source: bool = True) -> str:
        """
        Main preprocessing pipeline.
        
        Args:
            text: Input text
            is_source: Whether this is source text (affects some processing)
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Step 1: Unicode normalization
        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Step 2: Preserve placeholders (replace with tokens)
        if self.config.preserve_placeholders:
            text, self._placeholder_map = self._protect_placeholders(text)
        
        # Step 3: Preserve brand names
        if self.config.preserve_technical_terms:
            text = self._protect_brand_names(text)
        
        # Step 4: Normalize whitespace
        if self.config.strip_extra_whitespace:
            text = self._normalize_whitespace(text)
        
        # Step 5: Normalize punctuation
        if self.config.normalize_punctuation:
            text = self._normalize_punctuation(text)
        
        # Step 6: Apply length limit
        if self.config.max_length:
            text = self._truncate(text, self.config.max_length)
        
        return text
    
    def postprocess(self, text: str, placeholder_map: Optional[Dict[str, str]] = None) -> str:
        """
        Restore placeholders and special tokens after translation.
        
        Args:
            text: Translated text
            placeholder_map: Map of placeholder tokens to original values
            
        Returns:
            Text with restored placeholders
        """
        if placeholder_map is None:
            placeholder_map = self._placeholder_map
        
        # Restore placeholders
        for token, original in placeholder_map.items():
            text = text.replace(token, original)
        
        # Restore brand names
        text = self._restore_brand_names(text)
        
        # Final cleanup
        text = self._normalize_whitespace(text)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode to NFKC form."""
        # NFKC: Compatibility decomposition followed by canonical composition
        text = unicodedata.normalize('NFKC', text)
        return text
    
    def _protect_placeholders(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Replace placeholders with tokens and return mapping."""
        placeholder_map = {}
        
        for pattern, token_template in self.PLACEHOLDER_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                original = match.group(0)
                if original not in placeholder_map.values():
                    token = token_template.format(self._placeholder_counter)
                    self._placeholder_counter += 1
                    placeholder_map[token] = original
                    text = text.replace(original, token, 1)
        
        return text, placeholder_map
    
    def _protect_brand_names(self, text: str) -> str:
        """Replace brand names with tokens."""
        for pattern, token in self.BRAND_PATTERNS:
            text = re.sub(pattern, token, text, flags=re.IGNORECASE)
        return text
    
    def _restore_brand_names(self, text: str) -> str:
        """Restore brand name tokens."""
        restorations = {
            '__TURBOPOWER_TM__': 'TurboPower™',
            '__READY_FOR__': 'Ready For',
            '__FAMILY_SPACE__': 'Family Space',
            '__EXPERIENCE_HUB__': 'Experience hub',
        }
        for token, original in restorations.items():
            text = text.replace(token, original)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        # Normalize quotes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r"['']", "'", text)
        
        # Normalize dashes
        text = re.sub(r'[–—]', '-', text)
        
        # Normalize ellipsis
        text = re.sub(r'\.{3,}', '...', text)
        
        return text
    
    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length (word-level)."""
        words = text.split()
        if len(words) <= max_length:
            return text
        return ' '.join(words[:max_length])
    
    def detect_language_features(self, text: str) -> Dict:
        """
        Detect language and domain features for analysis.
        
        Returns:
            Dictionary with detected features
        """
        features = {
            'length_chars': len(text),
            'length_words': len(text.split()),
            'has_placeholders': bool(re.search(r'\{[^}]+\}', text)),
            'has_percent_vars': bool(re.search(r'%[sd%]', text)),
            'has_technical_terms': any(term.lower() in text.lower() for term in self.TECHNICAL_TERMS),
            'has_brand_names': any(brand in text for brand, _ in self.BRAND_PATTERNS),
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_urls': bool(re.search(r'https?://', text)),
            'has_special_chars': bool(re.search(r'[™®©]', text)),
        }
        return features


class DataAugmenter:
    """
    Data augmentation techniques for machine translation.
    
    Techniques:
    - Back-translation (requires model)
    - Placeholder variations
    - Case variations
    - Synonym replacement
    """
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        self.preprocessor = preprocessor or TextPreprocessor()
    
    def augment_with_placeholder_variations(
        self,
        source: str,
        target: str,
        n_variations: int = 3
    ) -> List[Tuple[str, str]]:
        """
        Generate variations by modifying placeholder values.
        
        Args:
            source: Source text
            target: Target text
            n_variations: Number of variations to generate
            
        Returns:
            List of (source, target) tuples
        """
        variations = [(source, target)]  # Original
        
        # Find numeric placeholders
        placeholders = re.findall(r'\{(\d+)\}', source)
        if not placeholders:
            return variations
        
        # Generate variations with different placeholder orders
        for i in range(n_variations):
            new_source = source
            new_target = target
            
            # Simple variation: swap placeholders if multiple exist
            if len(placeholders) >= 2:
                for j in range(len(placeholders) - 1):
                    old1 = f'{{{placeholders[j]}}}'
                    old2 = f'{{{placeholders[j+1]}}}'
                    
                    # Create swapped version
                    temp_source = new_source.replace(old1, '__TEMP__')
                    temp_source = temp_source.replace(old2, old1)
                    temp_source = temp_source.replace('__TEMP__', old2)
                    
                    temp_target = new_target.replace(old1, '__TEMP__')
                    temp_target = temp_target.replace(old2, old1)
                    temp_target = temp_target.replace('__TEMP__', old2)
                    
                    if (temp_source, temp_target) not in variations:
                        variations.append((temp_source, temp_target))
                        if len(variations) >= n_variations + 1:
                            break
        
        return variations[:n_variations + 1]


class SoftwareTermGlossary:
    """
    Glossary of software/IT terms for consistent translation.
    """
    
    # English -> Dutch glossary for common software terms
    EN_NL_GLOSSARY = {
        'settings': 'instellingen',
        'permission': 'toestemming',
        'permissions': 'machtigingen',
        'device': 'apparaat',
        'devices': 'apparaten',
        'screen': 'scherm',
        'display': 'display',
        'battery': 'accu',
        'notification': 'melding',
        'notifications': 'meldingen',
        'account': 'account',
        'password': 'wachtwoord',
        'username': 'gebruikersnaam',
        'download': 'downloaden',
        'upload': 'uploaden',
        'file': 'bestand',
        'files': 'bestanden',
        'folder': 'map',
        'app': 'app',
        'application': 'applicatie',
        'update': 'update',
        'upgrade': 'upgrade',
        'install': 'installeren',
        'uninstall': 'verwijderen',
        'sync': 'synchroniseren',
        'backup': 'back-up',
        'restore': 'herstellen',
        'connect': 'verbinden',
        'disconnect': 'verbinding verbreken',
        'wireless': 'draadloos',
        'cable': 'kabel',
        'charging': 'opladen',
        'power': 'energie',
        'mode': 'modus',
        'gesture': 'gebaar',
        'swipe': 'vegen',
        'tap': 'tikken',
        'hold': 'ingedrukt houden',
        'drag': 'slepen',
        'drop': 'neerzetten',
    }
    
    def __init__(self):
        self.glossary = self.EN_NL_GLOSSARY.copy()
    
    def add_term(self, english: str, dutch: str):
        """Add a term to the glossary."""
        self.glossary[english.lower()] = dutch
    
    def get_translation(self, term: str) -> Optional[str]:
        """Get Dutch translation for an English term."""
        return self.glossary.get(term.lower())
    
    def contains(self, term: str) -> bool:
        """Check if term is in glossary."""
        return term.lower() in self.glossary


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "30W TurboPower™ charging support*sold separately",
        "{1}Update window: From {2} to {3}",
        "Go to Settings, choose Permissions, and then allow 'Connect to nearby devices.'",
        "Transfer open apps from your phone to your PC or tablet using a fling gesture",
    ]
    
    print("=== Preprocessing Tests ===\n")
    for text in test_texts:
        processed = preprocessor.preprocess(text)
        features = preprocessor.detect_language_features(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print(f"Features: {features}")
        print()
