"""
Edge Case Handling Module
=========================
Robust handling of edge cases and fallback mechanisms for MT system.

Handles:
- Empty/invalid inputs
- Placeholder preservation
- Technical term handling
- OOV (Out-of-Vocabulary) tokens
- Length constraints
- Encoding issues
- Timeout/failure recovery
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from functools import wraps
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EdgeCaseResult:
    """Result from edge case processing."""
    original_input: str
    processed_input: str
    output: str
    warnings: List[str]
    fallback_used: bool
    processing_time: float


class EdgeCaseHandler:
    """
    Comprehensive edge case handler for machine translation.
    
    Features:
    - Input validation and sanitization
    - Placeholder preservation
    - Technical term handling
    - Fallback mechanisms
    - Error recovery
    """
    
    # Placeholder patterns
    PLACEHOLDER_PATTERNS = [
        (r'\{(\d+)\}', 'PLACEHOLDER_NUM'),           # {1}, {2}
        (r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', 'PLACEHOLDER_VAR'),  # {variable}
        (r'%([sd%])', 'PERCENT_VAR'),                # %s, %d, %%
        (r'\$\{([^}]+)\}', 'DOLLAR_VAR'),            # ${variable}
    ]
    
    # Technical terms to preserve (case-sensitive)
    PRESERVE_TERMS = {
        'USB', 'USB-C', 'HDMI', 'Bluetooth', 'Wi-Fi', 'WiFi',
        'LPDDR5', 'UFS', 'TurboPower™', 'Ready For',
        'Moto', 'Motorola', 'Android', 'iOS', 'Windows',
        'PTT', 'EMM', 'SDK', 'API', 'NLP', 'ML', 'AI',
        '4G', '5G', 'LTE', 'NFC', 'GPS', 'OLED', 'LCD'
    }
    
    # Default translations for common terms (EN -> NL)
    FALLBACK_TRANSLATIONS = {
        'settings': 'instellingen',
        'device': 'apparaat',
        'battery': 'accu',
        'display': 'display',
        'notification': 'melding',
        'app': 'app',
        'phone': 'telefoon',
        'tablet': 'tablet',
        'PC': 'pc',
        'update': 'update',
    }
    
    def __init__(
        self,
        max_length: int = 1024,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        fallback_response: str = "[Translation unavailable]"
    ):
        """
        Initialize edge case handler.
        
        Args:
            max_length: Maximum input length (characters)
            timeout_seconds: Timeout for translation operations
            max_retries: Maximum retry attempts
            fallback_response: Default response on failure
        """
        self.max_length = max_length
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.fallback_response = fallback_response
        
        # Cache for placeholder mappings
        self._placeholder_map: Dict[str, str] = {}
    
    def validate_input(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate input text for potential issues.
        
        Args:
            text: Input text to validate
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Empty input
        if not text or not text.strip():
            warnings.append("Empty or whitespace-only input")
            return False, warnings
        
        # Length check
        if len(text) > self.max_length:
            warnings.append(f"Input exceeds max length ({len(text)} > {self.max_length})")
        
        # Encoding issues
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            warnings.append("Input contains invalid UTF-8 characters")
        
        # Check for problematic characters
        if '\x00' in text:
            warnings.append("Input contains null characters")
        
        # Check for excessive special characters
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_ratio > 0.5:
            warnings.append("Input has high ratio of special characters")
        
        is_valid = len(warnings) == 0 or all('max length' not in w for w in warnings)
        return is_valid, warnings
    
    def sanitize_input(self, text: str) -> str:
        """
        Sanitize input text.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Remove null characters
        text = text.replace('\x00', '')
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long
        if len(text) > self.max_length:
            text = text[:self.max_length]
            # Try to break at word boundary
            last_space = text.rfind(' ')
            if last_space > self.max_length * 0.8:
                text = text[:last_space]
        
        return text.strip()
    
    def preserve_placeholders(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replace placeholders with tokens for preservation during translation.
        
        Args:
            text: Input text with placeholders
            
        Returns:
            Tuple of (processed_text, placeholder_map)
        """
        placeholder_map = {}
        counter = 0
        
        for pattern, prefix in self.PLACEHOLDER_PATTERNS:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                original = match.group(0)
                if original not in [v for v in placeholder_map.values()]:
                    token = f"__{prefix}_{counter}__"
                    placeholder_map[token] = original
                    text = text.replace(original, token, 1)
                    counter += 1
        
        self._placeholder_map = placeholder_map
        return text, placeholder_map
    
    def restore_placeholders(self, text: str, placeholder_map: Dict[str, str] = None) -> str:
        """
        Restore placeholders in translated text.
        
        Args:
            text: Translated text with placeholder tokens
            placeholder_map: Mapping of tokens to original placeholders
            
        Returns:
            Text with restored placeholders
        """
        if placeholder_map is None:
            placeholder_map = self._placeholder_map
        
        for token, original in placeholder_map.items():
            text = text.replace(token, original)
        
        return text
    
    def preserve_technical_terms(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Preserve technical terms that shouldn't be translated.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (processed_text, term_map)
        """
        term_map = {}
        counter = 0
        
        for term in sorted(self.PRESERVE_TERMS, key=len, reverse=True):
            if term in text:
                token = f"__TERM_{counter}__"
                term_map[token] = term
                text = text.replace(term, token)
                counter += 1
        
        return text, term_map
    
    def restore_technical_terms(self, text: str, term_map: Dict[str, str]) -> str:
        """Restore preserved technical terms."""
        for token, original in term_map.items():
            text = text.replace(token, original)
        return text
    
    def handle_empty_input(self, text: str) -> Tuple[str, bool]:
        """
        Handle empty or whitespace-only input.
        
        Returns:
            Tuple of (result, was_empty)
        """
        if not text or not text.strip():
            return "", True
        return text, False
    
    def handle_translation_failure(
        self,
        source: str,
        error: Exception,
        attempt: int
    ) -> str:
        """
        Handle translation failure with graceful fallback.
        
        Args:
            source: Original source text
            error: Exception that occurred
            attempt: Current attempt number
            
        Returns:
            Fallback translation or error message
        """
        logger.warning(f"Translation failed (attempt {attempt}): {str(error)}")
        
        # Try simple word-by-word fallback
        if attempt < self.max_retries:
            try:
                words = source.lower().split()
                translated_words = []
                for word in words:
                    if word in self.FALLBACK_TRANSLATIONS:
                        translated_words.append(self.FALLBACK_TRANSLATIONS[word])
                    else:
                        translated_words.append(word)
                return ' '.join(translated_words)
            except:
                pass
        
        return self.fallback_response
    
    def process_translation(
        self,
        source: str,
        translate_fn: Callable[[str], str]
    ) -> EdgeCaseResult:
        """
        Process a translation with full edge case handling.
        
        Args:
            source: Source text
            translate_fn: Translation function to call
            
        Returns:
            EdgeCaseResult with translation and metadata
        """
        start_time = time.time()
        warnings = []
        fallback_used = False
        
        # Validate input
        is_valid, validation_warnings = self.validate_input(source)
        warnings.extend(validation_warnings)
        
        # Handle empty input
        source, was_empty = self.handle_empty_input(source)
        if was_empty:
            return EdgeCaseResult(
                original_input=source,
                processed_input="",
                output="",
                warnings=["Empty input"],
                fallback_used=False,
                processing_time=time.time() - start_time
            )
        
        # Sanitize
        processed = self.sanitize_input(source)
        
        # Preserve placeholders
        processed, placeholder_map = self.preserve_placeholders(processed)
        
        # Preserve technical terms
        processed, term_map = self.preserve_technical_terms(processed)
        
        # Attempt translation with retries
        output = None
        for attempt in range(self.max_retries):
            try:
                output = translate_fn(processed)
                break
            except Exception as e:
                if attempt == self.max_retries - 1:
                    output = self.handle_translation_failure(source, e, attempt)
                    fallback_used = True
                    warnings.append(f"Translation failed: {str(e)}")
                else:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        # Restore technical terms
        if output:
            output = self.restore_technical_terms(output, term_map)
        
        # Restore placeholders
        if output:
            output = self.restore_placeholders(output, placeholder_map)
        
        return EdgeCaseResult(
            original_input=source,
            processed_input=processed,
            output=output or self.fallback_response,
            warnings=warnings,
            fallback_used=fallback_used,
            processing_time=time.time() - start_time
        )


def with_edge_case_handling(handler: EdgeCaseHandler = None):
    """
    Decorator for adding edge case handling to translation functions.
    
    Usage:
        @with_edge_case_handling()
        def translate(text):
            return model.translate(text)
    """
    if handler is None:
        handler = EdgeCaseHandler()
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(text: str, *args, **kwargs):
            result = handler.process_translation(
                text,
                lambda t: func(t, *args, **kwargs)
            )
            return result.output
        return wrapper
    return decorator


class BatchEdgeCaseHandler:
    """
    Batch processing with edge case handling.
    """
    
    def __init__(self, handler: EdgeCaseHandler = None):
        self.handler = handler or EdgeCaseHandler()
    
    def process_batch(
        self,
        sources: List[str],
        translate_fn: Callable[[List[str]], List[str]]
    ) -> List[EdgeCaseResult]:
        """
        Process a batch of translations with edge case handling.
        
        Args:
            sources: List of source texts
            translate_fn: Batch translation function
            
        Returns:
            List of EdgeCaseResult objects
        """
        results = []
        
        # Pre-process all inputs
        processed_sources = []
        metadata = []  # Store placeholder and term maps
        
        for source in sources:
            # Sanitize
            processed = self.handler.sanitize_input(source)
            
            # Preserve placeholders
            processed, placeholder_map = self.handler.preserve_placeholders(processed)
            
            # Preserve technical terms
            processed, term_map = self.handler.preserve_technical_terms(processed)
            
            processed_sources.append(processed)
            metadata.append({
                'original': source,
                'placeholder_map': placeholder_map,
                'term_map': term_map
            })
        
        # Batch translate
        try:
            translations = translate_fn(processed_sources)
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            translations = [self.handler.fallback_response] * len(sources)
        
        # Post-process
        for i, (translation, meta) in enumerate(zip(translations, metadata)):
            # Restore technical terms
            translation = self.handler.restore_technical_terms(
                translation, meta['term_map']
            )
            
            # Restore placeholders
            translation = self.handler.restore_placeholders(
                translation, meta['placeholder_map']
            )
            
            results.append(EdgeCaseResult(
                original_input=meta['original'],
                processed_input=processed_sources[i],
                output=translation,
                warnings=[],
                fallback_used=False,
                processing_time=0.0
            ))
        
        return results


if __name__ == "__main__":
    # Test edge case handling
    handler = EdgeCaseHandler()
    
    test_cases = [
        "",  # Empty
        "   ",  # Whitespace only
        "Normal translation test",
        "{1}Update window: From {2} to {3}",  # Placeholders
        "Connect your USB-C device to Bluetooth",  # Technical terms
        "30W TurboPower™ charging",  # Brand names with trademark
        "A" * 2000,  # Too long
    ]
    
    print("=== Edge Case Testing ===\n")
    
    def mock_translate(text):
        """Mock translation function."""
        return f"[TRANSLATED: {text[:50]}...]" if len(text) > 50 else f"[TRANSLATED: {text}]"
    
    for test in test_cases:
        result = handler.process_translation(test, mock_translate)
        print(f"Input: {repr(test[:50])}{'...' if len(test) > 50 else ''}")
        print(f"Output: {result.output}")
        print(f"Warnings: {result.warnings}")
        print(f"Fallback used: {result.fallback_used}")
        print()
