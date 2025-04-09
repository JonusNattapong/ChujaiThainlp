"""
Thai Dialect-Aware Tokenizer

This module provides tokenization support for various Thai regional dialects,
extending the standard ThaiTokenizer with dialect-specific capabilities.
"""

from typing import List, Optional, Union, Dict, Any
import re
from ..tokenization.tokenizer import ThaiTokenizer
from ..tokenization.maximum_matching import MaximumMatchingTokenizer
from .dialect_processor import DIALECTS, DIALECT_FEATURES, ThaiDialectProcessor
from ..utils.thai_utils import normalize_text, separate_thai_english, contains_thai
import torch
class DialectTokenizer(ThaiTokenizer):
    """Tokenizer with support for Thai dialects"""
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_thai_dict: bool = True,
        dialect: str = "central"
    ):
        """Initialize dialect-aware tokenizer
        
        Args:
            model_name: Pretrained model name for subword tokenization
            device: Device to run model on
            use_thai_dict: Whether to use Thai dictionary for word segmentation
            dialect: Default dialect to use (northern, northeastern, southern, central)
        """
        super().__init__(model_name, device, use_thai_dict)
        
        # Initialize dialect resources
        self.dialect = dialect if dialect in DIALECTS else "central"
        self.dialect_processor = ThaiDialectProcessor()
        
        # Load dialect-specific vocabulary
        self._init_dialect_vocabulary()
    
    def _init_dialect_vocabulary(self):
        """Initialize vocabulary specific to dialects"""
        self.dialect_words = {}
        
        for dialect, features in DIALECT_FEATURES.items():
            # Extract all vocabulary words specific to this dialect
            vocabulary = {}
            
            if "vocabulary" in features:
                vocabulary.update(features["vocabulary"])
            
            # Add particles, pronouns, and modifiers
            for category in ["particles", "pronouns", "verb_modifiers"]:
                if category in features:
                    for word in features[category]:
                        vocabulary[word] = word  # Map to itself
            
            self.dialect_words[dialect] = vocabulary
            
        # Add dialect words to dictionary tokenizer if used
        if self.thai_tokenizer:
            for dialect, vocab in self.dialect_words.items():
                for word in vocab:
                    self.thai_tokenizer.add_custom_word(word)
    
    def set_dialect(self, dialect: str):
        """Set the active dialect for tokenization
        
        Args:
            dialect: Dialect code (northern, northeastern, southern, central)
        """
        if dialect in DIALECTS:
            self.dialect = dialect
    
    def get_active_dialect(self) -> str:
        """Get the currently active dialect
        
        Returns:
            Active dialect code
        """
        return self.dialect
    
    def tokenize(self,
                text: Union[str, List[str]],
                dialect: Optional[str] = None,
                mode: str = "word",
                return_tensors: Optional[str] = None) -> Union[List[str], List[List[str]]]:
        """Tokenize text based on dialect
        
        Args:
            text: Text to tokenize
            dialect: Override the default dialect
            mode: Tokenization mode ('word' or 'subword')
            return_tensors: Whether to return tensors and which format
            
        Returns:
            List of tokens or list of token lists if input is a list
        """
        # Use specified dialect or default
        active_dialect = dialect if dialect in DIALECTS else self.dialect
        
        # Handle list input
        if isinstance(text, list):
            return [self.tokenize(t, active_dialect, mode, return_tensors) for t in text]
        
        # Normalize input text
        text = normalize_text(text)
        
        if mode == "word":
            return self._word_tokenize_dialect(text, active_dialect)
        else:
            return self._subword_tokenize(text, return_tensors)
    
    def _word_tokenize_dialect(self, text: str, dialect: str) -> List[str]:
        """Tokenize text into words with dialect awareness
        
        Args:
            text: Text to tokenize
            dialect: Dialect to use for tokenization
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # Separate Thai and English segments
        segments = separate_thai_english(text)
        tokens = []
        
        for segment in segments:
            if contains_thai(segment):
                # Use dialect-aware Thai word segmentation
                if self.thai_tokenizer:
                    # Add any dialect-specific words to the dictionary first
                    dialect_vocab = self.dialect_words.get(dialect, {})
                    for word in dialect_vocab:
                        if word in segment and len(word) > 1:
                            self.thai_tokenizer.add_custom_word(word)
                    
                    # Use the enhanced tokenizer
                    tokens.extend(self.thai_tokenizer.tokenize(segment))
                else:
                    # Fallback to character tokenization for Thai
                    tokens.extend(list(segment))
            else:
                # Use regex for English word boundaries
                tokens.extend(self.eng_word_pattern.findall(segment))
                
        return tokens
    
    def detect_and_tokenize(self, text: str, mode: str = "word") -> Dict[str, Any]:
        """Automatically detect dialect and tokenize accordingly
        
        Args:
            text: Text to analyze and tokenize
            mode: Tokenization mode ('word' or 'subword')
            
        Returns:
            Dictionary with detected dialect and tokens
        """
        # Detect dialect
        dialect_scores = self.dialect_processor.detect_dialect(text)
        detected_dialect = max(dialect_scores, key=lambda k: dialect_scores[k])
        
        # Tokenize using the detected dialect
        tokens = self.tokenize(text, dialect=detected_dialect, mode=mode)
        
        return {
            "dialect": detected_dialect,
            "dialect_confidence": dialect_scores[detected_dialect],
            "tokens": tokens,
            "all_dialects": dialect_scores
        }
    
    def add_dialect_words(self, dialect: str, words: List[str]):
        """Add custom words for a specific dialect
        
        Args:
            dialect: Dialect code
            words: List of words to add
        """
        if dialect not in self.dialect_words:
            self.dialect_words[dialect] = {}
            
        for word in words:
            self.dialect_words[dialect][word] = word
            
            # Also add to tokenizer dictionary
            if self.thai_tokenizer:
                self.thai_tokenizer.add_custom_word(word)