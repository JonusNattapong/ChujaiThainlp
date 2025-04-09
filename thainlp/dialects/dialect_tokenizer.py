"""
Thai Dialect-Aware Tokenizer

This module provides tokenization facilities that are aware of Thai dialects.
It's capable of correctly tokenizing text in different Thai regional dialects.
"""

import re
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from pathlib import Path
import json

from ..tokenization.tokenizer import ThaiTokenizer
from ..utils.thai_utils import normalize_text
from .dialect_processor import ThaiDialectProcessor, detect_dialect


class DialectTokenizer:
    """Tokenizer that accounts for Thai dialect variations"""
    
    def __init__(
        self,
        tokenizer: Optional[ThaiTokenizer] = None,
        dialect: Optional[str] = None,
        auto_detect: bool = True,
        data_dir: Optional[str] = None
    ):
        """Initialize dialect tokenizer
        
        Args:
            tokenizer: Base tokenizer for standard Thai
            dialect: Specific dialect to use (if known)
            auto_detect: Whether to auto-detect dialect
            data_dir: Directory containing dialect data
        """
        # Initialize or use provided tokenizer
        self.tokenizer = tokenizer or ThaiTokenizer()
        
        # Initialize dialect processor
        self.dialect_processor = ThaiDialectProcessor()
        
        # Set dialect if specified
        self.dialect = dialect
        self.auto_detect = auto_detect
        
        # Configure data directory
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Load dialect-specific token dictionaries
        self.dialect_tokens = self._load_dialect_tokens()
        
    def _load_dialect_tokens(self) -> Dict[str, Set[str]]:
        """Load dialect-specific tokens
        
        Returns:
            Dictionary mapping dialect codes to sets of special tokens
        """
        tokens_file = self.data_dir / "dialect_tokens.json"
        
        if tokens_file.exists():
            try:
                with open(tokens_file, "r", encoding="utf-8") as f:
                    loaded_tokens = json.load(f)
                    
                # Convert lists to sets for faster lookup
                return {
                    dialect: set(tokens)
                    for dialect, tokens in loaded_tokens.items()
                }
            except Exception as e:
                print(f"Error loading dialect tokens: {e}")
        
        # Default dialect tokens
        return {
            "northern": set(["เจ้า", "กำ", "ก้อ", "ละ", "เน้อ", "กา", "ปั๋น", "เปิ้น", "อู้", "ใจ้", "เฮา"]),
            "northeastern": set(["เด้อ", "สิ", "อีหลี", "อ้าย", "อีนาง", "เวา", "เบิ่ง", "กะ", "ข้อย"]),
            "southern": set(["หนิ", "โหล", "แอ", "ไซ", "หรอย", "นัก", "วั่น", "ก่อ", "ยะ"]),
            "pattani_malay": set(["มะ", "เลอ", "ยอ", "อาเกาะ", "เตะ", "มากัน", "เปอกี", "ตีโด"])
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with dialect awareness
        
        Args:
            text: Thai text to tokenize, potentially in a dialect
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Normalize text
        text = normalize_text(text)
        
        # Detect dialect if auto_detect is True and dialect is not specified
        detected_dialect = None
        if self.auto_detect and not self.dialect:
            dialect_scores = self.dialect_processor.detect_dialect(text)
            detected_dialect = max(dialect_scores.items(), key=lambda x: x[1])[0]
        else:
            detected_dialect = self.dialect or "central"
            
        # If it's central Thai, use standard tokenizer
        if detected_dialect == "central":
            return self.tokenizer.word_tokenize(text)
            
        # Get dialect-specific tokens
        dialect_tokens = self.dialect_tokens.get(detected_dialect, set())
        
        # Add dialect-specific tokens to the tokenizer's dictionary
        # and preserve the original dictionary to restore later
        original_dict = None
        if hasattr(self.tokenizer, 'custom_dict'):
            original_dict = self.tokenizer.custom_dict.copy()
            for token in dialect_tokens:
                self.tokenizer.custom_dict.add(token)
                
        # Tokenize with the augmented dictionary
        tokens = self.tokenizer.word_tokenize(text)
        
        # Restore original dictionary if modified
        if original_dict is not None:
            self.tokenizer.custom_dict = original_dict
            
        return tokens
        
    def tokenize_and_preserve_dialectal(self, text: str) -> List[Tuple[str, str]]:
        """Tokenize and mark dialect-specific tokens
        
        Args:
            text: Thai text to tokenize, potentially in a dialect
            
        Returns:
            List of (token, dialect|None) tuples
        """
        if not text:
            return []
            
        # Get tokens
        tokens = self.tokenize(text)
        
        # Identify dialect-specific tokens
        result = []
        for token in tokens:
            dialect_match = None
            for dialect, dialect_tokens in self.dialect_tokens.items():
                if token in dialect_tokens:
                    dialect_match = dialect
                    break
                    
            result.append((token, dialect_match))
            
        return result
        
    def add_dialect_token(self, token: str, dialect: str) -> bool:
        """Add a new dialect-specific token
        
        Args:
            token: Token to add
            dialect: Dialect code
            
        Returns:
            True if added successfully, False otherwise
        """
        if dialect not in self.dialect_tokens:
            self.dialect_tokens[dialect] = set()
            
        # Add token
        self.dialect_tokens[dialect].add(token)
        
        # Save updated tokens
        return self._save_dialect_tokens()
        
    def _save_dialect_tokens(self) -> bool:
        """Save dialect tokens to file
        
        Returns:
            True if saved successfully
        """
        tokens_file = self.data_dir / "dialect_tokens.json"
        
        try:
            # Convert sets to lists for JSON serialization
            tokens_dict = {
                dialect: list(tokens)
                for dialect, tokens in self.dialect_tokens.items()
            }
            
            with open(tokens_file, "w", encoding="utf-8") as f:
                json.dump(tokens_dict, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving dialect tokens: {e}")
            return False
            
    def count_dialect_specific_tokens(self, text: str) -> Dict[str, int]:
        """Count dialect-specific tokens in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping dialect codes to token counts
        """
        if not text:
            return {dialect: 0 for dialect in self.dialect_tokens}
            
        # Tokenize
        tokens = self.tokenize(text)
        
        # Count dialect tokens
        counts = {dialect: 0 for dialect in self.dialect_tokens}
        
        for token in tokens:
            for dialect, dialect_tokens in self.dialect_tokens.items():
                if token in dialect_tokens:
                    counts[dialect] += 1
                    
        return counts
        
    def get_dialectal_diversity(self, text: str) -> float:
        """Calculate dialectal diversity score
        
        Args:
            text: Text to analyze
            
        Returns:
            Dialectal diversity score (0.0-1.0)
        """
        if not text:
            return 0.0
            
        # Count tokens by dialect
        dialect_counts = self.count_dialect_specific_tokens(text)
        
        # Calculate total dialect tokens
        total_dialect_tokens = sum(dialect_counts.values())
        
        if total_dialect_tokens == 0:
            return 0.0
            
        # Count non-zero dialects
        non_zero_dialects = sum(1 for count in dialect_counts.values() if count > 0)
        
        # Calculate diversity (normalized by number of dialects)
        return non_zero_dialects / len(dialect_counts)


# Module-level functions
def tokenize_dialectal(text: str, dialect: Optional[str] = None) -> List[str]:
    """Tokenize text with dialect awareness
    
    Args:
        text: Thai text to tokenize
        dialect: Dialect code (auto-detected if None)
        
    Returns:
        List of tokens
    """
    tokenizer = DialectTokenizer(dialect=dialect)
    return tokenizer.tokenize(text)

def detect_dialect_from_tokens(tokens: List[str]) -> str:
    """Detect dialect based on tokens
    
    Args:
        tokens: List of tokens
        
    Returns:
        Detected dialect code
    """
    # Convert tokens to space-separated text for detection
    text = " ".join(tokens)
    dialect_scores = detect_dialect(text)
    return max(dialect_scores.items(), key=lambda x: x[1])[0]