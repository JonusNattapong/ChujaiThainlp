"""
Thai tokenization utilities
"""
from typing import List
import re
from .maximum_matching import MaximumMatchingTokenizer

# THAI_WORDS definition removed to break circular import.
# It should be imported from thainlp.resources if needed elsewhere,
# or maximum_matching should import directly from resources.

def word_tokenize(text: str) -> List[str]:
    """Tokenize Thai text using maximum matching algorithm"""
    tokenizer = MaximumMatchingTokenizer()
    return tokenizer.tokenize(text)

class Tokenizer:
    """Base tokenizer class"""
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError
