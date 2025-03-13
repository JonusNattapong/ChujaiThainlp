"""
Thai Natural Language Processing Library
"""

from .core import (
    word_tokenize,
    sentence_tokenize,
    normalize,
    is_thai,
    count_thai_chars,
    remove_thai_vowels,
    is_palindrome
)

from .transliteration import (
    thai_to_roman,
    roman_to_thai
)

from .sentiment import analyze_sentiment
from .pos_tag import pos_tag
from .ner import find_entities
from .summarize import summarize
from .spellcheck import check_spelling

__version__ = "0.1.0"
__all__ = [
    # Core functions
    "word_tokenize",
    "sentence_tokenize",
    "normalize",
    "is_thai",
    "count_thai_chars",
    "remove_thai_vowels",
    "is_palindrome",
    
    # Transliteration functions
    "thai_to_roman",
    "roman_to_thai",
    
    # Advanced NLP functions
    "analyze_sentiment",
    "pos_tag",
    "find_entities",
    "summarize",
    "check_spelling"
]