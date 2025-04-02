"""
Thai Natural Language Processing library
"""
from typing import List
from .tokenization import word_tokenize
from .utils.thai_utils import (
    is_thai_char,
    is_thai_word,
    remove_tone_marks,
    remove_diacritics,
    normalize_text,
    count_thai_words,
    extract_thai_text,
    thai_to_roman,
    detect_language,
    thai_number_to_text,
    thai_digit_to_arabic_digit,
    arabic_digit_to_thai_digit
)
from .tag import pos_tag
from .thai_spell_correction import ThaiSpellChecker
from .sentiment.lexicon_based import LexiconSentimentAnalyzer

# Initialize default analyzer
_sentiment_analyzer = LexiconSentimentAnalyzer()

def sentiment(text: str) -> float:
    """Get sentiment score for Thai text (-1 to 1)"""
    return _sentiment_analyzer.analyze(text)

__version__ = "1.0.0"
__all__ = [
    'word_tokenize',
    'is_thai_char',
    'is_thai_word',
    'remove_tone_marks',
    'remove_diacritics',
    'normalize_text',
    'count_thai_words',
    'extract_thai_text',
    'thai_to_roman',
    'detect_language',
    'thai_number_to_text',
    'thai_digit_to_arabic_digit',
    'arabic_digit_to_thai_digit',
    'pos_tag',
    'ThaiSpellChecker',
    'sentiment'
]
