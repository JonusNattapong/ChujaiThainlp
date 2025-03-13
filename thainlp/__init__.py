"""
ThaiNLP: Thai Natural Language Processing Library
"""

__version__ = "0.1.0"

# Import tokenization
from thainlp.tokenization.maximum_matching import tokenize

# Import POS tagging
from thainlp.pos_tagging.hmm_tagger import train_and_tag as pos_tag

# Import NER
from thainlp.ner.rule_based import extract_entities

# Import sentiment analysis
from thainlp.sentiment.lexicon_based import analyze_sentiment

# Import spell checking
from thainlp.spellcheck.edit_distance import check_spelling

# Import text summarization
from thainlp.summarization.textrank import summarize_text

# Import utilities
from thainlp.utils.thai_utils import (
    is_thai_char,
    is_thai_word,
    remove_tone_marks,
    remove_diacritics,
    normalize_text,
    count_thai_words,
    extract_thai_text,
    thai_to_roman,
    detect_language
)

# Define __all__ for wildcard imports
__all__ = [
    # Tokenization
    'tokenize',
    
    # POS tagging
    'pos_tag',
    
    # NER
    'extract_entities',
    
    # Sentiment analysis
    'analyze_sentiment',
    
    # Spell checking
    'check_spelling',
    
    # Text summarization
    'summarize_text',
    
    # Utilities
    'is_thai_char',
    'is_thai_word',
    'remove_tone_marks',
    'remove_diacritics',
    'normalize_text',
    'count_thai_words',
    'extract_thai_text',
    'thai_to_roman',
    'detect_language'
]