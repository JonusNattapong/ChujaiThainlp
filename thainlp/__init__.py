"""
ThaiNLP: Thai Natural Language Processing Library
"""

__version__ = "0.2.0"

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
    detect_language,
    # เพิ่มฟังก์ชันขั้นสูงจาก PyThaiNLP
    thai_number_to_text,
    thai_text_to_number,
    format_thai_date,
    thai_soundex,
    spell_correction,
    get_thai_stopwords,
    get_thai_syllables,
    get_thai_wordnet_synsets,
    get_thai_wordnet_synonyms,
    thai_digit_to_arabic_digit,
    arabic_digit_to_thai_digit,
    thai_time,
    thai_day_to_datetime
)

# Import classification
from thainlp.classification.text_classifier import (
    classify_text,
    zero_shot_classification
)
from thainlp.classification.token_classifier import (
    classify_tokens,
    find_entities
)

# Import question answering
from thainlp.question_answering.qa_system import (
    answer_question,
    answer_from_table
)

# Import translation
from thainlp.translation.translator import (
    translate_text,
    detect_language as detect_language_translation
)

# Import feature extraction
from thainlp.feature_extraction.feature_extractor import (
    extract_features,
    extract_advanced_features,
    create_document_vector
)

# Import text generation
from thainlp.generation.text_generator import (
    generate_text,
    generate_paragraph,
    complete_text
)

# Import text similarity
from thainlp.similarity.text_similarity import (
    calculate_similarity,
    find_most_similar,
    is_duplicate
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
    'detect_language',
    
    # ฟังก์ชันขั้นสูงจาก PyThaiNLP
    'thai_number_to_text',
    'thai_text_to_number',
    'format_thai_date',
    'thai_soundex',
    'spell_correction',
    'get_thai_stopwords',
    'get_thai_syllables',
    'get_thai_wordnet_synsets',
    'get_thai_wordnet_synonyms',
    'thai_digit_to_arabic_digit',
    'arabic_digit_to_thai_digit',
    'thai_time',
    'thai_day_to_datetime',
    
    # Classification
    'classify_text',
    'zero_shot_classification',
    'classify_tokens',
    'find_entities',
    
    # Question Answering
    'answer_question',
    'answer_from_table',
    
    # Translation
    'translate_text',
    'detect_language_translation',
    
    # Feature Extraction
    'extract_features',
    'extract_advanced_features',
    'create_document_vector',
    
    # Text Generation
    'generate_text',
    'generate_paragraph',
    'complete_text',
    
    # Text Similarity
    'calculate_similarity',
    'find_most_similar',
    'is_duplicate'
]