"""
ThaiNLP: Thai Natural Language Processing Library
"""

__version__ = "0.2.0"

# Import tokenization
from thainlp.tokenize import word_tokenize
tokenize = word_tokenize  # Main tokenization function

# Import POS tagging
from thainlp.tag import pos_tag

# Import NER
from thainlp.extensions.advanced_nlp import ThaiNamedEntityRecognition
extract_entities = ThaiNamedEntityRecognition().extract_entities

# Import sentiment analysis
from thainlp.extensions.advanced_nlp import ThaiSentimentAnalyzer
analyze_sentiment = ThaiSentimentAnalyzer().analyze_sentiment

# Import spell checking
check_spelling = spell_correction  # Using existing spell_correction from thai_utils

# Import text summarization
from thainlp.extensions.advanced_nlp import ThaiTextGenerator
summarize_text = ThaiTextGenerator().summarize

# Import text generation
from thainlp.extensions.advanced_nlp import ThaiTextGenerator
text_generator = ThaiTextGenerator()
generate_text = text_generator.generate_text
generate_paragraph = text_generator.generate_text  # Using same function with different params
complete_text = text_generator.generate_text  # Using same function with different params

# Import text similarity
from thainlp.extensions.advanced_nlp import ThaiTextAnalyzer
text_analyzer = ThaiTextAnalyzer()
calculate_similarity = text_analyzer.semantic_similarity
find_most_similar = text_analyzer.semantic_similarity  # Using same function
is_duplicate = text_analyzer.semantic_similarity  # Using same function with threshold

# Import advanced features
from thainlp.extensions.advanced_nlp import (
    TopicModeling,
    EmotionDetector,
    AdvancedThaiNLP
)

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

# Define __all__ for wildcard imports
__all__ = [
    # Basic NLP
    'tokenize',
    'pos_tag',
    'extract_entities',
    'analyze_sentiment',
    'check_spelling',
    'summarize_text',
    
    # Advanced NLP
    'TopicModeling',
    'EmotionDetector',
    'AdvancedThaiNLP',
    'ThaiTextAnalyzer',
    'ThaiTextGenerator',
    
    # Text Generation
    'generate_text',
    'generate_paragraph',
    'complete_text',
    
    # Text Similarity
    'calculate_similarity',
    'find_most_similar',
    'is_duplicate',
    
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
    
    # Advanced Thai Utils
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
    'create_document_vector'
]
