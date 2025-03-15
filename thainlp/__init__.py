"""
ThaiNLP: Thai Natural Language Processing Library
"""

# Version
__version__ = "0.2.0"

# Import core functionality that exists
from thainlp.sentiment.lexicon_based import LexiconSentimentAnalyzer
from thainlp.tokenize import thai_word_tokenize

# Check if PyThaiNLP is available
try:
    import pythainlp
    from pythainlp.tokenize import word_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    word_tokenize = None

# Initialize components
_sentiment_analyzer = LexiconSentimentAnalyzer()

from thainlp.utils.thai_utils import detect_language
from thainlp.util import normalize_text, count_thai_chars
from thainlp.ner.rule_based import extract_entities
from thainlp.question_answering.qa_system import answer_question, answer_from_table
from thainlp.translation.translator import ThaiTranslator
from thainlp.feature_extraction.feature_extractor import ThaiFeatureExtractor
from thainlp.generation.text_generator import ThaiTextGenerator
from thainlp.similarity.text_similarity import calculate_similarity, find_most_similar, is_duplicate
from thainlp.utils.thai_utils import (
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

# Core functionality defined directly in __init__.py
def classify_text(text: str, category: str = "general") -> dict:
    """Classify Thai text"""
    if category == "sentiment":
        return _sentiment_analyzer.analyze(text)
    return {"category": "unknown", "confidence": 0.0}

def zero_shot_classification(text: str, labels: list) -> dict:
    """Zero-shot classification"""
    result = {}
    text_lower = text.lower()
    for label in labels:
        result[label] = 1.0 if label.lower() in text_lower else 0.0
    return result

def classify_tokens(tokens: list, task: str = "pos") -> list:
    """Classify tokens"""
    return [(token, "NOUN") for token in tokens]

def tokenize(text: str) -> list:
    """Tokenize text (using PyThaiNLP if available, otherwise native implementation)."""
    if PYTHAINLP_AVAILABLE:
        return word_tokenize(text)  # Use PyThaiNLP's tokenizer
    else:
        return thai_word_tokenize(text) # Use native implementation


# Make functions available at package level
__all__ = [
    'classify_text',
    'zero_shot_classification',
    'classify_tokens',
    'tokenize',
    'extract_entities',
    'answer_question',
    'answer_from_table',
    'translate_text',
    'detect_language_translation',
    'extract_features',
    'generate_text',
    'calculate_similarity',
    'find_most_similar',
    'is_duplicate',
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
    'count_thai_chars'
]

_translator = None  # Global variable to hold the translator instance
_feature_extractor = None
_text_generator = None

def translate_text(text: str, source_lang: str = "th", target_lang: str = "en", **kwargs) -> dict:
    """Translate text using the initialized translator."""
    global _translator
    if _translator is None:
        _translator = ThaiTranslator()  # Initialize only when needed
    return _translator.translate(text, source_lang=source_lang, target_lang=target_lang, **kwargs)

def detect_language_translation(text: str) -> str:
    """Detect the language of the text using Thai-specific utils."""
    return detect_language(text)

def extract_features(text: str, **kwargs) -> dict:
    """Extract features from text using the initialized feature extractor."""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = ThaiFeatureExtractor()
    return _feature_extractor.extract_features(text, **kwargs)

def generate_text(method: str = "template", **kwargs) -> str:
    """Generate text using the initialized text generator."""
    global _text_generator
    if _text_generator is None:
        _text_generator = ThaiTextGenerator()
    if method == "template":
        if "template_type" not in kwargs:
            raise ValueError("template_type must be specified for template-based generation.")
        # Placeholder logic for template-based generation.
        # In a real implementation, this would use a template engine.
        template_type = kwargs["template_type"]
        if template_type == "greeting":
            return "สวัสดีครับ/ค่ะ"
        elif template_type == "farewell":
            return "ลาก่อนครับ/ค่ะ"
        else:
            return "ข้อความตัวอย่าง"  # Default text
    elif method == "pattern":
        if "pattern" not in kwargs:
            raise ValueError("pattern must be specified for pattern-based generation")
        pattern = kwargs["pattern"]
        return _text_generator.generate(prompt=" ".join(pattern)) # Simplified
    elif method == "paragraph":
        if "num_sentences" not in kwargs:
            raise ValueError("num_sentences must be specified for paragraph generation")
        return _text_generator.generate(prompt="", max_length=kwargs['num_sentences'] * 20) # Heuristic
    else:
        raise ValueError(f"Unknown generation method: {method}")

# Add functions to module namespace
globals().update({name: globals()[name] for name in __all__})
