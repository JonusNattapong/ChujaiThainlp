"""
ChujaiThainlp - Advanced Thai Natural Language Processing Library
"""

__version__ = "2.0.0"

# Core tokenization
from .tokenization import (
    word_tokenize,
    encode,
    decode,
    get_tokenizer,
    get_thai_tokenizer,
    ThaiTokenizer,
    MaximumMatchingTokenizer,
)

# Named Entity Recognition
from .ner import (
    ThaiNamedEntityRecognition,
    tag as ner_tag
)

# Sentiment Analysis
from .sentiment import (
    ThaiSentimentAnalyzer,
    analyze as analyze_sentiment
)

# Question answering
from .qa import (
    ThaiQuestionAnswering,
    TableQuestionAnswering,
    answer_question,
)

# Text generation
from .generation import (
    ThaiTextGenerator,
    TextGenerator,
    FillMask,
)

# Translation
from .translation import Translator

# Summarization
from .summarization import Summarizer

# Text similarity
from .similarity import (
    ThaiTextAnalyzer,
    SentenceSimilarity
)

# Unified pipeline
from .pipelines import ThaiNLPPipeline

# Utilities
from .utils.thai_utils import (
    normalize_text,
    clean_thai_text,
    contains_thai,
    separate_thai_english,
    detect_language_mix,
)

from .utils.monitoring import (
    ProgressTracker,
    ResourceMonitor,
)

# Default instances for convenience
_default_pipeline = None
_default_ner = None
_default_sentiment = None
_default_qa = None
_default_generator = None

def get_pipeline(**kwargs) -> ThaiNLPPipeline:
    """Get or create default pipeline instance"""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = ThaiNLPPipeline(**kwargs)
    return _default_pipeline

# Convenient high-level functions
def get_entities(text: str) -> list:
    """Extract named entities from Thai text"""
    global _default_ner
    if _default_ner is None:
        _default_ner = ThaiNamedEntityRecognition()
    return _default_ner.extract_entities(text)

def get_sentiment(text: str) -> dict:
    """Analyze sentiment of Thai text"""
    global _default_sentiment
    if _default_sentiment is None:
        _default_sentiment = ThaiSentimentAnalyzer()
    return _default_sentiment.analyze_sentiment(text)

def ask(question: str, context: str) -> dict:
    """Get answer from context"""
    global _default_qa
    if _default_qa is None:
        _default_qa = ThaiQuestionAnswering()
    return _default_qa.answer_question(question, context)

def generate(prompt: str, **kwargs) -> str:
    """Generate Thai text from prompt"""
    global _default_generator
    if _default_generator is None:
        _default_generator = ThaiTextGenerator()
    return _default_generator.generate_text(prompt, **kwargs)

# Package info
__author__ = "ThaiNLP"
__email__ = "contact@thainlp.org"
__url__ = "https://github.com/thainlp/ChujaiThainlp"
__license__ = "MIT"

__all__ = [
    # Version
    "__version__",
    
    # Tokenization
    "word_tokenize",
    "encode", 
    "decode",
    "get_tokenizer",
    "get_thai_tokenizer",
    "ThaiTokenizer",
    "MaximumMatchingTokenizer",
    
    # NER
    "ThaiNamedEntityRecognition",
    "ner_tag",
    "get_entities",
    
    # Sentiment
    "ThaiSentimentAnalyzer",
    "analyze_sentiment",
    "get_sentiment",
    
    # Question Answering
    "ThaiQuestionAnswering",
    "TableQuestionAnswering",
    "answer_question",
    "ask",
    
    # Generation
    "ThaiTextGenerator",
    "TextGenerator",
    "FillMask",
    "generate",
    
    # Translation & Summarization
    "Translator",
    "Summarizer",
    
    # Similarity
    "ThaiTextAnalyzer",
    "SentenceSimilarity",
    
    # Pipeline
    "ThaiNLPPipeline",
    "get_pipeline",
    
    # Utils
    "normalize_text",
    "clean_thai_text", 
    "contains_thai",
    "separate_thai_english",
    "detect_language_mix",
    "ProgressTracker",
    "ResourceMonitor",
]
