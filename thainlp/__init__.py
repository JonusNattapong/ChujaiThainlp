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

# Classification models
from .models.classification import TokenClassifier

# Question answering
from .models.qa import (
    TextQA,
    TableQA,
)

# Text generation
from .models.generation import (
    TextGenerator,
    FillMask,
)

# Translation
from .models.translation import Translator

# Summarization
from .models.summarization import Summarizer

# Text similarity
from .models.similarity import SentenceSimilarity

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

def get_pipeline(**kwargs) -> ThaiNLPPipeline:
    """Get or create default pipeline instance
    
    Args:
        **kwargs: Arguments to pass to pipeline constructor
        
    Returns:
        ThaiNLPPipeline instance
    """
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = ThaiNLPPipeline(**kwargs)
    return _default_pipeline

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
    
    # Models
    "TokenClassifier",
    "TextQA",
    "TableQA", 
    "TextGenerator",
    "FillMask",
    "Translator",
    "Summarizer",
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
