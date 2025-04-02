"""
Sentiment analysis for Thai text.

Provides access to both lexicon-based and transformer-based models.
"""
# Import from lexicon-based module
from .lexicon_based import analyze_sentiment as analyze_sentiment_lexicon
from .lexicon_based import LexiconSentimentAnalyzer

# Import from transformer module
from .transformer_sentiment import analyze as analyze_transformer
from .transformer_sentiment import DEFAULT_MODEL as DEFAULT_TRANSFORMER_MODEL

# Define what gets imported with 'from thainlp.sentiment import *'
__all__ = [
    'analyze_sentiment_lexicon',
    'LexiconSentimentAnalyzer',
    'analyze_transformer',
    'DEFAULT_TRANSFORMER_MODEL',
]

# Similar to NER, a top-level function could be added later
# to select the model type via an argument.
