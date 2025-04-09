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

# Define ThaiSentimentAnalyzer class
class ThaiSentimentAnalyzer:
    """
    Thai Sentiment Analyzer class for analyzing sentiment in Thai text.
    Wrapper around transformer-based and lexicon-based models.
    """
    
    def __init__(self, model_name=DEFAULT_TRANSFORMER_MODEL, use_lexicon=False):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: Name of the transformer model to use
            use_lexicon: Whether to use lexicon-based analyzer instead of transformer
        """
        self.model_name = model_name
        self.use_lexicon = use_lexicon
        
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text
        
        Args:
            text: Thai text to analyze
            
        Returns:
            Dictionary with sentiment score and label
        """
        if self.use_lexicon:
            return analyze_sentiment_lexicon(text)
        else:
            return analyze_transformer(text, model_name=self.model_name)
            
    def analyze(self, text):
        """Alias for analyze_sentiment"""
        return self.analyze_sentiment(text)

# Define what gets imported with 'from thainlp.sentiment import *'
__all__ = [
    'ThaiSentimentAnalyzer',
    'analyze_sentiment_lexicon',
    'LexiconSentimentAnalyzer',
    'analyze_transformer',
    'DEFAULT_TRANSFORMER_MODEL',
]

# Function that selects the model based on an argument
def analyze(text: str, model: str = DEFAULT_TRANSFORMER_MODEL) -> dict:
    """
    Analyze sentiment of text
    
    Args:
        text: Thai text to analyze
        model: Model name or 'lexicon'
        
    Returns:
        Dictionary with sentiment score and label
    """
    if model == "lexicon":
        return analyze_sentiment_lexicon(text)
    else:
        return analyze_transformer(text, model_name=model)
