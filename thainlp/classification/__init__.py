"""
Classification functionality for Thai text.
"""

from thainlp.classification.text_classifier import ThaiTextClassifier
from thainlp.sentiment.lexicon_based import LexiconSentimentAnalyzer

# Initialize default classifiers
_text_classifier = ThaiTextClassifier()
_sentiment_analyzer = LexiconSentimentAnalyzer()

__all__ = ['classify_text', 'zero_shot_classification', 'classify_tokens']

# Function definitions at module level
def classify_text(text: str, category: str = "general") -> dict:
    """
    Classify Thai text using various classification models.

    Args:
        text: Thai text to classify
        category: Classification category ('sentiment', 'topic', etc.)

    Returns:
        Classification result (label or detailed dict)
    """
    if category == "sentiment":
        result = _sentiment_analyzer.analyze(text)
        return result
    else:
        result = _text_classifier.classify(text)
        return result

def zero_shot_classification(text: str, labels: list) -> dict:
    """
    Perform zero-shot classification on Thai text.

    Args:
        text: Input text to classify
        labels: List of possible labels

    Returns:
        Dictionary mapping labels to confidence scores
    """
    return _text_classifier.zero_shot_classify(text, labels)

def classify_tokens(tokens: list, task: str = "pos") -> list:
    """
    Classify individual tokens (e.g., POS tagging).

    Args:
        tokens: List of tokens to classify
        task: Classification task ('pos', 'ner', etc.)

    Returns:
        List of (token, label) tuples
    """
    # For POS tagging, use text classifier with special template
    if task == "pos":
        results = _text_classifier.classify(tokens)
        return list(zip(tokens, [result["pos"] for result in results]))
    
    return [(token, "O") for token in tokens]  # Default fallback

globals().update({
    'classify_text': classify_text,
    'zero_shot_classification': zero_shot_classification,
    'classify_tokens': classify_tokens
})
