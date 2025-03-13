"""
Sentiment analysis for Thai text.
"""
from typing import Dict, Any
from thainlp.tokenize import word_tokenize

def analyze_sentiment(text: str, engine: str = "dict") -> Dict[str, Any]:
    """
    Analyze sentiment in Thai text.
    
    Args:
        text: Thai text to analyze
        engine: Engine for sentiment analysis ('dict' for dictionary-based)
        
    Returns:
        Dictionary with sentiment analysis results
    """
    # This is a very simplified implementation
    # In a real system, you would use ML models
    
    # Simple dictionary of sentiment words
    pos_words = {"ดี", "สุข", "รัก", "ชอบ", "สวย", "เยี่ยม"}
    neg_words = {"แย่", "เศร้า", "เกลียด", "โกรธ", "ผิด", "เสียใจ"}
    
    tokens = word_tokenize(text)
    
    pos_count = sum(1 for token in tokens if token in pos_words)
    neg_count = sum(1 for token in tokens if token in neg_words)
    
    if pos_count > neg_count:
        sentiment = "positive"
        score = min(1.0, pos_count / len(tokens) * 2)
    elif neg_count > pos_count:
        sentiment = "negative"
        score = min(1.0, neg_count / len(tokens) * 2)
    else:
        sentiment = "neutral"
        score = 0.5
    
    return {
        "sentiment": sentiment,
        "score": score,
        "positive_words": [t for t in tokens if t in pos_words],
        "negative_words": [t for t in tokens if t in neg_words],
    }