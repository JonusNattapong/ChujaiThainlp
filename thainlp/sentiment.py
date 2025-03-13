"""
Thai sentiment analysis functionality
"""

from typing import Dict, List, Tuple
from .resources import THAI_SENTIMENT_DICT

# Thai negation words
THAI_NEGATION_WORDS: List[str] = [
    'ไม่', 'มิ', 'ไม่มี', 'มิได้', 'ไม่ได้', 'อย่า', 'ห้าม'
]

# Thai intensifier words
THAI_INTENSIFIER_WORDS: Dict[str, float] = {
    'มาก': 1.5, 'มากๆ': 2.0, 'ที่สุด': 2.0,
    'มากมาย': 1.5, 'เหลือเกิน': 1.5, 'เกินไป': 1.5,
    'นิด': 0.5, 'น้อย': 0.5, 'เล็กน้อย': 0.5
}

def analyze_sentiment(text: str) -> Tuple[float, str, Dict[str, List[str]]]:
    """
    Analyze sentiment of Thai text with enhanced accuracy.
    
    Args:
        text (str): Thai text to analyze
        
    Returns:
        Tuple[float, str, Dict[str, List[str]]]: (sentiment score, sentiment label, categorized words)
    """
    words = text.split()
    score = 0.0
    negation = False
    intensifier = 1.0
    
    # Track categorized words
    categorized_words = {
        'positive': [],
        'negative': [],
        'neutral': []
    }
    
    for i, word in enumerate(words):
        # Check for negation
        if word in THAI_NEGATION_WORDS:
            negation = True
            continue
            
        # Check for intensifier
        if word in THAI_INTENSIFIER_WORDS:
            intensifier = THAI_INTENSIFIER_WORDS[word]
            continue
            
        # Get word sentiment
        if word in THAI_SENTIMENT_DICT:
            word_score = THAI_SENTIMENT_DICT[word]
            if negation:
                word_score = -word_score
            word_score *= intensifier
            score += word_score
            
            # Categorize word
            if word_score > 0:
                categorized_words['positive'].append(word)
            elif word_score < 0:
                categorized_words['negative'].append(word)
            else:
                categorized_words['neutral'].append(word)
                
            negation = False
            intensifier = 1.0
    
    # Normalize score
    if len(words) > 0:
        score = score / len(words)
    
    # Determine label with more granular thresholds
    if score > 0.3:
        label = "very_positive"
    elif score > 0.1:
        label = "positive"
    elif score < -0.3:
        label = "very_negative"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"
        
    return score, label, categorized_words