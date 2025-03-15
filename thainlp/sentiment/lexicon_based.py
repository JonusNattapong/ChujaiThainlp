"""
Lexicon-based sentiment analysis for Thai text.
"""
from typing import Dict, Set, Union, Optional
import os
import json

# Default sentiment lexicons
DEFAULT_POSITIVE_WORDS = {
    'ดี', 'เยี่ยม', 'สุข', 'รัก', 'ชอบ', 'สนุก', 'ชนะ',
    'ยอด', 'เก่ง', 'สวย', 'งาม', 'ประทับใจ', 'ยิ้ม',
    'สำเร็จ', 'พอใจ', 'สบาย', 'สดใส', 'มั่นใจ'
}

DEFAULT_NEGATIVE_WORDS = {
    'แย่', 'เสีย', 'เศร้า', 'โกรธ', 'เกลียด', 'ทุกข์', 'แพ้',
    'ผิด', 'กลัว', 'กังวล', 'เจ็บ', 'ปวด', 'ร้องไห้',
    'ล้มเหลว', 'ผิดหวัง', 'เหนื่อย', 'ท้อ', 'สิ้นหวัง'
}

class LexiconSentimentAnalyzer:
    """Lexicon-based sentiment analyzer for Thai text"""

    def __init__(
        self,
        lexicon_path: Optional[str] = None,
        positive_words: Optional[Set[str]] = None,
        negative_words: Optional[Set[str]] = None
    ):
        """Initialize sentiment analyzer
        
        Args:
            lexicon_path: Path to lexicon JSON file
            positive_words: Set of positive words
            negative_words: Set of negative words
        """
        self.positive_words = positive_words or DEFAULT_POSITIVE_WORDS.copy()
        self.negative_words = negative_words or DEFAULT_NEGATIVE_WORDS.copy()
        
        # Load custom lexicon if provided
        if lexicon_path and os.path.exists(lexicon_path):
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                lexicon = json.load(f)
                if 'positive' in lexicon:
                    self.positive_words.update(lexicon['positive'])
                if 'negative' in lexicon:
                    self.negative_words.update(lexicon['negative'])
                
    def analyze(self, text: str) -> Dict[str, Union[float, str]]:
        """Analyze sentiment of text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment score and polarity
        """
        # Tokenize into words (simple space-based for now)
        words = text.split()
        
        # Count sentiment words
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        total = pos_count + neg_count
        
        # Calculate sentiment score
        if total > 0:
            score = (pos_count - neg_count) / total
        else:
            score = 0.0
            
        # Determine polarity
        if score > 0.1:
            polarity = 'positive'
        elif score < -0.1:
            polarity = 'negative'
        else:
            polarity = 'neutral'
            
        return {
            'score': score,
            'polarity': polarity,
            'positive_words': pos_count,
            'negative_words': neg_count
        }
        
# Create default analyzer instance
_default_analyzer = LexiconSentimentAnalyzer()

def analyze_sentiment(text: str) -> float:
    """Analyze sentiment of text using default analyzer
    
    Args:
        text: Input text
        
    Returns:
        Sentiment score between -1 (negative) and 1 (positive)
    """
    return _default_analyzer.analyze(text)['score']
