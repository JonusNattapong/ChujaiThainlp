"""
Lexicon-based Sentiment Analysis for Thai Text
"""

from typing import List, Dict, Tuple
from collections import defaultdict

class ThaiSentimentAnalyzer:
    def __init__(self):
        """Initialize ThaiSentimentAnalyzer with sentiment dictionaries"""
        self.positive_words = {
            'ดี', 'สวย', 'เยี่ยม', 'ยอดเยี่ยม', 'สุดยอด', 'ดีมาก',
            'น่ารัก', 'สวยงาม', 'น่าชื่นชม', 'น่าชื่นใจ', 'น่าภูมิใจ',
            'สนุก', 'เพลิดเพลิน', 'น่าสนใจ', 'น่าตื่นเต้น',
            'อร่อย', 'หอม', 'สด', 'สะอาด', 'สดใส',
            'รัก', 'ชอบ', 'ชื่นชอบ', 'ชื่นชม', 'ชื่นใจ',
            'สุข', 'สุขใจ', 'สบาย', 'สบายใจ', 'สดชื่น',
            'เก่ง', 'ฉลาด', 'เฉลียวฉลาด', 'มีปัญญา',
            'ขอบคุณ', 'ขอบใจ', 'ขอบพระคุณ',
        }
        
        self.negative_words = {
            'แย่', 'ไม่ดี', 'เลว', 'ชั่ว', 'โหดร้าย',
            'น่าเกลียด', 'น่าขยะแขยง', 'น่าอับอาย',
            'น่าเบื่อ', 'น่าเหงา', 'น่าเศร้า', 'น่าหดหู่',
            'โกรธ', 'โมโห', 'หงุดหงิด', 'รำคาญ',
            'เจ็บ', 'ปวด', 'เมื่อย', 'เหนื่อย', 'อ่อนเพลีย',
            'กลัว', 'หวาดกลัว', 'หวาดหวั่น', 'กังวล',
            'เสีย', 'หาย', 'ขาด', 'หายไป',
            'ผิด', 'ผิดพลาด', 'ผิดปกติ',
        }
        
        self.negation_words = {
            'ไม่', 'ไม่มี', 'ไม่ได้', 'อย่า', 'ห้าม',
            'ไม่มีทาง', 'ไม่มีโอกาส', 'ไม่มีสิทธิ์',
        }
        
        self.intensifier_words = {
            'มาก', 'มากๆ', 'มากมาย', 'ที่สุด',
            'เกินไป', 'เกิน', 'มากเกินไป',
            'แทบ', 'เกือบ', 'ใกล้เคียง',
        }
        
        self.diminisher_words = {
            'น้อย', 'น้อยๆ', 'นิดหน่อย',
            'เล็กน้อย', 'เล็กๆ', 'นิดเดียว',
        }
        
    def _is_thai(self, char: str) -> bool:
        """Check if character is Thai"""
        return '\u0E00' <= char <= '\u0E7F'
        
    def _is_thai_word(self, word: str) -> bool:
        """Check if word contains Thai characters"""
        return any(self._is_thai(char) for char in word)
        
    def _get_word_sentiment(self, word: str) -> float:
        """
        Get sentiment score for a word
        
        Args:
            word (str): Input word
            
        Returns:
            float: Sentiment score (-1.0 to 1.0)
        """
        if word in self.positive_words:
            return 1.0
        elif word in self.negative_words:
            return -1.0
        return 0.0
        
    def _apply_modifiers(self, words: List[str], start_idx: int) -> float:
        """
        Apply negation and intensifier/diminisher modifiers
        
        Args:
            words (List[str]): List of words
            start_idx (int): Starting index
            
        Returns:
            float: Modified sentiment score
        """
        score = 0.0
        i = start_idx
        
        while i < len(words):
            word = words[i]
            
            # Check for negation
            if word in self.negation_words:
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    score = -self._get_word_sentiment(next_word)
                    i += 2
                    continue
                    
            # Check for intensifier
            if word in self.intensifier_words:
                if i > 0:
                    prev_word = words[i - 1]
                    score = self._get_word_sentiment(prev_word) * 1.5
                    
            # Check for diminisher
            if word in self.diminisher_words:
                if i > 0:
                    prev_word = words[i - 1]
                    score = self._get_word_sentiment(prev_word) * 0.5
                    
            # Get base sentiment
            if self._is_thai_word(word):
                score = self._get_word_sentiment(word)
                
            i += 1
            
        return score
        
    def analyze(self, text: str) -> Tuple[float, str, Dict[str, List[str]]]:
        """
        Analyze sentiment of Thai text
        
        Args:
            text (str): Input text
            
        Returns:
            Tuple[float, str, Dict[str, List[str]]]: (score, label, categorized_words)
        """
        # Split text into words
        words = text.split()
        
        # Initialize result
        score = 0.0
        categorized_words = {
            'positive': [],
            'negative': [],
            'neutral': []
        }
        
        # Process each word
        i = 0
        while i < len(words):
            word = words[i]
            
            if self._is_thai_word(word):
                # Get sentiment with modifiers
                word_score = self._apply_modifiers(words, i)
                score += word_score
                
                # Categorize word
                if word_score > 0:
                    categorized_words['positive'].append(word)
                elif word_score < 0:
                    categorized_words['negative'].append(word)
                else:
                    categorized_words['neutral'].append(word)
                    
            i += 1
            
        # Normalize score
        if len(categorized_words['positive']) + len(categorized_words['negative']) > 0:
            score = score / (len(categorized_words['positive']) + len(categorized_words['negative']))
            
        # Determine label
        if score > 0.5:
            label = 'very_positive'
        elif score > 0:
            label = 'positive'
        elif score < -0.5:
            label = 'very_negative'
        elif score < 0:
            label = 'negative'
        else:
            label = 'neutral'
            
        return score, label, categorized_words

def analyze_sentiment(text: str) -> Tuple[float, str, Dict[str, List[str]]]:
    """
    Analyze sentiment of Thai text
    
    Args:
        text (str): Input text
        
    Returns:
        Tuple[float, str, Dict[str, List[str]]]: (score, label, categorized_words)
    """
    analyzer = ThaiSentimentAnalyzer()
    return analyzer.analyze(text) 