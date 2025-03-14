"""
Text Classification for Thai Text
"""

from typing import List, Dict, Tuple, Union, Optional
import numpy as np
from collections import Counter

class ThaiTextClassifier:
    def __init__(self):
        """Initialize ThaiTextClassifier"""
        # Predefined categories for common classification tasks
        self.categories = {
            'sentiment': ['positive', 'neutral', 'negative'],
            'topic': ['politics', 'sports', 'entertainment', 'technology', 'business', 'health'],
            'spam': ['spam', 'not_spam'],
            'intent': ['question', 'statement', 'command', 'request']
        }
        
        # Simple keyword-based classification
        self.keywords = {
            'sentiment': {
                'positive': [
                    'ดี', 'สวย', 'เยี่ยม', 'ยอดเยี่ยม', 'สุดยอด', 'ชอบ', 'รัก', 'ประทับใจ',
                    'พอใจ', 'สนุก', 'สุข', 'สบาย', 'อร่อย', 'น่ารัก', 'เก่ง'
                ],
                'negative': [
                    'แย่', 'ไม่ดี', 'เลว', 'แห้ว', 'เสีย', 'เสียใจ', 'ผิดหวัง', 'โกรธ',
                    'เกลียด', 'น่าเบื่อ', 'น่ารำคาญ', 'น่ากลัว', 'กลัว', 'เจ็บ', 'ปวด'
                ]
            },
            'topic': {
                'politics': [
                    'รัฐบาล', 'นายกรัฐมนตรี', 'รัฐมนตรี', 'การเมือง', 'พรรค', 'เลือกตั้ง',
                    'ประชาธิปไตย', 'นโยบาย', 'กฎหมาย', 'รัฐสภา', 'ส.ส.', 'ส.ว.'
                ],
                'sports': [
                    'กีฬา', 'ฟุตบอล', 'บอล', 'บาสเกตบอล', 'วอลเลย์บอล', 'เทนนิส', 'กอล์ฟ',
                    'แข่งขัน', 'นักกีฬา', 'โอลิมปิก', 'เหรียญ', 'ชนะ', 'แพ้', 'เสมอ'
                ],
                'entertainment': [
                    'ดารา', 'นักแสดง', 'นักร้อง', 'เพลง', 'ภาพยนตร์', 'หนัง', 'ละคร', 'ซีรีส์',
                    'คอนเสิร์ต', 'บันเทิง', 'ศิลปิน', 'อัลบั้ม', 'เพลง', 'ฮิต'
                ],
                'technology': [
                    'เทคโนโลยี', 'คอมพิวเตอร์', 'โทรศัพท์', 'มือถือ', 'แอพ', 'แอปพลิเคชัน',
                    'อินเทอร์เน็ต', 'ดิจิทัล', 'ซอฟต์แวร์', 'ฮาร์ดแวร์', 'โค้ด', 'โปรแกรม'
                ],
                'business': [
                    'ธุรกิจ', 'การเงิน', 'เศรษฐกิจ', 'ตลาด', 'หุ้น', 'บริษัท', 'ลงทุน',
                    'กำไร', 'ขาดทุน', 'ราคา', 'ต้นทุน', 'รายได้', 'ค่าใช้จ่าย', 'ภาษี'
                ],
                'health': [
                    'สุขภาพ', 'โรค', 'แพทย์', 'หมอ', 'โรงพยาบาล', 'ยา', 'รักษา', 'ป่วย',
                    'อาการ', 'ผู้ป่วย', 'วัคซีน', 'ไวรัส', 'เชื้อ', 'สุขภาพจิต', 'ออกกำลังกาย'
                ]
            },
            'spam': {
                'spam': [
                    'ฟรี', 'โปรโมชั่น', 'ลด', 'แลก', 'แจก', 'ด่วน', 'พิเศษ', 'โอกาสสุดท้าย',
                    'รวย', 'เงิน', 'กำไร', 'ลงทุน', 'รับประกัน', 'ไม่ต้องลงทุน', 'รายได้'
                ]
            },
            'intent': {
                'question': [
                    'ไหม', 'หรือไม่', 'หรือเปล่า', 'ใช่ไหม', 'ทำไม', 'อย่างไร', 'เมื่อไร',
                    'ที่ไหน', 'อะไร', 'ใคร', 'กี่', 'เท่าไร', '?', 'เหรอ', 'หรอ'
                ],
                'command': [
                    'จง', 'ต้อง', 'ควร', 'ให้', 'ทำ', 'อย่า', 'ห้าม', 'กรุณา', 'โปรด',
                    'ช่วย', 'เชิญ', 'เร่ง', 'รีบ', '!'
                ],
                'request': [
                    'ขอ', 'อยาก', 'ต้องการ', 'ช่วย', 'กรุณา', 'โปรด', 'รบกวน', 'ขอร้อง',
                    'ขอความกรุณา', 'ขอบคุณ'
                ]
            }
        }
        
        # Feature weights for each category
        self.weights = {
            'sentiment': {'keyword': 0.7, 'length': 0.1, 'punctuation': 0.2},
            'topic': {'keyword': 0.9, 'length': 0.05, 'punctuation': 0.05},
            'spam': {'keyword': 0.6, 'length': 0.2, 'punctuation': 0.2},
            'intent': {'keyword': 0.5, 'length': 0.2, 'punctuation': 0.3}
        }
        
    def _count_keywords(self, text: str, category_type: str) -> Dict[str, int]:
        """
        Count keywords for each class in the category
        
        Args:
            text (str): Input text
            category_type (str): Category type (sentiment, topic, etc.)
            
        Returns:
            Dict[str, int]: Counts of keywords for each class
        """
        words = text.lower().split()
        counts = {cls: 0 for cls in self.categories[category_type]}
        
        for cls, keywords in self.keywords.get(category_type, {}).items():
            for word in words:
                if word in keywords:
                    counts[cls] = counts.get(cls, 0) + 1
                    
        return counts
        
    def _extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract features from text
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Features
        """
        features = {}
        
        # Text length
        features['length'] = len(text)
        
        # Punctuation count
        features['punctuation'] = sum(1 for c in text if c in '.,!?;:')
        
        # Uppercase ratio (for non-Thai text)
        if any(c.isalpha() and not ('\u0E00' <= c <= '\u0E7F') for c in text):
            uppercase_chars = sum(1 for c in text if c.isupper())
            total_chars = sum(1 for c in text if c.isalpha())
            features['uppercase_ratio'] = uppercase_chars / total_chars if total_chars > 0 else 0
            
        return features
        
    def classify(self, text: str, category_type: str = 'sentiment') -> Tuple[str, Dict[str, float]]:
        """
        Classify text into predefined categories
        
        Args:
            text (str): Input text
            category_type (str): Category type (sentiment, topic, spam, intent)
            
        Returns:
            Tuple[str, Dict[str, float]]: (predicted class, confidence scores)
        """
        if category_type not in self.categories:
            raise ValueError(f"Category type '{category_type}' not supported. Available types: {list(self.categories.keys())}")
            
        # Count keywords
        keyword_counts = self._count_keywords(text, category_type)
        
        # Extract other features
        features = self._extract_features(text)
        
        # Calculate scores
        scores = {}
        for cls in self.categories[category_type]:
            # Keyword score
            keyword_score = keyword_counts.get(cls, 0)
            
            # Length score (normalized)
            length_score = 0
            if category_type == 'spam' and features['length'] > 100:
                length_score = 0.5  # Longer texts are more likely to be spam
            elif category_type == 'intent':
                if cls == 'question' and features['length'] < 50:
                    length_score = 0.3  # Questions tend to be shorter
                elif cls == 'command' and features['length'] < 30:
                    length_score = 0.3  # Commands tend to be shorter
                    
            # Punctuation score
            punctuation_score = 0
            if category_type == 'intent':
                if cls == 'question' and '?' in text:
                    punctuation_score = 0.8
                elif cls == 'command' and '!' in text:
                    punctuation_score = 0.8
                    
            # Combine scores with weights
            weights = self.weights.get(category_type, {'keyword': 0.7, 'length': 0.15, 'punctuation': 0.15})
            scores[cls] = (
                keyword_score * weights['keyword'] +
                length_score * weights['length'] +
                punctuation_score * weights['punctuation']
            )
            
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {cls: score / total_score for cls, score in scores.items()}
            
        # Get predicted class
        predicted_class = max(scores.items(), key=lambda x: x[1])[0]
        
        return predicted_class, scores
        
    def zero_shot_classify(self, text: str, candidate_labels: List[str]) -> Tuple[List[str], List[float]]:
        """
        Zero-shot classification with custom labels
        
        Args:
            text (str): Input text
            candidate_labels (List[str]): List of candidate labels
            
        Returns:
            Tuple[List[str], List[float]]: (sorted labels, scores)
        """
        words = text.lower().split()
        scores = {}
        
        # Simple keyword matching for each label
        for label in candidate_labels:
            # Convert label to keywords (simple approach)
            keywords = label.lower().split('_')
            
            # Count occurrences
            count = 0
            for keyword in keywords:
                count += sum(1 for word in words if keyword in word)
                
            scores[label] = count
            
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {label: score / total for label, score in scores.items()}
            
        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        sorted_labels = [label for label, _ in sorted_results]
        sorted_scores = [score for _, score in sorted_results]
        
        return sorted_labels, sorted_scores

def classify_text(text: str, category_type: str = 'sentiment') -> Tuple[str, Dict[str, float]]:
    """
    Classify text into predefined categories
    
    Args:
        text (str): Input text
        category_type (str): Category type (sentiment, topic, spam, intent)
        
    Returns:
        Tuple[str, Dict[str, float]]: (predicted class, confidence scores)
    """
    classifier = ThaiTextClassifier()
    return classifier.classify(text, category_type)

def zero_shot_classification(text: str, candidate_labels: List[str]) -> Tuple[List[str], List[float]]:
    """
    Zero-shot classification with custom labels
    
    Args:
        text (str): Input text
        candidate_labels (List[str]): List of candidate labels
        
    Returns:
        Tuple[List[str], List[float]]: (sorted labels, scores)
    """
    classifier = ThaiTextClassifier()
    return classifier.zero_shot_classify(text, candidate_labels) 