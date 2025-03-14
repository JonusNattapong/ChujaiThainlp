"""
Context-aware text analysis for ThaiNLP.
"""
from typing import Dict, List, Any
import re
from datetime import datetime
from thainlp.extensions.advanced_nlp import ThaiSentimentAnalyzer
from transformers import pipeline

class ContextAnalyzer:
    def __init__(self):
        """Initialize Context Analyzer."""
        self.context_patterns = {
            'social_media': {
                'hashtags': r'#\w+',
                'mentions': r'@\w+',
                'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            },
            'email': {
                'email_address': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                'subject': r'Subject:.*',
                'date': r'Date:.*'
            },
            'legal': {
                'case_numbers': r'[A-Z]{2,3}\s*\d{1,4}/\d{4}',
                'dates': r'\d{1,2}\s+[มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม]\s+\d{4}'
            }
        }
        self.sentiment_analyzer = ThaiSentimentAnalyzer()
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    def analyze_social_media(self, text: str) -> Dict[str, Any]:
        """
        Analyze social media text.
        
        Args:
            text: Input text from social media
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'hashtags': re.findall(self.context_patterns['social_media']['hashtags'], text),
            'mentions': re.findall(self.context_patterns['social_media']['mentions'], text),
            'urls': re.findall(self.context_patterns['social_media']['urls'], text),
            'sentiment': self._analyze_sentiment(text),
            'engagement_metrics': self._calculate_engagement_metrics(text)
        }
        return results
    
    def analyze_email(self, text: str) -> Dict[str, Any]:
        """
        Analyze email content.
        
        Args:
            text: Input email text
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'email_addresses': re.findall(self.context_patterns['email']['email_address'], text),
            'subject': re.findall(self.context_patterns['email']['subject'], text),
            'date': re.findall(self.context_patterns['email']['date'], text),
            'priority': self._analyze_email_priority(text),
            'spam_score': self._calculate_spam_score(text)
        }
        return results
    
    def analyze_legal_document(self, text: str) -> Dict[str, Any]:
        """
        Analyze legal document content.
        
        Args:
            text: Input legal document text
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'case_numbers': re.findall(self.context_patterns['legal']['case_numbers'], text),
            'dates': re.findall(self.context_patterns['legal']['dates'], text),
            'document_type': self._identify_document_type(text),
            'key_entities': self._extract_legal_entities(text),
            'summary': self._generate_legal_summary(text)
        }
        return results
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text using zero-shot classification."""
        labels = ["positive", "negative", "neutral"]
        result = self.zero_shot_classifier(text, labels, multi_label=False)
        sentiment = result['labels'][0]
        score = result['scores'][0]
        return f"{sentiment} (score: {score:.2f})"
    
    def _calculate_engagement_metrics(self, text: str) -> Dict[str, float]:
        """Calculate engagement metrics."""
        # Placeholder implementation - สามารถปรับปรุงได้ในอนาคต
        words = text.split()
        num_words = len(words)
        num_hashtags = len(re.findall(self.context_patterns['social_media']['hashtags'], text))
        num_mentions = len(re.findall(self.context_patterns['social_media']['mentions'], text))
        num_urls = len(re.findall(self.context_patterns['social_media']['urls'], text))
        
        engagement_score = num_words + (num_hashtags * 2) + (num_mentions * 2) + (num_urls * 3)
        
        return {
            'words': num_words,
            'hashtags': num_hashtags,
            'mentions': num_mentions,
            'urls': num_urls,
            'engagement_score': engagement_score
        }
    
    def _analyze_email_priority(self, text: str) -> str:
        """Analyze email priority."""
        # Placeholder implementation - สามารถปรับปรุงได้ในอนาคต
        if "ด่วน" in text or "สำคัญ" in text:
            return "High"
        elif "สอบถาม" in text or "แจ้ง" in text:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_spam_score(self, text: str) -> float:
        """Calculate spam score."""
        # Placeholder implementation - สามารถปรับปรุงได้ในอนาคต
        spam_keywords = ["ฟรี", "รางวัล", "โปรโมชั่น", "ลดราคา", "คลิกที่นี่"]
        score = 0
        for keyword in spam_keywords:
            if keyword in text:
                score += 0.1
        
        email_patterns = self.context_patterns['email']
        if not re.findall(email_patterns['email_address'], text):
            score += 0.2
        
        if len(text) < 20: # อีเมลสแปมมักสั้น
            score += 0.1
            
        return min(score, 1.0) # คะแนนสแปมไม่เกิน 1.0
    
    def _identify_document_type(self, text: str) -> str:
        """Identify legal document type."""
        # Placeholder implementation - สามารถปรับปรุงได้ในอนาคต
        if "สัญญา" in text:
            return "Contract"
        elif "ข้อตกลง" in text:
            return "Agreement"
        elif "คำฟ้อง" in text or "คำร้อง" in text:
            return "Legal Document"
        else:
            return "Unknown Legal Document"
    
    def _extract_legal_entities(self, text: str) -> List[str]:
        """Extract legal entities."""
        # Placeholder implementation - สามารถปรับปรุงได้ในอนาคต
        # ในความเป็นจริงควรใช้ NER model ที่ train มาสำหรับ legal entities
        legal_entities_keywords = ["บริษัท", "จำกัด", "ห้างหุ้นส่วน", "นาย", "นาง", "นางสาว"]
        entities = []
        words = text.split()
        for i in range(len(words)):
            for keyword in legal_entities_keywords:
                if words[i] == keyword and i + 1 < len(words):
                    entities.append(f"{keyword} {words[i+1]}")
        return entities
    
    def _generate_legal_summary(self, text: str) -> str:
        """Generate legal document summary."""
        # Implement legal summary generation
        pass 