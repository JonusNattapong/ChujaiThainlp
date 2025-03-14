"""
Thai Text Feature Extraction Module
"""

from typing import Dict, List, Union, Optional, Any
import re
from collections import Counter
import math

class ThaiFeatureExtractor:
    def __init__(self):
        """Initialize ThaiFeatureExtractor"""
        # Thai vowels
        self.vowels = [
            'ะ', 'ั', 'า', 'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู',
            'เ', 'แ', 'โ', 'ใ', 'ไ', '็', '่', '้', '๊', '๋',
            '์', 'ํ', 'ๆ', 'ฯ'
        ]
        
        # Thai consonants
        self.consonants = [
            'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช',
            'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด',
            'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ',
            'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส',
            'ห', 'ฬ', 'อ', 'ฮ'
        ]
        
        # Thai tone marks
        self.tone_marks = ['่', '้', '๊', '๋']
        
        # Thai digits
        self.thai_digits = ['๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙']
        
        # Common Thai words (simplified list)
        self.common_words = [
            'และ', 'หรือ', 'แต่', 'จะ', 'ที่', 'ของ', 'ใน', 'มี', 'เป็น', 'การ',
            'ไม่', 'ได้', 'ให้', 'มา', 'ไป', 'กับ', 'ว่า', 'นี้', 'อยู่', 'คน',
            'เรา', 'เขา', 'คุณ', 'ฉัน', 'ผม', 'ดี', 'ต้อง', 'เมื่อ', 'ถ้า', 'แล้ว'
        ]
        
    def _count_char_types(self, text: str) -> Dict[str, int]:
        """
        Count different character types in text
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, int]: Counts of different character types
        """
        counts = {
            'consonants': 0,
            'vowels': 0,
            'tone_marks': 0,
            'thai_digits': 0,
            'arabic_digits': 0,
            'spaces': 0,
            'english_chars': 0,
            'special_chars': 0,
            'total_chars': len(text)
        }
        
        for char in text:
            if char in self.consonants:
                counts['consonants'] += 1
            elif char in self.vowels:
                counts['vowels'] += 1
            elif char in self.tone_marks:
                counts['tone_marks'] += 1
            elif char in self.thai_digits:
                counts['thai_digits'] += 1
            elif char.isdigit():
                counts['arabic_digits'] += 1
            elif char.isspace():
                counts['spaces'] += 1
            elif 'a' <= char.lower() <= 'z':
                counts['english_chars'] += 1
            else:
                counts['special_chars'] += 1
                
        return counts
        
    def _calculate_char_ratios(self, counts: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate character type ratios
        
        Args:
            counts (Dict[str, int]): Character type counts
            
        Returns:
            Dict[str, float]: Character type ratios
        """
        total = counts['total_chars']
        if total == 0:
            return {k + '_ratio': 0.0 for k in counts if k != 'total_chars'}
            
        return {
            k + '_ratio': v / total 
            for k, v in counts.items() 
            if k != 'total_chars'
        }
        
    def _count_word_frequencies(self, tokens: List[str]) -> Dict[str, int]:
        """
        Count word frequencies
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            Dict[str, int]: Word frequencies
        """
        return dict(Counter(tokens))
        
    def _calculate_tf_idf(self, tokens: List[str], document_freq: Dict[str, int], total_docs: int) -> Dict[str, float]:
        """
        Calculate TF-IDF scores for tokens
        
        Args:
            tokens (List[str]): List of tokens
            document_freq (Dict[str, int]): Document frequencies
            total_docs (int): Total number of documents
            
        Returns:
            Dict[str, float]: TF-IDF scores
        """
        # Count term frequencies
        term_freq = Counter(tokens)
        
        # Calculate TF-IDF
        tf_idf = {}
        for term, freq in term_freq.items():
            # Term frequency
            tf = freq / len(tokens) if len(tokens) > 0 else 0
            
            # Inverse document frequency
            doc_freq = document_freq.get(term, 1)  # Avoid division by zero
            idf = math.log(total_docs / doc_freq) if doc_freq > 0 else 0
            
            # TF-IDF
            tf_idf[term] = tf * idf
            
        return tf_idf
        
    def _extract_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """
        Extract n-grams from tokens
        
        Args:
            tokens (List[str]): List of tokens
            n (int): n-gram size
            
        Returns:
            List[str]: List of n-grams
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
            
        return ngrams
        
    def _calculate_text_statistics(self, text: str, tokens: List[str]) -> Dict[str, Union[int, float]]:
        """
        Calculate text statistics
        
        Args:
            text (str): Input text
            tokens (List[str]): List of tokens
            
        Returns:
            Dict[str, Union[int, float]]: Text statistics
        """
        # Count sentences (simplified)
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s for s in sentences if s.strip()]
        
        # Calculate statistics
        stats = {
            'char_count': len(text),
            'token_count': len(tokens),
            'sentence_count': len(sentences),
            'avg_token_length': sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
            'avg_sentence_length': len(tokens) / len(sentences) if sentences else 0,
            'unique_token_ratio': len(set(tokens)) / len(tokens) if tokens else 0
        }
        
        return stats
        
    def _extract_pos_patterns(self, pos_tags: List[tuple]) -> Dict[str, int]:
        """
        Extract part-of-speech patterns
        
        Args:
            pos_tags (List[tuple]): List of (token, pos_tag) tuples
            
        Returns:
            Dict[str, int]: POS pattern frequencies
        """
        patterns = []
        
        # Extract bigram patterns
        for i in range(len(pos_tags) - 1):
            pattern = f"{pos_tags[i][1]}_{pos_tags[i+1][1]}"
            patterns.append(pattern)
            
        return dict(Counter(patterns))
        
    def extract_basic_features(self, text: str, tokens: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract basic features from Thai text
        
        Args:
            text (str): Input text
            tokens (Optional[List[str]]): List of tokens (if None, text is treated as a single token)
            
        Returns:
            Dict[str, Any]: Basic features
        """
        if tokens is None:
            tokens = [text]
            
        # Character type counts and ratios
        char_counts = self._count_char_types(text)
        char_ratios = self._calculate_char_ratios(char_counts)
        
        # Word frequencies
        word_freq = self._count_word_frequencies(tokens)
        
        # Text statistics
        text_stats = self._calculate_text_statistics(text, tokens)
        
        # Combine features
        features = {
            **char_counts,
            **char_ratios,
            'word_frequencies': word_freq,
            **text_stats
        }
        
        return features
        
    def extract_advanced_features(self, text: str, tokens: List[str], pos_tags: Optional[List[tuple]] = None, 
                                document_freq: Optional[Dict[str, int]] = None, total_docs: int = 1) -> Dict[str, Any]:
        """
        Extract advanced features from Thai text
        
        Args:
            text (str): Input text
            tokens (List[str]): List of tokens
            pos_tags (Optional[List[tuple]]): List of (token, pos_tag) tuples
            document_freq (Optional[Dict[str, int]]): Document frequencies for TF-IDF
            total_docs (int): Total number of documents for TF-IDF
            
        Returns:
            Dict[str, Any]: Advanced features
        """
        # Get basic features
        features = self.extract_basic_features(text, tokens)
        
        # Extract n-grams
        bigrams = self._extract_ngrams(tokens, 2)
        trigrams = self._extract_ngrams(tokens, 3)
        
        features['bigrams'] = dict(Counter(bigrams))
        features['trigrams'] = dict(Counter(trigrams))
        
        # Calculate TF-IDF if document frequencies are provided
        if document_freq is not None:
            features['tf_idf'] = self._calculate_tf_idf(tokens, document_freq, total_docs)
            
        # Extract POS patterns if POS tags are provided
        if pos_tags is not None:
            features['pos_patterns'] = self._extract_pos_patterns(pos_tags)
            
        # Common word ratio
        common_word_count = sum(1 for token in tokens if token in self.common_words)
        features['common_word_ratio'] = common_word_count / len(tokens) if tokens else 0
        
        return features
        
    def create_document_vector(self, features: Dict[str, Any], vector_size: int = 100) -> List[float]:
        """
        Create a fixed-size vector from features (simplified)
        
        Args:
            features (Dict[str, Any]): Features dictionary
            vector_size (int): Size of the output vector
            
        Returns:
            List[float]: Document vector
        """
        # This is a simplified implementation
        # In a real system, use proper dimensionality reduction or embedding techniques
        
        # Extract numeric features
        numeric_features = []
        
        # Add character counts and ratios
        for key, value in features.items():
            if isinstance(value, (int, float)) and key != 'total_chars':
                numeric_features.append(value)
                
        # Add word frequency statistics
        if 'word_frequencies' in features:
            word_freq = features['word_frequencies']
            if word_freq:
                numeric_features.append(len(word_freq))  # Vocabulary size
                numeric_features.append(sum(word_freq.values()) / len(word_freq))  # Average frequency
                numeric_features.append(max(word_freq.values()))  # Max frequency
                
        # Add TF-IDF statistics if available
        if 'tf_idf' in features and features['tf_idf']:
            tf_idf = features['tf_idf']
            numeric_features.append(sum(tf_idf.values()) / len(tf_idf))  # Average TF-IDF
            numeric_features.append(max(tf_idf.values()))  # Max TF-IDF
            
        # Pad or truncate to vector_size
        if len(numeric_features) > vector_size:
            return numeric_features[:vector_size]
        else:
            return numeric_features + [0.0] * (vector_size - len(numeric_features))

def extract_features(text: str, tokens: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extract features from Thai text
    
    Args:
        text (str): Input text
        tokens (Optional[List[str]]): List of tokens (if None, text is treated as a single token)
        
    Returns:
        Dict[str, Any]: Extracted features
    """
    extractor = ThaiFeatureExtractor()
    return extractor.extract_basic_features(text, tokens)

def extract_advanced_features(text: str, tokens: List[str], pos_tags: Optional[List[tuple]] = None) -> Dict[str, Any]:
    """
    Extract advanced features from Thai text
    
    Args:
        text (str): Input text
        tokens (List[str]): List of tokens
        pos_tags (Optional[List[tuple]]): List of (token, pos_tag) tuples
        
    Returns:
        Dict[str, Any]: Advanced features
    """
    extractor = ThaiFeatureExtractor()
    return extractor.extract_advanced_features(text, tokens, pos_tags)

def create_document_vector(features: Dict[str, Any], vector_size: int = 100) -> List[float]:
    """
    Create a fixed-size vector from features
    
    Args:
        features (Dict[str, Any]): Features dictionary
        vector_size (int): Size of the output vector
        
    Returns:
        List[float]: Document vector
    """
    extractor = ThaiFeatureExtractor()
    return extractor.create_document_vector(features, vector_size) 