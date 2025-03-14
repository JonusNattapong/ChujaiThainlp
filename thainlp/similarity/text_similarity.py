"""
Thai Text Similarity Module
"""

from typing import Dict, List, Union, Optional, Any, Tuple, Set
import re
from collections import Counter
import math

class ThaiTextSimilarity:
    def __init__(self):
        """Initialize ThaiTextSimilarity"""
        # Thai stopwords (common words that don't contribute much to meaning)
        self.stopwords = [
            'และ', 'หรือ', 'แต่', 'จะ', 'ที่', 'ของ', 'ใน', 'มี', 'เป็น', 'การ',
            'ไม่', 'ได้', 'ให้', 'มา', 'ไป', 'กับ', 'ว่า', 'นี้', 'อยู่', 'คน',
            'เรา', 'เขา', 'คุณ', 'ฉัน', 'ผม', 'ดี', 'ต้อง', 'เมื่อ', 'ถ้า', 'แล้ว',
            'ก็', 'จาก', 'โดย', 'ด้วย', 'อีก', 'ถึง', 'เพื่อ', 'ต่อ', 'จน', 'เพราะ',
            'ทำ', 'ความ', 'อย่าง', 'เคย', 'ตาม', 'แบบ', 'ทุก', 'ช่วย', 'ระหว่าง', 'นั้น'
        ]
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for Thai text
        
        Args:
            text (str): Thai text to tokenize
            
        Returns:
            List[str]: List of Thai tokens
        """
        # This is a simplified tokenization
        # In a real system, use a proper Thai tokenizer
        # For now, we'll just split by whitespace and treat each character as a token
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char
                
        if current_token:
            tokens.append(current_token)
            
        return tokens
        
    def _preprocess(self, text: str) -> List[str]:
        """
        Preprocess text for similarity comparison
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: Preprocessed tokens
        """
        # Tokenize
        tokens = self._tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stopwords]
        
        # Remove punctuation and convert to lowercase
        tokens = [re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9]', '', token).lower() for token in tokens]
        
        # Remove empty tokens
        tokens = [token for token in tokens if token]
        
        return tokens
        
    def _get_character_ngrams(self, text: str, n: int = 3) -> List[str]:
        """
        Get character n-grams from text
        
        Args:
            text (str): Input text
            n (int): n-gram size
            
        Returns:
            List[str]: List of character n-grams
        """
        # Remove spaces
        text = re.sub(r'\s+', '', text)
        
        # Generate n-grams
        ngrams = []
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            ngrams.append(ngram)
            
        return ngrams
        
    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Jaccard similarity score (0-1)
        """
        # Preprocess texts
        tokens1 = set(self._preprocess(text1))
        tokens2 = set(self._preprocess(text2))
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
        
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        # Preprocess texts
        tokens1 = self._preprocess(text1)
        tokens2 = self._preprocess(text2)
        
        # Count token frequencies
        vec1 = Counter(tokens1)
        vec2 = Counter(tokens2)
        
        # Find common tokens
        common_tokens = set(vec1.keys()).intersection(set(vec2.keys()))
        
        # Calculate dot product
        dot_product = sum(vec1[token] * vec2[token] for token in common_tokens)
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(count ** 2 for count in vec1.values()))
        magnitude2 = math.sqrt(sum(count ** 2 for count in vec2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)
        
    def ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """
        Calculate n-gram similarity between two texts
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            n (int): n-gram size
            
        Returns:
            float: n-gram similarity score (0-1)
        """
        # Get character n-grams
        ngrams1 = set(self._get_character_ngrams(text1, n))
        ngrams2 = set(self._get_character_ngrams(text2, n))
        
        # Calculate Jaccard similarity of n-grams
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
        
    def levenshtein_distance(self, text1: str, text2: str) -> int:
        """
        Calculate Levenshtein distance between two texts
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            int: Levenshtein distance
        """
        # Remove spaces
        text1 = re.sub(r'\s+', '', text1)
        text2 = re.sub(r'\s+', '', text2)
        
        # Create distance matrix
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if text1[i-1] == text2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
                
        return dp[m][n]
        
    def levenshtein_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Levenshtein similarity between two texts
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Levenshtein similarity score (0-1)
        """
        # Calculate Levenshtein distance
        distance = self.levenshtein_distance(text1, text2)
        
        # Calculate maximum possible distance
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 1.0
            
        # Convert distance to similarity
        return 1.0 - (distance / max_len)
        
    def longest_common_subsequence(self, text1: str, text2: str) -> int:
        """
        Calculate length of longest common subsequence between two texts
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            int: Length of longest common subsequence
        """
        # Remove spaces
        text1 = re.sub(r'\s+', '', text1)
        text2 = re.sub(r'\s+', '', text2)
        
        # Create LCS matrix
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        return dp[m][n]
        
    def lcs_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity based on longest common subsequence
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: LCS similarity score (0-1)
        """
        # Calculate LCS length
        lcs_length = self.longest_common_subsequence(text1, text2)
        
        # Calculate maximum possible length
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 1.0
            
        # Convert length to similarity
        return lcs_length / max_len
        
    def calculate_similarity(self, text1: str, text2: str, method: str = 'combined') -> Dict[str, float]:
        """
        Calculate similarity between two texts using specified method
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            method (str): Similarity method ('jaccard', 'cosine', 'ngram', 'levenshtein', 'lcs', or 'combined')
            
        Returns:
            Dict[str, float]: Similarity scores
        """
        if method == 'jaccard':
            return {'jaccard': self.jaccard_similarity(text1, text2)}
        elif method == 'cosine':
            return {'cosine': self.cosine_similarity(text1, text2)}
        elif method == 'ngram':
            return {'ngram': self.ngram_similarity(text1, text2)}
        elif method == 'levenshtein':
            return {'levenshtein': self.levenshtein_similarity(text1, text2)}
        elif method == 'lcs':
            return {'lcs': self.lcs_similarity(text1, text2)}
        elif method == 'combined':
            # Calculate all similarity measures
            jaccard = self.jaccard_similarity(text1, text2)
            cosine = self.cosine_similarity(text1, text2)
            ngram = self.ngram_similarity(text1, text2)
            levenshtein = self.levenshtein_similarity(text1, text2)
            lcs = self.lcs_similarity(text1, text2)
            
            # Calculate combined score (weighted average)
            combined = (jaccard * 0.2 + cosine * 0.3 + ngram * 0.2 + levenshtein * 0.15 + lcs * 0.15)
            
            return {
                'jaccard': jaccard,
                'cosine': cosine,
                'ngram': ngram,
                'levenshtein': levenshtein,
                'lcs': lcs,
                'combined': combined
            }
        else:
            return {'error': f"Unknown similarity method: {method}"}
            
    def find_most_similar(self, query: str, texts: List[str], method: str = 'combined') -> List[Tuple[int, float]]:
        """
        Find most similar texts to a query
        
        Args:
            query (str): Query text
            texts (List[str]): List of texts to compare against
            method (str): Similarity method
            
        Returns:
            List[Tuple[int, float]]: List of (index, similarity_score) tuples, sorted by similarity
        """
        similarities = []
        
        for i, text in enumerate(texts):
            # Calculate similarity
            result = self.calculate_similarity(query, text, method)
            
            # Get the relevant score
            if method == 'combined':
                score = result['combined']
            else:
                score = result.get(method, 0.0)
                
            similarities.append((i, score))
            
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
        
    def is_duplicate(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Check if two texts are duplicates
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            threshold (float): Similarity threshold for considering as duplicate
            
        Returns:
            bool: True if texts are duplicates, False otherwise
        """
        # Calculate combined similarity
        result = self.calculate_similarity(text1, text2, 'combined')
        
        # Check if combined score exceeds threshold
        return result['combined'] >= threshold

def calculate_similarity(text1: str, text2: str, method: str = 'combined') -> Dict[str, float]:
    """
    Calculate similarity between two Thai texts
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        method (str): Similarity method ('jaccard', 'cosine', 'ngram', 'levenshtein', 'lcs', or 'combined')
        
    Returns:
        Dict[str, float]: Similarity scores
    """
    similarity = ThaiTextSimilarity()
    return similarity.calculate_similarity(text1, text2, method)

def find_most_similar(query: str, texts: List[str], method: str = 'combined') -> List[Tuple[int, float]]:
    """
    Find most similar texts to a query
    
    Args:
        query (str): Query text
        texts (List[str]): List of texts to compare against
        method (str): Similarity method
        
    Returns:
        List[Tuple[int, float]]: List of (index, similarity_score) tuples, sorted by similarity
    """
    similarity = ThaiTextSimilarity()
    return similarity.find_most_similar(query, texts, method)

def is_duplicate(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """
    Check if two texts are duplicates
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        threshold (float): Similarity threshold for considering as duplicate
        
    Returns:
        bool: True if texts are duplicates, False otherwise
    """
    similarity = ThaiTextSimilarity()
    return similarity.is_duplicate(text1, text2, threshold) 