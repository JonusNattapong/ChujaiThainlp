"""
TextRank Algorithm for Thai Text Summarization
"""

from typing import List, Dict, Tuple, Set
import numpy as np
import re
from collections import defaultdict

class ThaiTextRank:
    def __init__(self):
        """Initialize ThaiTextRank with stopwords"""
        # Thai stopwords
        self.stopwords = {
            'ที่', 'และ', 'ใน', 'ของ', 'ให้', 'ได้', 'ไป', 'มา', 'เป็น', 'กับ',
            'แต่', 'หรือ', 'เมื่อ', 'โดย', 'จาก', 'ถ้า', 'แล้ว', 'จะ', 'ต้อง', 'ยัง',
            'อยู่', 'คือ', 'ว่า', 'ซึ่ง', 'ตาม', 'นี้', 'นั้น', 'อีก', 'ทั้ง', 'เพราะ',
            'เพื่อ', 'มี', 'ก็', 'คง', 'ควร', 'อาจ', 'ช่วย', 'ทำให้', 'ทำ', 'เคย',
            'ต่อ', 'จน', 'เพียง', 'พอ', 'ถึง', 'อย่าง', 'เช่น', 'เนื่องจาก', 'ด้วย', 'ครับ',
            'ค่ะ', 'นะ', 'ครับผม', 'ดิฉัน', 'ผม', 'ฉัน', 'เธอ', 'คุณ', 'เรา', 'พวกเรา',
            'พวกเขา', 'พวกมัน', 'มัน', 'เขา', 'เอง', 'ทุก', 'บาง', 'หลาย', 'ส่วน', 'ขณะ',
        }
        
    def _is_thai(self, char: str) -> bool:
        """Check if character is Thai"""
        return '\u0E00' <= char <= '\u0E7F'
        
    def _is_thai_word(self, word: str) -> bool:
        """Check if word contains Thai characters"""
        return any(self._is_thai(char) for char in word)
        
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        # Split by common Thai sentence delimiters
        sentences = re.split(r'[.!?।॥\n]+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def _preprocess(self, sentence: str) -> List[str]:
        """
        Preprocess sentence by removing stopwords
        
        Args:
            sentence (str): Input sentence
            
        Returns:
            List[str]: List of words
        """
        words = sentence.split()
        return [word for word in words if word.lower() not in self.stopwords]
        
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Build similarity matrix between sentences
        
        Args:
            sentences (List[str]): List of sentences
            
        Returns:
            np.ndarray: Similarity matrix
        """
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        # Preprocess sentences
        preprocessed_sentences = [set(self._preprocess(s)) for s in sentences]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate similarity using word overlap
                    words_i = preprocessed_sentences[i]
                    words_j = preprocessed_sentences[j]
                    
                    if not words_i or not words_j:
                        continue
                        
                    # Jaccard similarity
                    intersection = len(words_i.intersection(words_j))
                    union = len(words_i.union(words_j))
                    
                    if union > 0:
                        similarity_matrix[i][j] = intersection / union
                        
        return similarity_matrix
        
    def _pagerank(self, similarity_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """
        Apply PageRank algorithm to similarity matrix
        
        Args:
            similarity_matrix (np.ndarray): Similarity matrix
            damping (float): Damping factor
            max_iter (int): Maximum iterations
            tol (float): Convergence tolerance
            
        Returns:
            np.ndarray: PageRank scores
        """
        n = similarity_matrix.shape[0]
        
        # Normalize similarity matrix
        for i in range(n):
            row_sum = similarity_matrix[i].sum()
            if row_sum > 0:
                similarity_matrix[i] = similarity_matrix[i] / row_sum
                
        # Initialize scores
        scores = np.ones(n) / n
        
        # PageRank iterations
        for _ in range(max_iter):
            prev_scores = scores.copy()
            
            for i in range(n):
                scores[i] = (1 - damping) / n + damping * np.sum(similarity_matrix[:, i] * prev_scores)
                
            # Check convergence
            if np.abs(scores - prev_scores).sum() < tol:
                break
                
        return scores
        
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Summarize text using TextRank algorithm
        
        Args:
            text (str): Input text
            num_sentences (int): Number of sentences in summary
            
        Returns:
            str: Summarized text
        """
        # Split text into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= num_sentences:
            return text
            
        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(sentences)
        
        # Apply PageRank
        scores = self._pagerank(similarity_matrix)
        
        # Get top sentences
        ranked_sentences = [(score, i, sentence) for i, (score, sentence) in enumerate(zip(scores, sentences))]
        ranked_sentences.sort(reverse=True)
        
        # Select top sentences and sort by original order
        top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])
        
        # Join sentences
        summary = ' '.join([sentence for _, _, sentence in top_sentences])
        
        return summary

def summarize_text(text: str, num_sentences: int = 3) -> str:
    """
    Summarize text using TextRank algorithm
    
    Args:
        text (str): Input text
        num_sentences (int): Number of sentences in summary
        
    Returns:
        str: Summarized text
    """
    summarizer = ThaiTextRank()
    return summarizer.summarize(text, num_sentences) 