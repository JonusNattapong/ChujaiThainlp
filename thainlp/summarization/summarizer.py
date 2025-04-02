"""
Thai text summarization
"""
from typing import List, Dict, Set, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from ..tokenization import word_tokenize

def split_sentences(text: str) -> List[str]:
    """Split Thai text into sentences using basic rules"""
    sentences = []
    current = []
    
    for char in text:
        current.append(char)
        if char in {'。', '！', '？', '।', '។', '။', '၏', '?', '!', '.', '\n'}:
            if current:
                sentences.append(''.join(current).strip())
                current = []
                
    if current:
        sentences.append(''.join(current).strip())
        
    return [s for s in sentences if s]

class ThaiSummarizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=word_tokenize,
            stop_words=[]  # Thai stopwords handled by tokenizer
        )
        
    def summarize(
        self,
        text: str,
        ratio: float = 0.3,
        min_length: int = 40,
        max_length: int = 600
    ) -> str:
        """Generate extractive summary of Thai text
        
        Args:
            text: Input text
            ratio: Target summary ratio (0.0-1.0)
            min_length: Minimum summary length
            max_length: Maximum summary length
            
        Returns:
            Extractive summary
        """
        # Split into sentences
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return text
            
        # Convert to TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores using similarity matrix
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        scores = self._score_sentences(similarity_matrix)
        
        # Select top sentences
        num_sentences = max(1, int(len(sentences) * ratio))
        selected_indices = np.argsort(scores)[-num_sentences:]
        selected_indices.sort()  # Maintain original order
        
        # Build summary
        summary_sentences = [sentences[i] for i in selected_indices]
        summary = ' '.join(summary_sentences)
        
        # Apply length constraints
        if len(summary) < min_length and len(text) > min_length:
            return text[:max_length]
        elif len(summary) > max_length:
            return summary[:max_length]
            
        return summary
        
    def _score_sentences(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Score sentences based on similarity matrix using TextRank"""
        # Constants for TextRank
        damping = 0.85
        epsilon = 1e-8
        max_iter = 100
        
        n_sentences = len(similarity_matrix)
        
        # Normalize similarity matrix
        norm = similarity_matrix.sum(axis=1, keepdims=True)
        norm[norm == 0] = 1  # Avoid division by zero
        transition_matrix = similarity_matrix / norm
        
        # Initialize scores
        scores = np.ones(n_sentences) / n_sentences
        
        # Power iteration
        for _ in range(max_iter):
            prev_scores = scores
            scores = (1 - damping) + damping * (transition_matrix.T @ scores)
            
            # Check convergence
            if np.abs(scores - prev_scores).sum() < epsilon:
                break
                
        return scores

    def keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text using TF-IDF scores
        
        Args:
            text: Input text
            top_k: Number of keywords to return
            
        Returns:
            List of keywords
        """
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Convert to TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top scoring words
        scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, score in sorted_words[:top_k]]