"""
Thai sentence similarity calculation
"""
from typing import List, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ..tokenization import word_tokenize

class SentenceSimilarity:
    def __init__(self, model_name: str = "distiluse-base-multilingual-cased-v2"):
        """Initialize with specified sentence transformer model"""
        self.model = SentenceTransformer(model_name)
        
    def get_similarity(self, text1: str, text2: str, method: str = 'transformer') -> float:
        """Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method ('transformer', 'tfidf', or 'token')
            
        Returns:
            Similarity score between 0 and 1
        """
        if method == 'transformer':
            return self._transformer_similarity(text1, text2)
        elif method == 'tfidf':
            return self._tfidf_similarity(text1, text2)
        elif method == 'token':
            return self._token_similarity(text1, text2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
            
    def _transformer_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using sentence transformers"""
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
        
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF based similarity"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
        tfidf = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(similarity)
        
    def _token_similarity(self, text1: str, text2: str) -> float:
        """Calculate token overlap based similarity"""
        tokens1 = set(word_tokenize(text1))
        tokens2 = set(word_tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
        
    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        method: str = 'transformer',
        top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """Find most similar candidates for query text
        
        Args:
            query: Query text
            candidates: List of candidate texts
            method: Similarity method to use
            top_k: Number of results to return
            
        Returns:
            List of (text, score) tuples, sorted by similarity
        """
        if not candidates:
            return []
            
        similarities = []
        for candidate in candidates:
            score = self.get_similarity(query, candidate, method)
            similarities.append((candidate, score))
            
        # Sort by score descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]