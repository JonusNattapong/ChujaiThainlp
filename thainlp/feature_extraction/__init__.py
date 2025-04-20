"""
Feature extraction for Thai text processing
"""
from typing import List, Union, Dict
import numpy as np
from collections import defaultdict
import math
from ..tokenization import word_tokenize

class TfidfVectorizer:
    """TF-IDF Vectorizer for Thai text"""
    
    def __init__(self):
        self.document_freq = defaultdict(int)
        self.vocab = {}
        self.idf = {}
        self.num_docs = 0
        
    def fit(self, documents: List[str]):
        """Calculate document frequencies and IDF values"""
        # Reset state
        self.document_freq.clear()
        self.vocab.clear()
        self.num_docs = len(documents)
        
        # Calculate document frequencies
        for doc in documents:
            tokens = word_tokenize(doc)
            # Count each token only once per document
            seen_tokens = set(tokens)
            for token in seen_tokens:
                self.document_freq[token] += 1
        
        # Create vocabulary
        self.vocab = {token: idx for idx, token in enumerate(self.document_freq.keys())}
        
        # Calculate IDF values
        self.idf = {
            token: math.log((self.num_docs + 1) / (freq + 1)) + 1
            for token, freq in self.document_freq.items()
        }
        
    def transform(self, text: str) -> np.ndarray:
        """Transform text to TF-IDF vector"""
        tokens = word_tokenize(text)
        
        if not self.vocab:
            raise ValueError("Vectorizer must be fit before transform")
            
        # Calculate term frequencies
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
            
        # Convert to TF-IDF
        vector = np.zeros(len(self.vocab))
        for token, freq in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                tf_idf = freq * self.idf.get(token, 0)
                vector[idx] = tf_idf
                
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector

_default_vectorizer = None

def get_vectorizer() -> TfidfVectorizer:
    """Get or create default vectorizer"""
    global _default_vectorizer
    if _default_vectorizer is None:
        _default_vectorizer = TfidfVectorizer()
    return _default_vectorizer

def create_document_vector(text: str, model_name: str = None) -> np.ndarray:
    """Create document vector representation
    
    Args:
        text: Input text
        model_name: Name of the pretrained model (unused in current implementation)
        
    Returns:
        Document vector as numpy array
    """
    vectorizer = get_vectorizer()
    
    # Fit on single document if not already fit
    if not vectorizer.vocab:
        vectorizer.fit([text])
        
    return vectorizer.transform(text)

__all__ = ['TfidfVectorizer', 'create_document_vector']