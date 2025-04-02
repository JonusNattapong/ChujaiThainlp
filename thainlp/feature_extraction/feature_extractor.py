"""
Feature extraction utilities for Thai text
"""
from typing import List, Dict, Set
from collections import Counter
import math
from ..tokenization import word_tokenize
from ..resources import get_stopwords

class FeatureExtractor:
    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords
        self.stopwords = get_stopwords() if remove_stopwords else set()
        self.vocabulary = set()
        self.idf_scores = {}
        
    def extract_bow(self, text: str) -> Dict[str, int]:
        """Extract bag-of-words features
        
        Args:
            text: Input text
            
        Returns:
            Dict[str, int]: Word frequency dictionary
        """
        tokens = word_tokenize(text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
            
        return dict(Counter(tokens))
        
    def extract_tfidf(self, documents: List[str]) -> List[Dict[str, float]]:
        """Extract TF-IDF features for a collection of documents
        
        Args:
            documents: List of input documents
            
        Returns:
            List[Dict[str, float]]: TF-IDF vectors for each document
        """
        # Extract tokens
        doc_tokens = []
        for doc in documents:
            tokens = word_tokenize(doc)
            if self.remove_stopwords:
                tokens = [t for t in tokens if t not in self.stopwords]
            doc_tokens.append(tokens)
            
        # Build vocabulary
        for tokens in doc_tokens:
            self.vocabulary.update(tokens)
            
        # Calculate IDF scores
        doc_count = len(documents)
        for word in self.vocabulary:
            doc_with_word = sum(1 for tokens in doc_tokens if word in tokens)
            self.idf_scores[word] = math.log(doc_count / (1 + doc_with_word))
            
        # Calculate TF-IDF for each document
        tfidf_vectors = []
        for tokens in doc_tokens:
            bow = Counter(tokens)
            tfidf = {}
            for word, tf in bow.items():
                tfidf[word] = tf * self.idf_scores[word]
            tfidf_vectors.append(tfidf)
            
        return tfidf_vectors
        
    def extract_ngrams(self, text: str, n: int = 2) -> List[str]:
        """Extract character n-grams from text
        
        Args:
            text: Input text
            n: n-gram size
            
        Returns:
            List[str]: List of n-grams
        """
        return [text[i:i+n] for i in range(len(text)-n+1)]
        
    def extract_pos_patterns(self, text: str, pos_tagger=None) -> List[str]:
        """Extract POS tag patterns from text
        
        Args:
            text: Input text
            pos_tagger: Optional POS tagger to use
            
        Returns:
            List[str]: List of POS tag sequences
        """
        if pos_tagger is None:
            from ..tag import pos_tag
            pos_tagger = pos_tag
            
        tokens = word_tokenize(text)
        tagged = pos_tagger(text, tokenize=False)
        patterns = []
        
        for i in range(len(tagged)-1):
            pattern = f"{tagged[i][1]}_{tagged[i+1][1]}"
            patterns.append(pattern)
            
        return patterns
