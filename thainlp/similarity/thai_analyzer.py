"""
Thai text analysis and semantic similarity
"""
from typing import List
import torch
from torch.nn import functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

class ThaiTextAnalyzer:
    """Analyze Thai text"""
    
    def __init__(self):
        self.model_name = "airesearch/wangchanberta-base-att-spm-uncased"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Encode texts
        encoding1 = self.tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
        encoding2 = self.tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
        
        # Get embeddings
        with torch.no_grad():
            embedding1 = self.model(**encoding1).logits
            embedding2 = self.model(**encoding2).logits
            
        # Calculate cosine similarity
        similarity = F.cosine_similarity(embedding1, embedding2).item()
        return (similarity + 1) / 2  # Normalize to [0,1]