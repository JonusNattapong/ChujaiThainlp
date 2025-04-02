"""
Thai masked language model inference
"""
from typing import List, Dict, Tuple, Optional
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize

class FillMask(TransformerBase):
    def __init__(self, model_name: str = ""):
        super().__init__(model_name)
        self.mask_token = "[MASK]"
        self._init_vocab()
        
    def _init_vocab(self):
        """Initialize basic Thai vocabulary"""
        self.vocab = {
            # Common nouns
            'คน': 0.8, 'บ้าน': 0.7, 'รถ': 0.7, 'หมา': 0.6, 'แมว': 0.6,
            'อาหาร': 0.7, 'น้ำ': 0.7, 'ต้นไม้': 0.6, 'ดอกไม้': 0.6,
            
            # Common verbs
            'กิน': 0.8, 'เดิน': 0.7, 'วิ่ง': 0.7, 'นอน': 0.7, 'ดู': 0.7,
            'ฟัง': 0.7, 'พูด': 0.7, 'คิด': 0.7, 'ทำ': 0.8,
            
            # Common adjectives
            'ดี': 0.8, 'สวย': 0.7, 'ใหญ่': 0.7, 'เล็ก': 0.7, 'เร็ว': 0.7,
            'ช้า': 0.7, 'ร้อน': 0.7, 'เย็น': 0.7, 'ง่าย': 0.7, 'ยาก': 0.7
        }
        
    def fill_mask(
        self,
        text: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, any]]:
        """Fill masked token in text
        
        Args:
            text: Input text with [MASK] token
            top_k: Number of top predictions to return
            threshold: Minimum score threshold
            
        Returns:
            List of dicts containing:
            - token: Predicted token
            - score: Prediction score
            - sequence: Full sequence with prediction
        """
        if self.mask_token not in text:
            return []
            
        try:
            # Get predictions from model if available
            if self.model:
                return self.model.fill_mask(
                    text,
                    top_k=top_k,
                    threshold=threshold
                )
                
            # Fall back to basic vocabulary-based prediction
            return self._basic_fill_mask(text, top_k, threshold)
            
        except Exception as e:
            self.logger.error(f"Fill mask failed: {str(e)}")
            return []
            
    def _basic_fill_mask(
        self,
        text: str,
        top_k: int,
        threshold: float
    ) -> List[Dict[str, any]]:
        """Basic fill mask using vocabulary frequencies"""
        # Get context around mask
        prefix, suffix = text.split(self.mask_token)
        prefix_tokens = word_tokenize(prefix.strip())
        suffix_tokens = word_tokenize(suffix.strip())
        
        predictions = []
        
        # Score each vocabulary word
        for token, base_score in self.vocab.items():
            # Adjust score based on context
            score = base_score
            
            # Check prefix context
            if prefix_tokens:
                last_prefix = prefix_tokens[-1]
                if self._tokens_often_adjacent(last_prefix, token):
                    score *= 1.2
                    
            # Check suffix context
            if suffix_tokens:
                first_suffix = suffix_tokens[0]
                if self._tokens_often_adjacent(token, first_suffix):
                    score *= 1.2
                    
            if score >= threshold:
                sequence = f"{prefix}{token}{suffix}"
                predictions.append({
                    'token': token,
                    'score': float(score),
                    'sequence': sequence.strip()
                })
                
        # Sort by score and return top k
        predictions.sort(key=lambda x: x['score'], reverse=True)
        return predictions[:top_k]
        
    def _tokens_often_adjacent(self, token1: str, token2: str) -> bool:
        """Check if tokens commonly appear next to each other"""
        # Add logic for common token pairs
        common_pairs = {
            ('คน', 'ดี'),
            ('บ้าน', 'ใหญ่'),
            ('รถ', 'เร็ว'),
            ('อาหาร', 'อร่อย'),
            ('น้ำ', 'เย็น')
        }
        return (token1, token2) in common_pairs
