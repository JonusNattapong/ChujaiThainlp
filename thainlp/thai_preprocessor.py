"""
Thai text preprocessing utilities
"""
import re
from typing import List, Union

class ThaiTextPreprocessor:
    """Basic Thai text preprocessor"""
    
    def __init__(self):
        # Common Thai abbreviations
        self.abbreviations = {
            'พศ': 'พ.ศ.',
            'มค': 'มกราคม',
            'กพ': 'กุมภาพันธ์',
            'มีค': 'มีนาคม',
            'เมย': 'เมษายน',
            'พค': 'พฤษภาคม',
            'มิย': 'มิถุนายน',
            'กค': 'กรกฎาคม',
            'สค': 'สิงหาคม',
            'กย': 'กันยายน',
            'ตค': 'ตุลาคม',
            'พย': 'พฤศจิกายน',
            'ธค': 'ธันวาคม'
        }
        
        # Common misspellings
        self.corrections = {
            'เเ': 'แ',
            'กั': 'ก็',
            'เรือง': 'เรื่อง',
            'เปน': 'เป็น'
        }
    
    def preprocess(self, text: str) -> str:
        """
        Basic Thai text preprocessing
        
        Args:
            text: Input Thai text
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return text
            
        # Fix common errors
        for wrong, correct in self.corrections.items():
            text = text.replace(wrong, correct)
            
        # Expand abbreviations
        for abbrev, full in self.abbreviations.items():
            text = re.sub(rf'\b{abbrev}\b', full, text)
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
        
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """
        Preprocess multiple texts
        
        Args:
            texts: List of Thai texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]