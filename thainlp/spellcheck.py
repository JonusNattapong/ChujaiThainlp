"""
Thai spell checking functionality
"""

from typing import List, Tuple, Dict
import re

# Thai dictionary (basic)
THAI_DICT: Dict[str, bool] = {
    'สวัสดี': True, 'ครับ': True, 'ค่ะ': True, 'ขอบคุณ': True,
    'ยินดี': True, 'ต้อนรับ': True, 'บ้าน': True, 'รถ': True,
    'คน': True, 'หนังสือ': True, 'อาหาร': True, 'น้ำ': True,
    'อากาศ': True, 'กิน': True, 'เดิน': True, 'นอน': True,
    'อ่าน': True, 'เขียน': True, 'พูด': True, 'คิด': True,
}

def check_spelling(text: str) -> List[Tuple[str, int, int, List[str]]]:
    """
    Check spelling in Thai text.
    
    Args:
        text (str): Thai text to check
        
    Returns:
        List[Tuple[str, int, int, List[str]]]: List of (word, start, end, suggestions) tuples
    """
    # Split text into words
    words = text.split()
    misspelled = []
    
    for word in words:
        # Skip if word is in dictionary
        if word in THAI_DICT:
            continue
            
        # Find position in original text
        start = text.find(word)
        end = start + len(word)
        
        # Generate suggestions (basic implementation)
        suggestions = []
        
        # 1. Check for common typos (missing/extra characters)
        for i in range(len(word)):
            # Remove character
            without_char = word[:i] + word[i+1:]
            if without_char in THAI_DICT:
                suggestions.append(without_char)
            
            # Add character
            for char in 'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ':
                with_char = word[:i] + char + word[i:]
                if with_char in THAI_DICT:
                    suggestions.append(with_char)
        
        # 2. Check for swapped characters
        for i in range(len(word)-1):
            swapped = word[:i] + word[i+1] + word[i] + word[i+2:]
            if swapped in THAI_DICT:
                suggestions.append(swapped)
        
        if suggestions:
            misspelled.append((word, start, end, suggestions))
    
    return misspelled
