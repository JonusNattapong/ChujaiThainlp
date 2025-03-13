"""
Utility functions for Thai text processing.
"""
import re
from typing import List, Dict

# Thai character classifications
_TH_CHARS = {
    "consonants": "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ",
    "vowels": "ะัาำิีึืุูเแโใไ็่้๊๋",
    "tonemarks": "่้๊๋",
}

def remove_tonemarks(text: str) -> str:
    """
    Remove Thai tone marks from text.
    
    Args:
        text: Thai text
        
    Returns:
        Text without tone marks
    """
    pattern = f"[{_TH_CHARS['tonemarks']}]"
    return re.sub(pattern, "", text)

def normalize_text(text: str) -> str:
    """
    Normalize Thai text by removing duplicate spaces,
    converting line endings, etc.
    
    Args:
        text: Thai text to normalize
        
    Returns:
        Normalized text
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text

def count_thai_chars(text: str) -> Dict[str, int]:
    """
    Count Thai characters in text by category.
    
    Args:
        text: Thai text
        
    Returns:
        Dictionary with counts by character category
    """
    result = {
        "consonants": 0,
        "vowels": 0,
        "tonemarks": 0,
        "total_thai": 0,
        "other": 0
    }
    
    for char in text:
        if char in _TH_CHARS["consonants"]:
            result["consonants"] += 1
            result["total_thai"] += 1
        elif char in _TH_CHARS["vowels"]:
            result["vowels"] += 1
            result["total_thai"] += 1
        elif char in _TH_CHARS["tonemarks"]:
            result["tonemarks"] += 1
            result["total_thai"] += 1
        else:
            result["other"] += 1
            
    return result