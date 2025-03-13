"""
Thai Part-of-Speech tagging functionality
"""

from typing import List, Tuple, Dict
from .resources import THAI_POS_DICT

def pos_tag(text: str) -> List[Tuple[str, str]]:
    """
    Tag parts of speech in Thai text with enhanced accuracy.
    
    Args:
        text (str): Thai text to tag
        
    Returns:
        List[Tuple[str, str]]: List of (word, POS tag) tuples
    """
    words = text.split()
    tagged_words = []
    
    # Use dictionary lookup for known words
    for word in words:
        # Get POS tag from dictionary
        pos = THAI_POS_DICT.get(word, 'UNKNOWN')
        
        # Apply some basic rules for unknown words
        if pos == 'UNKNOWN':
            # Check for common suffixes
            if word.endswith('ๆ'):
                pos = 'ADV'
            elif word.endswith('การ'):
                pos = 'NOUN'
            elif word.endswith('ความ'):
                pos = 'NOUN'
            elif word.endswith('ที่'):
                pos = 'DET'
            elif word.endswith('ของ'):
                pos = 'PREP'
            
            # Check for common prefixes
            elif word.startswith('การ'):
                pos = 'NOUN'
            elif word.startswith('ความ'):
                pos = 'NOUN'
            elif word.startswith('ที่'):
                pos = 'DET'
            elif word.startswith('ของ'):
                pos = 'PREP'
            
            # Check for numbers
            elif word.replace('.', '').isdigit():
                pos = 'NUM'
            
            # Check for Thai characters
            elif any('\u0E00' <= c <= '\u0E7F' for c in word):
                # Default to NOUN for unknown Thai words
                pos = 'NOUN'
        
        tagged_words.append((word, pos))
    
    return tagged_words 