"""
Utility Functions for Thai Language Processing
"""

from typing import List, Dict, Tuple, Set
import re

def is_thai_char(char: str) -> bool:
    """
    Check if a character is a Thai character
    
    Args:
        char (str): Input character
        
    Returns:
        bool: True if character is Thai
    """
    return '\u0E00' <= char <= '\u0E7F'

def is_thai_word(word: str) -> bool:
    """
    Check if a word contains Thai characters
    
    Args:
        word (str): Input word
        
    Returns:
        bool: True if word contains Thai characters
    """
    return any(is_thai_char(char) for char in word)

def remove_tone_marks(text: str) -> str:
    """
    Remove tone marks from Thai text
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text without tone marks
    """
    # Thai tone marks: 0E48-0E4B
    tone_marks = '\u0E48\u0E49\u0E4A\u0E4B'
    return ''.join(c for c in text if c not in tone_marks)

def remove_diacritics(text: str) -> str:
    """
    Remove diacritics from Thai text
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text without diacritics
    """
    # Thai diacritics: 0E31, 0E34-0E3A, 0E47-0E4E
    diacritics = '\u0E31\u0E34\u0E35\u0E36\u0E37\u0E38\u0E39\u0E3A\u0E47\u0E48\u0E49\u0E4A\u0E4B\u0E4C\u0E4D\u0E4E'
    return ''.join(c for c in text if c not in diacritics)

def normalize_text(text: str) -> str:
    """
    Normalize Thai text by removing whitespace and converting to lowercase
    
    Args:
        text (str): Input text
        
    Returns:
        str: Normalized text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Convert to lowercase (for non-Thai characters)
    result = ''
    for c in text:
        if not is_thai_char(c):
            result += c.lower()
        else:
            result += c
            
    return result

def count_thai_words(text: str) -> int:
    """
    Count Thai words in text (approximation)
    
    Args:
        text (str): Input text
        
    Returns:
        int: Number of Thai words
    """
    # This is a simple approximation
    # For better results, use a proper tokenizer
    words = text.split()
    thai_words = [word for word in words if is_thai_word(word)]
    return len(thai_words)

def extract_thai_text(text: str) -> str:
    """
    Extract only Thai text from mixed text
    
    Args:
        text (str): Input text
        
    Returns:
        str: Thai text only
    """
    return ''.join(c for c in text if is_thai_char(c) or c.isspace())

def thai_to_roman(text: str) -> str:
    """
    Convert Thai text to Roman alphabet (basic transliteration)
    
    Args:
        text (str): Input Thai text
        
    Returns:
        str: Romanized text
    """
    # Basic mapping of Thai consonants to Roman alphabet
    consonant_map = {
        'ก': 'k', 'ข': 'kh', 'ฃ': 'kh', 'ค': 'kh', 'ฅ': 'kh', 'ฆ': 'kh',
        'ง': 'ng', 'จ': 'ch', 'ฉ': 'ch', 'ช': 'ch', 'ซ': 's', 'ฌ': 'ch',
        'ญ': 'y', 'ฎ': 'd', 'ฏ': 't', 'ฐ': 'th', 'ฑ': 'th', 'ฒ': 'th',
        'ณ': 'n', 'ด': 'd', 'ต': 't', 'ถ': 'th', 'ท': 'th', 'ธ': 'th',
        'น': 'n', 'บ': 'b', 'ป': 'p', 'ผ': 'ph', 'ฝ': 'f', 'พ': 'ph',
        'ฟ': 'f', 'ภ': 'ph', 'ม': 'm', 'ย': 'y', 'ร': 'r', 'ล': 'l',
        'ว': 'w', 'ศ': 's', 'ษ': 's', 'ส': 's', 'ห': 'h', 'ฬ': 'l',
        'อ': '', 'ฮ': 'h'
    }
    
    # Basic mapping of Thai vowels to Roman alphabet
    vowel_map = {
        'ะ': 'a', 'ั': 'a', 'า': 'a', 'ำ': 'am', 'ิ': 'i', 'ี': 'i',
        'ึ': 'ue', 'ื': 'ue', 'ุ': 'u', 'ู': 'u', 'เ': 'e', 'แ': 'ae',
        'โ': 'o', 'ใ': 'ai', 'ไ': 'ai', '็': '', '่': '', '้': '',
        '๊': '', '๋': '', '์': '', 'ๆ': ''
    }
    
    result = ''
    i = 0
    while i < len(text):
        char = text[i]
        
        if char in consonant_map:
            result += consonant_map[char]
        elif char in vowel_map:
            result += vowel_map[char]
        elif char.isspace():
            result += ' '
        else:
            result += char
            
        i += 1
        
    return result

def detect_language(text: str) -> str:
    """
    Detect if text is primarily Thai, English, or mixed
    
    Args:
        text (str): Input text
        
    Returns:
        str: 'thai', 'english', or 'mixed'
    """
    # Count characters
    thai_chars = sum(1 for c in text if is_thai_char(c))
    english_chars = sum(1 for c in text if c.isalpha() and not is_thai_char(c))
    
    # Calculate percentages
    total_chars = thai_chars + english_chars
    if total_chars == 0:
        return 'unknown'
        
    thai_percent = thai_chars / total_chars * 100
    english_percent = english_chars / total_chars * 100
    
    # Determine language
    if thai_percent > 80:
        return 'thai'
    elif english_percent > 80:
        return 'english'
    else:
        return 'mixed' 