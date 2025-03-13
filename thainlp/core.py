"""
Core functionality for Thai Natural Language Processing
"""

from typing import List, Union
import re

def word_tokenize(text: str) -> List[str]:
    """
    Tokenize Thai text into words.
    
    Args:
        text (str): Thai text to tokenize
        
    Returns:
        List[str]: List of words
    """
    # Basic implementation using regex
    pattern = r'[ก-๛]+|[0-9]+|[a-zA-Z]+'
    return re.findall(pattern, text)

def sentence_tokenize(text: str) -> List[str]:
    """
    Tokenize Thai text into sentences.
    
    Args:
        text (str): Thai text to tokenize
        
    Returns:
        List[str]: List of sentences
    """
    # Basic implementation using regex
    pattern = r'[^.!?]+[.!?]+'
    return [s.strip() for s in re.findall(pattern, text)]

def normalize(text: str) -> str:
    """
    Normalize Thai text by removing extra spaces and normalizing whitespace.
    
    Args:
        text (str): Thai text to normalize
        
    Returns:
        str: Normalized text
    """
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Normalize whitespace
    text = text.strip()
    return text

def is_thai(text: str) -> bool:
    """
    Check if text contains Thai characters.
    
    Args:
        text (str): Text to check
        
    Returns:
        bool: True if text contains Thai characters
    """
    thai_pattern = r'[ก-๛]'
    return bool(re.search(thai_pattern, text))

def count_thai_chars(text: str) -> int:
    """
    Count number of Thai characters in text.
    
    Args:
        text (str): Text to count Thai characters
        
    Returns:
        int: Number of Thai characters
    """
    thai_pattern = r'[ก-๛]'
    return len(re.findall(thai_pattern, text))

def remove_thai_vowels(text: str) -> str:
    """
    Remove Thai vowels from text.
    
    Args:
        text (str): Thai text to remove vowels
        
    Returns:
        str: Text without Thai vowels
    """
    vowels = r'[ะาิีึืุูเแโใไ็่้๊๋ั'
    return re.sub(vowels, '', text)

def is_palindrome(text: str) -> bool:
    """
    Check if Thai text is palindrome.
    
    Args:
        text (str): Thai text to check
        
    Returns:
        bool: True if text is palindrome
    """
    # Remove spaces and convert to lowercase
    text = normalize(text)
    return text == text[::-1] 