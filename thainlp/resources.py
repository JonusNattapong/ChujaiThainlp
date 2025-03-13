"""
Thai language resources (dictionaries, stopwords, etc.)
"""
from typing import List, Set, Dict, Optional
import os
import json
import warnings
import pkg_resources

try:
    import pythainlp
    from pythainlp.corpus import thai_stopwords as pythainlp_stopwords
    from pythainlp.corpus import thai_words as pythainlp_words
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    warnings.warn("PyThaiNLP not found. Using simplified language resources.")

# Basic Thai stopwords
_THAI_STOPWORDS = {
    "และ", "แล้ว", "แต่", "หรือ", "ของ", "ใน", "มี", "ไป", "มา", "ว่า", "ที่",
    "จะ", "ไม่", "ให้", "ได้", "เป็น", "มาก", "ความ", "การ", "เพราะ", "อยู่", 
    "อย่าง", "ก็", "นี้"
}

# Function to get Thai stopwords
def thai_stopwords() -> Set[str]:
    """
    Get Thai stopwords.
    
    Returns:
        Set of Thai stopwords
    """
    if PYTHAINLP_AVAILABLE:
        return set(pythainlp_stopwords())
    else:
        return _THAI_STOPWORDS

# Function to get Thai words dictionary
def thai_words() -> Set[str]:
    """
    Get Thai words dictionary.
    
    Returns:
        Set of Thai words
    """
    if PYTHAINLP_AVAILABLE:
        return set(pythainlp_words())
    else:
        # Return our small dictionary as fallback
        from thainlp.tokenize import _THAI_WORDS
        return set(_THAI_WORDS.keys())

def load_custom_dictionary(file_path: Optional[str] = None) -> Dict[str, bool]:
    """
    Load a custom dictionary from a file.
    
    Args:
        file_path: Path to dictionary file (one word per line)
        
    Returns:
        Dictionary of words
    """
    words = {}
    
    if file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    words[word] = True
    
    return words

def load_custom_stopwords(file_path: Optional[str] = None) -> Set[str]:
    """
    Load custom stopwords from a file (one word per line).

    Args:
        file_path: Path to the stopword file.

    Returns:
        Set of stopwords.
    """
    stopwords = set()
    if file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
    return stopwords

def combine_dictionaries(custom_dict_path: Optional[str] = None,
                        custom_stopwords_path: Optional[str] = None) -> tuple[Dict[str, bool], Set[str]]:
    """
    Combine the default dictionary, PyThaiNLP's dictionary (if available),
    a custom dictionary, and custom stopwords.

    Args:
        custom_dict_path: Path to a custom dictionary file.
        custom_stopwords_path: Path to a custom stopwords file.

    Returns:
        A tuple containing:
        - Combined dictionary (Dict[str, bool])
        - Combined stopwords (Set[str])
    """
    words = thai_words()
    stopwords = thai_stopwords()

    if custom_dict_path:
        custom_dict = load_custom_dictionary(custom_dict_path)
        words.update(custom_dict.keys())  # Add custom dictionary words

    if custom_stopwords_path:
        custom_stopwords = load_custom_stopwords(custom_stopwords_path)
        stopwords.update(custom_stopwords)

    return words, stopwords

# Thai romanization mappings
_THAI_ROMANIZE = {
    'ก': 'k', 'ข': 'kh', 'ฃ': 'kh', 'ค': 'kh', 'ฅ': 'kh', 'ฆ': 'kh',
    'ง': 'ng', 'จ': 'ch', 'ฉ': 'ch', 'ช': 'ch', 'ซ': 's', 'ฌ': 'ch',
    'ญ': 'y', 'ฎ': 'd', 'ฏ': 't', 'ฐ': 'th', 'ฑ': 'th', 'ฒ': 'th',
    'ณ': 'n', 'ด': 'd', 'ต': 't', 'ถ': 'th', 'ท': 'th', 'ธ': 'th',
    'น': 'n', 'บ': 'b', 'ป': 'p', 'ผ': 'ph', 'ฝ': 'f', 'พ': 'ph',
    'ฟ': 'f', 'ภ': 'ph', 'ม': 'm', 'ย': 'y', 'ร': 'r', 'ฤ': 'rue',
    'ล': 'l', 'ฦ': 'lue', 'ว': 'w', 'ศ': 's', 'ษ': 's', 'ส': 's',
    'ห': 'h', 'ฬ': 'l', 'อ': '', 'ฮ': 'h',
    # vowels and tone marks
    'ะ': 'a', 'ั': 'a', 'า': 'a', 'ำ': 'am', 'ิ': 'i', 'ี': 'i',
    'ึ': 'ue', 'ื': 'ue', 'ุ': 'u', 'ู': 'u', 'เ': 'e', 'แ': 'ae',
    'โ': 'o', 'ใ': 'ai', 'ไ': 'ai', '่': '', '้': '', '๊': '', '๋': '',
    '็': '', '์': '', 'ๆ': '2', 'ฯ': '...',
}

def romanize(text: str) -> str:
    """
    Convert Thai text to Romanized form.
    
    Args:
        text: Thai text
        
    Returns:
        Romanized text
    """
    result = []
    for char in text:
        if char in _THAI_ROMANIZE:
            result.append(_THAI_ROMANIZE[char])
        else:
            result.append(char)
    return ''.join(result)
