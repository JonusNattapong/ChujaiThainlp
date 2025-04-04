"""
Thai text processing utilities
"""
import re
from typing import List, Set

# Thai character ranges
THAI_CHARS = re.compile(r'[\u0E00-\u0E7F]')
THAI_VOWELS = re.compile(r'[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]')
THAI_TONEMARKS = re.compile(r'[\u0E48-\u0E4E]')

def is_thai_char(char: str) -> bool:
    """Check if character is Thai
    
    Args:
        char: Single character to check
        
    Returns:
        True if character is Thai, False otherwise
    """
    return bool(THAI_CHARS.match(char))

def contains_thai(text: str) -> bool:
    """Check if text contains Thai characters
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains Thai characters, False otherwise
    """
    return bool(THAI_CHARS.search(text))

def normalize_text(text: str) -> str:
    """Normalize Thai text
    
    Performs:
    - Removes duplicate whitespace
    - Normalizes quotes and spaces
    - Standardizes line endings
    - Removes zero-width characters
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return text
        
    # Remove zero-width characters
    text = re.sub(r'\u200B', '', text)
    
    # Normalize quotes and spaces
    text = re.sub(r'["\u201C\u201D]', '"', text)  # Smart double quotes
    text = re.sub(r'[\u2018\u2019\u0060]', "'", text)  # Smart single quotes
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Standardize line endings
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    return text.strip()

def remove_tonemark(text: str) -> str:
    """Remove Thai tone marks from text
    
    Args:
        text: Text to process
        
    Returns:
        Text with tone marks removed
    """
    return THAI_TONEMARKS.sub('', text)

def remove_vowels(text: str) -> str:
    """Remove Thai vowel marks from text
    
    Args:
        text: Text to process
        
    Returns:
        Text with vowel marks removed
    """
    return THAI_VOWELS.sub('', text)

def separate_thai_english(text: str) -> List[str]:
    """Separate Thai and English segments in text
    
    Args:
        text: Mixed Thai-English text
        
    Returns:
        List of text segments, alternating between Thai and non-Thai
    """
    segments = []
    current = []
    prev_is_thai = None
    
    for char in text:
        is_thai = is_thai_char(char)
        
        # Start new segment if switching between Thai/non-Thai
        if prev_is_thai is not None and is_thai != prev_is_thai:
            segments.append(''.join(current))
            current = []
            
        current.append(char)
        prev_is_thai = is_thai
        
    if current:
        segments.append(''.join(current))
        
    return segments

def clean_thai_text(text: str) -> str:
    """Clean Thai text for processing
    
    Performs:
    - Normalization
    - Removes HTML tags
    - Removes special characters
    - Standardizes whitespace
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return text
        
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize text
    text = normalize_text(text)
    
    # Remove special characters but keep Thai characters
    text = re.sub(r'[^\u0E00-\u0E7F\s\w]', '', text)
    
    # Standardize whitespace
    text = ' '.join(text.split())
    
    return text

def extract_thai_segments(text: str) -> List[str]:
    """Extract Thai text segments
    
    Args:
        text: Mixed text
        
    Returns:
        List of Thai text segments
    """
    segments = []
    current = []
    
    for char in text:
        if is_thai_char(char):
            current.append(char)
        elif current:
            segments.append(''.join(current))
            current = []
            
    if current:
        segments.append(''.join(current))
        
    return segments

def get_thai_words(text: str) -> Set[str]:
    """Extract unique Thai words from text
    
    Args:
        text: Text to process
        
    Returns:
        Set of unique Thai words
    """
    from ..tokenization import word_tokenize
    
    # Get Thai segments
    thai_segments = extract_thai_segments(text)
    
    # Tokenize and collect unique words
    words = set()
    for segment in thai_segments:
        words.update(word_tokenize(segment))
        
    return words

def detect_language_mix(text: str) -> dict:
    """Detect language mix in text
    
    Args:
        text: Text to analyze
        
    Returns:
        Dict with:
        - thai_char_ratio: Ratio of Thai characters
        - eng_char_ratio: Ratio of English characters
        - other_char_ratio: Ratio of other characters
        - primary_script: 'thai', 'english' or 'mixed'
    """
    if not text:
        return {
            'thai_char_ratio': 0,
            'eng_char_ratio': 0,
            'other_char_ratio': 0,
            'primary_script': 'unknown'
        }
        
    thai_chars = 0
    eng_chars = 0
    other_chars = 0
    total_chars = 0
    
    for char in text:
        if is_thai_char(char):
            thai_chars += 1
        elif char.isascii() and char.isalpha():
            eng_chars += 1
        else:
            other_chars += 1
        total_chars += 1
        
    thai_ratio = thai_chars / total_chars if total_chars > 0 else 0
    eng_ratio = eng_chars / total_chars if total_chars > 0 else 0
    other_ratio = other_chars / total_chars if total_chars > 0 else 0
    
    # Determine primary script
    if thai_ratio > 0.5:
        primary = 'thai'
    elif eng_ratio > 0.5:
        primary = 'english'
    else:
        primary = 'mixed'
        
    return {
        'thai_char_ratio': thai_ratio,
        'eng_char_ratio': eng_ratio,
        'other_char_ratio': other_ratio,
        'primary_script': primary
    }
