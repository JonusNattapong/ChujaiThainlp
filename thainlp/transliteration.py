"""
Thai text transliteration functionality
"""

from typing import Dict, Optional

# Thai consonant mapping
THAI_CONSONANTS: Dict[str, str] = {
    'ก': 'k', 'ข': 'kh', 'ค': 'kh', 'ฆ': 'kh', 'ง': 'ng',
    'จ': 'ch', 'ฉ': 'ch', 'ช': 'ch', 'ซ': 's', 'ฌ': 'ch',
    'ญ': 'y', 'ฎ': 'd', 'ฏ': 't', 'ฐ': 'th', 'ฑ': 'th',
    'ฒ': 'th', 'ณ': 'n', 'ด': 'd', 'ต': 't', 'ถ': 'th',
    'ท': 'th', 'ธ': 'th', 'น': 'n', 'บ': 'b', 'ป': 'p',
    'ผ': 'ph', 'ฝ': 'f', 'พ': 'ph', 'ฟ': 'f', 'ภ': 'ph',
    'ม': 'm', 'ย': 'y', 'ร': 'r', 'ล': 'l', 'ว': 'w',
    'ศ': 's', 'ษ': 's', 'ส': 's', 'ห': 'h', 'ฬ': 'l',
    'อ': '', 'ฮ': 'h'
}

# Thai vowel mapping
THAI_VOWELS: Dict[str, str] = {
    'ะ': 'a', 'า': 'a', 'ิ': 'i', 'ี': 'i', 'ึ': 'ue',
    'ื': 'ue', 'ุ': 'u', 'ู': 'u', 'เ': 'e', 'แ': 'ae',
    'โ': 'o', 'ใ': 'ai', 'ไ': 'ai', '็': '', '่': '',
    '้': '', '๊': '', '๋': '', 'ั': 'a'
}

def thai_to_roman(text: str) -> str:
    """
    Convert Thai text to Roman script.
    
    Args:
        text (str): Thai text to convert
        
    Returns:
        str: Romanized text
    """
    result = []
    i = 0
    while i < len(text):
        char = text[i]
        
        # Handle consonants
        if char in THAI_CONSONANTS:
            result.append(THAI_CONSONANTS[char])
            i += 1
            continue
            
        # Handle vowels
        if char in THAI_VOWELS:
            result.append(THAI_VOWELS[char])
            i += 1
            continue
            
        # Handle other characters
        if char.isspace():
            result.append(' ')
        else:
            result.append(char)
        i += 1
        
    return ''.join(result)

def roman_to_thai(text: str) -> str:
    """
    Convert Roman script to Thai text.
    
    Args:
        text (str): Roman text to convert
        
    Returns:
        str: Thai text
    """
    # This is a basic implementation and may need improvement
    # for handling complex cases
    result = []
    i = 0
    while i < len(text):
        char = text[i].lower()
        
        # Handle consonants
        for thai, roman in THAI_CONSONANTS.items():
            if roman == char:
                result.append(thai)
                i += 1
                break
        else:
            # Handle vowels
            for thai, roman in THAI_VOWELS.items():
                if roman == char:
                    result.append(thai)
                    i += 1
                    break
            else:
                # Handle other characters
                if char.isspace():
                    result.append(' ')
                else:
                    result.append(char)
                i += 1
                
    return ''.join(result) 