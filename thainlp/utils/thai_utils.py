"""
Thai language utilities

This module provides comprehensive utilities for Thai language processing including:
- Text normalization
- Character/word detection
- Romanization
- Number/date formatting
- Basic spell checking
"""
from typing import List, Dict, Set, Tuple, Optional, Union
import re
import datetime
from collections import defaultdict

# Thai character ranges
THAI_CHARS = '\u0E00-\u0E7F'  # ช่วงอักขระไทยทั้งหมด
THAI_CONSONANTS = 'ก-ฮ'  # พยัญชนะไทย
THAI_VOWELS = 'ะัาำิีึืุูเแโใไๅ'  # สระไทย
THAI_TONEMARKS = '่้๊๋'  # วรรณยุกต์
THAI_DIGITS = '๐-๙'  # เลขไทย

# Thai number words
THAI_NUMBERS = {
    0: 'ศูนย์',
    1: 'หนึ่ง',
    2: 'สอง', 
    3: 'สาม',
    4: 'สี่',
    5: 'ห้า',
    6: 'หก',
    7: 'เจ็ด',
    8: 'แปด',
    9: 'เก้า',
    10: 'สิบ',
    20: 'ยี่สิบ',
    100: 'ร้อย',
    1000: 'พัน',
    10000: 'หมื่น',
    100000: 'แสน',
    1000000: 'ล้าน'
}

def is_thai_char(char: str) -> bool:
    """Check if a character is Thai"""
    return len(char) == 1 and bool(re.match(f'[{THAI_CHARS}]', char))

def is_thai_word(word: str) -> bool:
    """Check if word contains Thai characters"""
    return bool(re.search(f'[{THAI_CHARS}]', word))

def remove_tone_marks(text: str) -> str:
    """Remove Thai tone marks"""
    return re.sub(f'[{THAI_TONEMARKS}]', '', text)

def remove_diacritics(text: str) -> str:
    """Remove all Thai diacritics including tone marks and other symbols"""
    pattern = f'[{THAI_TONEMARKS}]|[์ํฺ]'  # รวมวรรณยุกต์และสัญลักษณ์พิเศษอื่นๆ
    return re.sub(pattern, '', text)

def normalize_text(text: str) -> str:
    """Normalize Thai text (remove redundant spaces, normalize characters)
    
    Performs:
    - Replace multiple spaces with single space
    - Convert Thai numerals to Arabic
    - Remove zero-width spaces and other invisible characters
    - Normalize line endings
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert Thai numerals to Arabic
    text = thai_digit_to_arabic_digit(text)
    
    # Remove zero-width spaces and other invisible characters
    text = re.sub(r'[\u200B-\u200F\uFEFF]', '', text)
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text.strip()

def count_thai_words(text: str) -> int:
    """Count Thai words in text"""
    words = text.split()
    return len([word for word in words if is_thai_word(word)])

def extract_thai_text(text: str) -> str:
    """Extract only Thai characters from text"""
    return ''.join(char for char in text if is_thai_char(char) or char.isspace())

def thai_to_roman(text: str) -> str:
    """Basic romanization of Thai text"""
    char_map = {
        'ก': 'k', 'ข': 'kh', 'ค': 'kh', 'ฆ': 'kh', 'ง': 'ng',
        'จ': 'ch', 'ฉ': 'ch', 'ช': 'ch', 'ซ': 's', 'ฌ': 'ch',
        'ญ': 'y', 'ฎ': 'd', 'ฏ': 't', 'ฐ': 'th', 'ฑ': 'th',
        'ฒ': 'th', 'ณ': 'n', 'ด': 'd', 'ต': 't', 'ถ': 'th',
        'ท': 'th', 'ธ': 'th', 'น': 'n', 'บ': 'b', 'ป': 'p',
        'ผ': 'ph', 'ฝ': 'f', 'พ': 'ph', 'ฟ': 'f', 'ภ': 'ph',
        'ม': 'm', 'ย': 'y', 'ร': 'r', 'ล': 'l', 'ว': 'w',
        'ศ': 's', 'ษ': 's', 'ส': 's', 'ห': 'h', 'ฬ': 'l',
        'อ': '', 'ฮ': 'h',
        'ะ': 'a', 'ั': 'a', 'า': 'a', 'ำ': 'am', 'ิ': 'i',
        'ี': 'i', 'ึ': 'ue', 'ื': 'ue', 'ุ': 'u', 'ู': 'u',
        'เ': 'e', 'แ': 'ae', 'โ': 'o', 'ใ': 'ai', 'ไ': 'ai',
        '่': '', '้': '', '๊': '', '๋': '', '็': '', '์': ''
    }
    return ''.join(char_map.get(char, char) for char in text)

def detect_language(text: str) -> Dict[str, float]:
    """Detect script usage ratios in text
    
    Returns:
        Dict with script name keys and their usage ratio values:
        - 'thai': Thai script ratio
        - 'latin': Latin script ratio
        - 'chinese': Chinese characters ratio
        - 'japanese': Japanese characters ratio
        - 'korean': Korean characters ratio
        - 'other': Other scripts ratio
    """
    if not text:
        return {'thai': 0, 'latin': 0, 'chinese': 0, 'japanese': 0, 'korean': 0, 'other': 0}
        
    total = len(text)
    counts = {
        'thai': sum(1 for c in text if '\u0E00' <= c <= '\u0E7F'),
        'latin': sum(1 for c in text if ('\u0041' <= c <= '\u005A') or ('\u0061' <= c <= '\u007A')),
        'chinese': sum(1 for c in text if ('\u4E00' <= c <= '\u9FFF')),
        'japanese': sum(1 for c in text if ('\u3040' <= c <= '\u309F') or ('\u30A0' <= c <= '\u30FF')),
        'korean': sum(1 for c in text if '\uAC00' <= c <= '\uD7A3'),
    }
    counts['other'] = total - sum(counts.values())
    
    return {script: count/total for script, count in counts.items()}

def thai_number_to_text(number: Union[int, float]) -> str:
    """Convert number to Thai words"""
    if isinstance(number, float):
        int_part = int(number)
        decimal_part = round(number - int_part, 6)
        if decimal_part == 0:
            return thai_number_to_text(int_part)
        return thai_number_to_text(int_part) + "จุด" + thai_number_to_text(int(decimal_part * 1000000))

    if number < 0:
        return "ลบ" + thai_number_to_text(-number)
        
    if number == 0:
        return THAI_NUMBERS[0]
        
    if number < 10:
        return THAI_NUMBERS[number]
        
    if number < 100:
        tens = (number // 10) * 10
        units = number % 10
        if tens == 10:
            if units == 1:
                return "สิบเอ็ด"
            elif units == 0:
                return THAI_NUMBERS[tens]
            else:
                return "สิบ" + THAI_NUMBERS[units]
        if units == 0:
            return THAI_NUMBERS[tens]
        elif units == 1:
            return THAI_NUMBERS[tens] + "เอ็ด"
        return THAI_NUMBERS[tens] + THAI_NUMBERS[units]
        
    if number < 1000:
        hundreds = number // 100
        remainder = number % 100
        if remainder == 0:
            return THAI_NUMBERS[hundreds] + THAI_NUMBERS[100]
        return THAI_NUMBERS[hundreds] + THAI_NUMBERS[100] + thai_number_to_text(remainder)
        
    for unit in [1000000, 100000, 10000, 1000]:
        if number >= unit:
            main = number // unit
            remainder = number % unit
            if remainder == 0:
                return thai_number_to_text(main) + THAI_NUMBERS[unit]
            return thai_number_to_text(main) + THAI_NUMBERS[unit] + thai_number_to_text(remainder)
            
    return str(number)  # Fallback for very large numbers

def thai_digit_to_arabic_digit(text: str) -> str:
    """Convert Thai digits to Arabic digits"""
    thai_digits = '๐๑๒๓๔๕๖๗๘๙'  
    arabic_digits = '0123456789'
    trans = str.maketrans(thai_digits, arabic_digits)
    return text.translate(trans)

def arabic_digit_to_thai_digit(text: str) -> str:
    """Convert Arabic digits to Thai digits"""
    arabic_digits = '0123456789'
    thai_digits = '๐๑๒๓๔๕๖๗๘๙'
    trans = str.maketrans(arabic_digits, thai_digits)
    return text.translate(trans)

def split_thai_sentences(text: str) -> List[str]:
    """Split Thai text into sentences using multiple criteria"""
    # Common Thai sentence endings
    endings = {
        # Spaces and newlines
        ' ', '\n', '\r', '\t',
        # Thai specific
        'ครับ', 'ค่ะ', 'จ้า', 'จ้ะ', 'นะ', 'จ๊ะ',
        # Punctuation (Thai and universal)
        '।', '॥', '၊', '။', '።', '。', '︒', '﹒', '．', '｡',
        '?', '!', '?', '！', '?', '？', '…', '฿',
        # Combinations
        'นะคะ', 'นะครับ', 'นะจ๊ะ', 'นะจ้า',
        # Additional Thai endings
        'ละ', 'ล่ะ', 'เลย', 'ด้วย', 'สิ', 'น่ะ',
        # Formal endings
        'ดังนั้น', 'ฉะนั้น', 'เพราะฉะนั้น', 'อย่างไรก็ตาม',
        'กล่าวคือ', 'ได้แก่', 'ตัวอย่างเช่น'
    }
    
    # Common abbreviations to avoid false splits
    abbrevs = {
        'น.ส.', 'ด.ช.', 'ด.ญ.', 'นาย', 'นาง', 'ร.ศ.', 'ศ.', 'ดร.',
        'กม.', 'ตร.', 'จ.', 'อ.', 'ต.', 'ถ.', 'พ.ศ.', 'ม.ค.', 'ก.พ.',
        'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.', 'ก.ค.', 'ส.ค.', 'ก.ย.',
        'ต.ค.', 'พ.ย.', 'ธ.ค.',
        # Additional common abbreviations
        'มหา.', 'รร.', 'รพ.', 'ธ.', 'ธอส.', 'กทม.', 'สนง.',
        'บจก.', 'หจก.', 'บมจ.', 'จก.', 'ศูนย์ฯ', 'ฯลฯ', 'ฯพณฯ'
    }
    
    sentences = []
    current = []
    words = text.split()
    
    i = 0
    while i < len(words):
        word = words[i]
        current.append(word)
        
        # Check if current word is an abbreviation
        is_abbrev = any(word.startswith(abbr) for abbr in abbrevs)
        
        # Check for sentence endings
        has_ending = any(word.endswith(end) for end in endings)
        
        if has_ending and not is_abbrev:
            # Additional check for false sentence breaks
            next_word = words[i+1] if i+1 < len(words) else None
            if next_word and any(next_word.startswith(w.lower()) for w in ['และ', 'หรือ', 'แต่']):
                i += 1
                continue
                
            sentences.append(' '.join(current).strip())
            current = []
        i += 1
        
    if current:
        sentences.append(' '.join(current).strip())
        
    return [s for s in sentences if s]

def count_thai_characters(text: str) -> Dict[str, int]:
    """Count Thai character types"""
    counts = defaultdict(int)
    for char in text:
        if re.match(f'[{THAI_CONSONANTS}]', char):
            counts['consonants'] += 1
        elif re.match(f'[{THAI_VOWELS}]', char):
            counts['vowels'] += 1
        elif re.match(f'[{THAI_TONEMARKS}]', char):
            counts['tonemarks'] += 1
        elif re.match(f'[{THAI_DIGITS}]', char):
            counts['digits'] += 1
        elif re.match(f'[{THAI_CHARS}]', char):
            counts['other'] += 1
    return dict(counts)

def get_thai_character_types() -> Dict[str, str]:
    """Get mapping of Thai characters to their types"""
    return {
        'consonants': THAI_CONSONANTS,
        'vowels': THAI_VOWELS,
        'tonemarks': THAI_TONEMARKS,
        'digits': THAI_DIGITS,
        'all': THAI_CHARS
    }

def get_thai_syllable_pattern() -> str:
    """Get regex pattern for matching Thai syllables
    
    The pattern follows Thai syllable structure:
    - Initial consonant(s)
    - Optional leading vowel
    - Optional following vowel
    - Optional final consonant
    - Optional tone mark
    """
    return (
        f'[{THAI_CONSONANTS}]'  # Initial consonant
        f'(?:[{THAI_VOWELS}])?'  # Optional vowel
        f'(?:[{THAI_CONSONANTS}])?'  # Optional final consonant
        f'(?:[{THAI_TONEMARKS}])?'  # Optional tone mark
    )

def is_valid_thai_word(word: str) -> bool:
    """Check if a word follows valid Thai syllable patterns"""
    syllable_pattern = get_thai_syllable_pattern()
    # Word should consist of one or more valid syllables
    pattern = f'^{syllable_pattern}+$'
    return bool(re.match(pattern, word))
