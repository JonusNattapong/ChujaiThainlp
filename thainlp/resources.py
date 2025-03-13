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

# Load dictionaries from JSON files
def load_dictionary(filename: str) -> Dict:
    """Load dictionary from JSON file"""
    try:
        with open(os.path.join(os.path.dirname(__file__), 'data', filename), 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Thai word dictionary
THAI_WORDS: Dict[str, bool] = {
    # Common words
    'สวัสดี': True, 'ครับ': True, 'ค่ะ': True, 'ขอบคุณ': True,
    'ยินดี': True, 'ต้อนรับ': True, 'บ้าน': True, 'รถ': True,
    'คน': True, 'หนังสือ': True, 'อาหาร': True, 'น้ำ': True,
    'อากาศ': True, 'กิน': True, 'เดิน': True, 'นอน': True,
    'อ่าน': True, 'เขียน': True, 'พูด': True, 'คิด': True,
    
    # Additional words
    'โรงเรียน': True, 'มหาวิทยาลัย': True, 'โรงพยาบาล': True,
    'ร้านอาหาร': True, 'ตลาด': True, 'ห้างสรรพสินค้า': True,
    'สวนสาธารณะ': True, 'สนามกีฬา': True, 'โรงภาพยนตร์': True,
    'สถานีรถไฟ': True, 'สนามบิน': True, 'ท่าเรือ': True,
    
    # Emotions
    'สุข': True, 'ดีใจ': True, 'ยิ้ม': True, 'หัวเราะ': True,
    'รัก': True, 'ชอบ': True, 'สนุก': True, 'ตื่นเต้น': True,
    'เศร้า': True, 'เสียใจ': True, 'โกรธ': True, 'กลัว': True,
    'กังวล': True, 'เบื่อ': True, 'เหนื่อย': True, 'หิว': True,
    
    # Colors
    'แดง': True, 'น้ำเงิน': True, 'เขียว': True, 'เหลือง': True,
    'ขาว': True, 'ดำ': True, 'ม่วง': True, 'ส้ม': True,
    'ชมพู': True, 'เทา': True, 'น้ำตาล': True, 'ฟ้า': True,
    
    # Numbers
    'หนึ่ง': True, 'สอง': True, 'สาม': True, 'สี่': True,
    'ห้า': True, 'หก': True, 'เจ็ด': True, 'แปด': True,
    'เก้า': True, 'สิบ': True, 'ร้อย': True, 'พัน': True,
    
    # Time
    'วัน': True, 'เดือน': True, 'ปี': True, 'สัปดาห์': True,
    'เช้า': True, 'กลางวัน': True, 'เย็น': True, 'กลางคืน': True,
    'วานนี้': True, 'วันนี้': True, 'พรุ่งนี้': True,
    
    # Directions
    'เหนือ': True, 'ใต้': True, 'ตะวันออก': True, 'ตะวันตก': True,
    'ซ้าย': True, 'ขวา': True, 'บน': True, 'ล่าง': True,
    'ใกล้': True, 'ไกล': True, 'ตรง': True, 'รอบ': True,
}

# Thai sentiment dictionary with more words and weights
THAI_SENTIMENT_DICT: Dict[str, float] = {
    # Positive words (weight: 1.0)
    'ดี': 1.0, 'สวย': 1.0, 'น่ารัก': 1.0, 'เยี่ยม': 1.0,
    'สุดยอด': 1.0, 'ยอดเยี่ยม': 1.0, 'ดีมาก': 1.0,
    'มหัศจรรย์': 1.0, 'วิเศษ': 1.0, 'เลิศ': 1.0,
    
    # Positive words (weight: 0.8)
    'ชอบ': 0.8, 'รัก': 0.8, 'สนุก': 0.8, 'สุข': 0.8,
    'ดีใจ': 0.8, 'ยิ้ม': 0.8, 'หัวเราะ': 0.8,
    'ตื่นเต้น': 0.8, 'สดใส': 0.8, 'สดชื่น': 0.8,
    
    # Positive words (weight: 0.6)
    'พอใจ': 0.6, 'สบาย': 0.6, 'เรียบง่าย': 0.6,
    'สะอาด': 0.6, 'เป็นระเบียบ': 0.6, 'เรียบร้อย': 0.6,
    
    # Negative words (weight: -1.0)
    'แย่': -1.0, 'น่าเกลียด': -1.0, 'โทรม': -1.0,
    'แย่': -1.0, 'แย่': -1.0, 'แย่': -1.0,
    'น่าขยะแขยง': -1.0, 'น่าอับอาย': -1.0,
    
    # Negative words (weight: -0.8)
    'เกลียด': -0.8, 'เบื่อ': -0.8, 'ท้อ': -0.8, 'เศร้า': -0.8,
    'เสียใจ': -0.8, 'โกรธ': -0.8, 'กลัว': -0.8,
    'กังวล': -0.8, 'หงุดหงิด': -0.8, 'รำคาญ': -0.8,
    
    # Negative words (weight: -0.6)
    'เหนื่อย': -0.6, 'หิว': -0.6, 'ง่วง': -0.6,
    'ปวด': -0.6, 'เจ็บ': -0.6, 'เมื่อย': -0.6,
}

# Thai POS dictionary with more words and tags
THAI_POS_DICT: Dict[str, str] = {
    # Nouns
    'บ้าน': 'NOUN', 'รถ': 'NOUN', 'คน': 'NOUN', 'หนังสือ': 'NOUN',
    'อาหาร': 'NOUN', 'น้ำ': 'NOUN', 'อากาศ': 'NOUN',
    'โรงเรียน': 'NOUN', 'มหาวิทยาลัย': 'NOUN', 'โรงพยาบาล': 'NOUN',
    'ร้านอาหาร': 'NOUN', 'ตลาด': 'NOUN', 'ห้างสรรพสินค้า': 'NOUN',
    
    # Verbs
    'กิน': 'VERB', 'เดิน': 'VERB', 'นอน': 'VERB', 'อ่าน': 'VERB',
    'เขียน': 'VERB', 'พูด': 'VERB', 'คิด': 'VERB',
    'วิ่ง': 'VERB', 'ว่าย': 'VERB', 'กระโดด': 'VERB',
    'ร้องเพลง': 'VERB', 'เต้น': 'VERB', 'เล่น': 'VERB',
    
    # Adjectives
    'สวย': 'ADJ', 'ดี': 'ADJ', 'ใหญ่': 'ADJ', 'เล็ก': 'ADJ',
    'ร้อน': 'ADJ', 'เย็น': 'ADJ', 'เร็ว': 'ADJ',
    'สว่าง': 'ADJ', 'มืด': 'ADJ', 'สูง': 'ADJ', 'ต่ำ': 'ADJ',
    'หนัก': 'ADJ', 'เบา': 'ADJ', 'แข็ง': 'ADJ', 'อ่อน': 'ADJ',
    
    # Adverbs
    'เร็ว': 'ADV', 'ช้า': 'ADV', 'ดี': 'ADV', 'มาก': 'ADV',
    'น้อย': 'ADV', 'มาก': 'ADV', 'น้อย': 'ADV',
    'ค่อยๆ': 'ADV', 'เร็วๆ': 'ADV', 'ช้าๆ': 'ADV',
    'อย่างดี': 'ADV', 'อย่างมาก': 'ADV', 'อย่างน้อย': 'ADV',
    
    # Pronouns
    'ผม': 'PRON', 'ฉัน': 'PRON', 'คุณ': 'PRON', 'เขา': 'PRON',
    'เธอ': 'PRON', 'มัน': 'PRON', 'พวกเขา': 'PRON',
    'ดิฉัน': 'PRON', 'กระผม': 'PRON', 'ท่าน': 'PRON',
    
    # Prepositions
    'ใน': 'PREP', 'บน': 'PREP', 'ใต้': 'PREP', 'ข้าง': 'PREP',
    'ระหว่าง': 'PREP', 'ใกล้': 'PREP', 'ไกล': 'PREP',
    'ข้างๆ': 'PREP', 'รอบ': 'PREP', 'ผ่าน': 'PREP',
    
    # Conjunctions
    'และ': 'CONJ', 'หรือ': 'CONJ', 'แต่': 'CONJ', 'เพราะ': 'CONJ',
    'ถ้า': 'CONJ', 'แม้ว่า': 'CONJ', 'จน': 'CONJ',
    'ทั้ง': 'CONJ', 'ทั้งที่': 'CONJ', 'ทั้งๆที่': 'CONJ',
    
    # Articles
    'นี้': 'DET', 'นั้น': 'DET', 'นั่น': 'DET', 'นี่': 'DET',
    'ทุก': 'DET', 'บาง': 'DET', 'หลาย': 'DET',
    'ทั้ง': 'DET', 'ทั้งหลาย': 'DET', 'ทั้งหมด': 'DET',
}

# Thai NER patterns with more entities
THAI_NER_PATTERNS: Dict[str, str] = {
    'PERSON': r'[ก-๛]+ (?:[ก-๛]+ )*[ก-๛]+',  # Thai name pattern
    'LOCATION': r'[ก-๛]+(?:[ก-๛]+)*',  # Location name pattern
    'ORGANIZATION': r'[ก-๛]+(?:[ก-๛]+)*',  # Organization name pattern
    'DATE': r'\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}',  # Date pattern
    'TIME': r'\d{1,2}:\d{2}',  # Time pattern
    'MONEY': r'\d+(?:,\d+)*(?:\.\d+)? (?:บาท|USD|EUR)',  # Money pattern
    'PHONE': r'\d{3}-\d{3}-\d{4}|\d{10}',  # Phone number pattern
    'EMAIL': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email pattern
    'URL': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)',  # URL pattern
    'HASHTAG': r'#[\w\u0E00-\u0E7F]+',  # Hashtag pattern
    'MENTION': r'@[\w\u0E00-\u0E7F]+',  # Mention pattern
}
