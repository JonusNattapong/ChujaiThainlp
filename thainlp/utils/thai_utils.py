"""
Thai language utilities
"""

from typing import List, Dict, Set, Tuple, Optional, Union
import re
import warnings
import datetime

try:
    import pythainlp
    from pythainlp.util import normalize as pythainlp_normalize
    from pythainlp.util import thai_strftime
    from pythainlp.util import num_to_thaiword, thaiword_to_num
    from pythainlp.util import thai_digit_to_arabic, arabic_digit_to_thai
    from pythainlp.util import thai_time, thai_day2datetime
    from pythainlp.util import countthai, isthai
    from pythainlp.transliterate import romanize, transliterate
    from pythainlp.soundex import soundex
    from pythainlp.spell import spell
    from pythainlp.corpus import thai_stopwords, thai_words, thai_syllables
    from pythainlp.corpus import wordnet
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    warnings.warn("PyThaiNLP not found. Using simplified Thai utilities.")

# Thai character ranges
THAI_CHARS = '\u0E00-\u0E7F'
THAI_CONSONANTS = '\u0E01-\u0E2E'
THAI_VOWELS = '\u0E30-\u0E4E'
THAI_TONEMARKS = '\u0E48-\u0E4B'
THAI_DIGITS = '\u0E50-\u0E59'

def is_thai_char(char: str) -> bool:
    """
    Check if a character is a Thai character
    
    Args:
        char (str): Character to check
        
    Returns:
        bool: True if character is Thai
    """
    if PYTHAINLP_AVAILABLE:
        return isthai(char)
    else:
        if len(char) != 1:
            return False
        return bool(re.match(f'[{THAI_CHARS}]', char))

def is_thai_word(word: str) -> bool:
    """
    Check if a word is a Thai word
    
    Args:
        word (str): Word to check
        
    Returns:
        bool: True if word contains Thai characters
    """
    if PYTHAINLP_AVAILABLE:
        return countthai(word) > 0
    else:
        return bool(re.search(f'[{THAI_CHARS}]', word))

def remove_tone_marks(text: str) -> str:
    """
    Remove tone marks from Thai text
    
    Args:
        text (str): Thai text
        
    Returns:
        str: Thai text without tone marks
    """
    if PYTHAINLP_AVAILABLE:
        return pythainlp_normalize(text, delete_tone=True)
    else:
        return re.sub(f'[{THAI_TONEMARKS}]', '', text)

def remove_diacritics(text: str) -> str:
    """
    Remove all diacritics from Thai text
    
    Args:
        text (str): Thai text
        
    Returns:
        str: Thai text without diacritics
    """
    if PYTHAINLP_AVAILABLE:
        return pythainlp_normalize(text)
    else:
        # Remove tone marks and other diacritics
        return re.sub(f'[{THAI_TONEMARKS}\u0E31\u0E34-\u0E3A\u0E47\u0E4C-\u0E4E]', '', text)

def normalize_text(text: str) -> str:
    """
    Normalize Thai text (remove excess spaces, normalize characters)
    
    Args:
        text (str): Thai text
        
    Returns:
        str: Normalized Thai text
    """
    if PYTHAINLP_AVAILABLE:
        return pythainlp_normalize(text)
    else:
        # Simple normalization
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.strip()
        return text

def count_thai_words(text: str) -> int:
    """
    Count Thai words in text (approximation)
    
    Args:
        text (str): Input text
        
    Returns:
        int: Number of Thai words
    """
    if PYTHAINLP_AVAILABLE:
        from pythainlp.tokenize import word_tokenize
        tokens = word_tokenize(text)
        thai_tokens = [token for token in tokens if isthai(token)]
        return len(thai_tokens)
    else:
        # This is a simple approximation
        # For better results, use a proper tokenizer
        words = text.split()
        thai_words = [word for word in words if is_thai_word(word)]
        return len(thai_words)

def extract_thai_text(text: str) -> str:
    """
    Extract only Thai text from mixed text
    
    Args:
        text (str): Mixed text
        
    Returns:
        str: Thai text only
    """
    if PYTHAINLP_AVAILABLE:
        result = ""
        for char in text:
            if isthai(char) or char.isspace():
                result += char
        return result
    else:
        # Extract Thai characters and spaces
        return ''.join(char for char in text if re.match(f'[{THAI_CHARS}]', char) or char.isspace())

def thai_to_roman(text: str, engine: str = "default") -> str:
    """
    Romanize Thai text
    
    Args:
        text (str): Thai text
        engine (str): Romanization engine (default, royin, thai2rom)
        
    Returns:
        str: Romanized text
    """
    if PYTHAINLP_AVAILABLE:
        return romanize(text, engine)
    else:
        # Very basic romanization (not accurate)
        # This is just a placeholder for when pythainlp is not available
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
        result = ""
        for char in text:
            if char in char_map:
                result += char_map[char]
            else:
                result += char
        return result

def detect_language(text: str) -> str:
    """
    Detect if text is Thai, English, or mixed
    
    Args:
        text (str): Text to detect
        
    Returns:
        str: 'th' for Thai, 'en' for English, 'mixed' for mixed
    """
    if PYTHAINLP_AVAILABLE:
        thai_count = sum(1 for char in text if isthai(char))
    else:
        thai_count = sum(1 for char in text if re.match(f'[{THAI_CHARS}]', char))
        
    eng_count = sum(1 for char in text if char.isalpha() and not re.match(f'[{THAI_CHARS}]', char))
    
    if thai_count > 0 and eng_count == 0:
        return 'th'
    elif eng_count > 0 and thai_count == 0:
        return 'en'
    elif thai_count > 0 and eng_count > 0:
        return 'mixed'
    else:
        return 'other'

# เพิ่มฟังก์ชันขั้นสูงที่ใช้ pythainlp

def thai_number_to_text(number: Union[int, float, str]) -> str:
    """
    แปลงตัวเลขเป็นคำอ่านภาษาไทย
    
    Args:
        number (Union[int, float, str]): ตัวเลขที่ต้องการแปลง
        
    Returns:
        str: คำอ่านภาษาไทย
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("ต้องติดตั้ง PyThaiNLP เพื่อใช้ฟังก์ชันนี้")
    
    return num_to_thaiword(number)

def thai_text_to_number(text: str) -> Union[int, float]:
    """
    แปลงคำอ่านภาษาไทยเป็นตัวเลข
    
    Args:
        text (str): คำอ่านภาษาไทย
        
    Returns:
        Union[int, float]: ตัวเลข
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("ต้องติดตั้ง PyThaiNLP เพื่อใช้ฟังก์ชันนี้")
    
    return thaiword_to_num(text)

def format_thai_date(date_obj, format_str: str = "%c") -> str:
    """
    จัดรูปแบบวันที่เป็นภาษาไทย
    
    Args:
        date_obj: วัตถุวันที่ (datetime.datetime หรือ datetime.date)
        format_str (str): รูปแบบการจัดรูปแบบ (เหมือน strftime)
        
    Returns:
        str: วันที่ในรูปแบบภาษาไทย
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("ต้องติดตั้ง PyThaiNLP เพื่อใช้ฟังก์ชันนี้")
    
    return thai_strftime(date_obj, format_str)

def thai_soundex(text: str, engine: str = "lk82") -> str:
    """
    คำนวณรหัส Soundex สำหรับคำภาษาไทย
    
    Args:
        text (str): คำภาษาไทย
        engine (str): อัลกอริทึม Soundex (lk82, metasound, udom83)
        
    Returns:
        str: รหัส Soundex
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("ต้องติดตั้ง PyThaiNLP เพื่อใช้ฟังก์ชันนี้")
    
    return soundex(text, engine)

def spell_correction(text: str) -> List[Tuple[str, float]]:
    """
    แก้ไขคำผิดในภาษาไทย
    
    Args:
        text (str): คำที่อาจจะสะกดผิด
        
    Returns:
        List[Tuple[str, float]]: รายการคำที่อาจจะถูกต้องพร้อมคะแนนความมั่นใจ
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("ต้องติดตั้ง PyThaiNLP เพื่อใช้ฟังก์ชันนี้")
    
    return spell(text)

def get_thai_stopwords() -> Set[str]:
    """
    รับรายการคำหยุดภาษาไทย
    
    Returns:
        Set[str]: ชุดของคำหยุดภาษาไทย
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("ต้องติดตั้ง PyThaiNLP เพื่อใช้ฟังก์ชันนี้")
    
    return thai_stopwords()

def get_thai_syllables() -> Set[str]:
    """
    รับรายการพยางค์ภาษาไทย
    
    Returns:
        Set[str]: ชุดของพยางค์ภาษาไทย
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("ต้องติดตั้ง PyThaiNLP เพื่อใช้ฟังก์ชันนี้")
    
    return thai_syllables()

def get_thai_wordnet_synsets(word: str) -> List:
    """
    รับ synsets จาก WordNet ภาษาไทย
    
    Args:
        word (str): คำภาษาไทย
        
    Returns:
        List: รายการ synsets
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("ต้องติดตั้ง PyThaiNLP เพื่อใช้ฟังก์ชันนี้")
    
    return wordnet.synsets(word)

def get_thai_wordnet_synonyms(word: str) -> List[str]:
    """
    รับคำที่มีความหมายเหมือนกันจาก WordNet ภาษาไทย
    
    Args:
        word (str): คำภาษาไทย
        
    Returns:
        List[str]: รายการคำที่มีความหมายเหมือนกัน
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("ต้องติดตั้ง PyThaiNLP เพื่อใช้ฟังก์ชันนี้")
    
    synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas(lang='tha'):
            synonyms.append(lemma.name())
    return list(set(synonyms))

def thai_digit_to_arabic_digit(text: str) -> str:
    """
    แปลงเลขไทยเป็นเลขอารบิก
    
    Args:
        text (str): ข้อความที่มีเลขไทย
        
    Returns:
        str: ข้อความที่มีเลขอารบิก
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("ต้องติดตั้ง PyThaiNLP เพื่อใช้ฟังก์ชันนี้")
    
    return thai_digit_to_arabic(text)

def arabic_digit_to_thai_digit(text: str) -> str:
    """
    แปลงเลขอารบิกเป็นเลขไทย
    
    Args:
        text (str): ข้อความที่มีเลขอารบิก
        
    Returns:
        str: ข้อความที่มีเลขไทย
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("ต้องติดตั้ง PyThaiNLP เพื่อใช้ฟังก์ชันนี้")
    
    return arabic_digit_to_thai(text)

def thai_time(time_str: str) -> str:
    """
    แปลงเวลาเป็นข้อความภาษาไทย
    
    Args:
        time_str (str): เวลาในรูปแบบ HH:MM
        
    Returns:
        str: เวลาในรูปแบบข้อความภาษาไทย
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("ต้องติดตั้ง PyThaiNLP เพื่อใช้ฟังก์ชันนี้")
    
    return thai_time(time_str)

def thai_day_to_datetime(text: str) -> 'datetime.datetime':
    """
    แปลงข้อความวันที่ภาษาไทยเป็นวัตถุ datetime
    
    Args:
        text (str): ข้อความวันที่ภาษาไทย เช่น "พรุ่งนี้", "วันจันทร์หน้า"
        
    Returns:
        datetime.datetime: วัตถุ datetime
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("ต้องติดตั้ง PyThaiNLP เพื่อใช้ฟังก์ชันนี้")
    
    return thai_day2datetime(text) 