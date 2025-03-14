"""
Maximum Matching Tokenization for Thai text
"""

from typing import List, Dict, Set, Optional
import re
import warnings

try:
    import pythainlp
    from pythainlp.tokenize import word_tokenize as pythainlp_tokenize
    from pythainlp.tokenize import Tokenizer
    from pythainlp.corpus import thai_words
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    warnings.warn("PyThaiNLP not found. Using simplified tokenization.")

# Dictionary for Maximum Matching algorithm
# This is a simplified dictionary for demonstration
# In a real system, use a more comprehensive dictionary
_THAI_WORDS = {
    "สวัสดี": True,
    "ประเทศ": True,
    "ไทย": True,
    "คน": True,
    "กิน": True,
    "ข้าว": True,
    "น้ำ": True,
    "รถ": True,
    "บ้าน": True,
    "เมือง": True,
    "จังหวัด": True,
    "เชียงใหม่": True,
    "กรุงเทพ": True,
    "ภูเก็ต": True,
    "ท่องเที่ยว": True,
    "เดินทาง": True,
    "นักวิจัย": True,
    "ศึกษา": True,
    "ปรากฏการณ์": True,
    "ธรรมชาติ": True,
    "ซับซ้อน": True,
    "การประชุม": True,
    "วิชาการ": True,
    "นานาชาติ": True,
    "เศรษฐกิจ": True,
    "ฟื้นตัว": True,
    "อย่าง": True,
    "ช้า": True,
    "รวดเร็ว": True,
    "ปัญญาประดิษฐ์": True,
    "เทคโนโลยี": True,
    "เปลี่ยนแปลง": True,
    "อุตสาหกรรม": True,
    "การแพทย์": True,
    "การเงิน": True,
    "การศึกษา": True,
    "การขนส่ง": True,
    "การท่องเที่ยว": True,
    "การเกษตร": True,
    "การสื่อสาร": True,
    "การพัฒนา": True,
    "การวิจัย": True,
    "การค้า": True,
    "การลงทุน": True,
    "การผลิต": True,
    "การบริโภค": True,
    "การส่งออก": True,
    "การนำเข้า": True,
    "การแข่งขัน": True,
    "การเติบโต": True,
    "การพัฒนา": True,
    "การปรับปรุง": True,
    "การเปลี่ยนแปลง": True,
    "การเรียนรู้": True,
    "การสอน": True,
    "การฝึกอบรม": True,
    "การทดสอบ": True,
    "การทดลอง": True,
    "การวิเคราะห์": True,
    "การสังเคราะห์": True,
    "การประเมิน": True,
    "การตรวจสอบ": True,
    "การติดตาม": True,
    "การควบคุม": True,
    "การจัดการ": True,
    "การบริหาร": True,
    "การวางแผน": True,
    "การดำเนินการ": True,
    "การปฏิบัติ": True,
    "การทำงาน": True,
    "การใช้งาน": True,
    "การพัฒนา": True,
    "การออกแบบ": True,
    "การสร้าง": True,
    "การผลิต": True,
    "การประกอบ": True,
    "การติดตั้ง": True,
    "การบำรุงรักษา": True,
    "การซ่อมแซม": True,
    "การทดสอบ": True,
    "การตรวจสอบ": True,
    "การรับรอง": True,
    "การรับประกัน": True,
    "การขาย": True,
    "การตลาด": True,
    "การโฆษณา": True,
    "การประชาสัมพันธ์": True,
    "การบริการ": True,
    "การสนับสนุน": True,
    "การช่วยเหลือ": True,
    "การแก้ไข": True,
    "การปรับปรุง": True,
    "การพัฒนา": True,
    "การเพิ่ม": True,
    "การลด": True,
    "การขยาย": True,
    "การหด": True,
    "การเติบโต": True,
    "การถดถอย": True,
    "การฟื้นตัว": True,
    "การล่ม": True,
    "การล้ม": True,
    "การเกิด": True,
    "การตาย": True,
    "การเริ่ม": True,
    "การจบ": True,
    "การเปิด": True,
    "การปิด": True,
    "การเข้า": True,
    "การออก": True,
    "การขึ้น": True,
    "การลง": True,
    "การไป": True,
    "การมา": True,
    "การถึง": True,
    "การกลับ": True,
    "การหยุด": True,
    "การพัก": True,
    "การนอน": True,
    "การตื่น": True,
    "การกิน": True,
    "การดื่ม": True,
    "การเล่น": True,
    "การทำงาน": True,
    "การเรียน": True,
    "การสอน": True,
    "การอ่าน": True,
    "การเขียน": True,
    "การพูด": True,
    "การฟัง": True,
    "การดู": True,
    "การเห็น": True,
    "การคิด": True,
    "การรู้สึก": True,
    "การรับรู้": True,
    "การเข้าใจ": True,
    "การจำ": True,
    "การลืม": True,
    "การรัก": True,
    "การเกลียด": True,
    "การชอบ": True,
    "การไม่ชอบ": True,
    "การสุข": True,
    "การทุกข์": True,
    "การสบาย": True,
    "การเจ็บ": True,
    "การป่วย": True,
    "การหาย": True,
    "การเป็น": True,
    "การตาย": True,
}

def _get_thai_words_dict() -> Dict[str, bool]:
    """
    Get Thai words dictionary
    
    Returns:
        Dict[str, bool]: Dictionary of Thai words
    """
    if PYTHAINLP_AVAILABLE:
        return {word: True for word in thai_words()}
    else:
        return _THAI_WORDS

def _maximum_matching_tokenize(text: str, dictionary: Dict[str, bool]) -> List[str]:
    """
    Tokenize Thai text using Maximum Matching algorithm
    
    Args:
        text (str): Thai text to tokenize
        dictionary (Dict[str, bool]): Dictionary of Thai words
        
    Returns:
        List[str]: List of tokens
    """
    tokens = []
    i = 0
    
    while i < len(text):
        # Skip whitespace
        if text[i].isspace():
            tokens.append(text[i])
            i += 1
            continue
            
        # Skip non-Thai characters
        if not '\u0E00' <= text[i] <= '\u0E7F':
            # Extract non-Thai segment
            j = i
            while j < len(text) and not '\u0E00' <= text[j] <= '\u0E7F':
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
            
        # Maximum matching
        found = False
        for j in range(min(20, len(text) - i), 0, -1):  # Maximum word length is 20
            word = text[i:i+j]
            if word in dictionary:
                tokens.append(word)
                i += j
                found = True
                break
                
        # If no match found, use character as token
        if not found:
            tokens.append(text[i])
            i += 1
            
    return tokens

def tokenize(text: str, engine: str = "pythainlp") -> List[str]:
    """
    Tokenize Thai text into words
    
    Args:
        text (str): Thai text to tokenize
        engine (str): Tokenization engine
               - 'maximum_matching': Use Maximum Matching algorithm
               - 'pythainlp': Use PyThaiNLP tokenizer (recommended)
               - 'pythainlp:newmm': Use PyThaiNLP with newmm engine
               - 'pythainlp:longest': Use PyThaiNLP with longest matching
               - 'pythainlp:attacut': Use PyThaiNLP with attacut (neural)
               - 'pythainlp:ulmfit': Use PyThaiNLP with ULMFit (neural)
               - 'custom': Use custom dictionary with Maximum Matching
        
    Returns:
        List[str]: List of Thai tokens
    """
    if engine == "maximum_matching":
        dictionary = _get_thai_words_dict()
        return _maximum_matching_tokenize(text, dictionary)
    elif engine.startswith("pythainlp"):
        if not PYTHAINLP_AVAILABLE:
            warnings.warn("PyThaiNLP not available. Falling back to Maximum Matching.")
            dictionary = _get_thai_words_dict()
            return _maximum_matching_tokenize(text, dictionary)
            
        if ":" in engine:
            _, pythainlp_engine = engine.split(":", 1)
            return pythainlp_tokenize(text, engine=pythainlp_engine)
        else:
            # Default to newmm
            return pythainlp_tokenize(text, engine="newmm")
    elif engine == "custom":
        # Use custom dictionary with Maximum Matching
        dictionary = _get_thai_words_dict()
        return _maximum_matching_tokenize(text, dictionary)
    else:
        raise ValueError(f"Tokenization engine '{engine}' is not supported.")

def create_custom_tokenizer(custom_dict: Optional[Set[str]] = None) -> callable:
    """
    Create a custom tokenizer with a specified dictionary
    
    Args:
        custom_dict (Optional[Set[str]]): Custom dictionary of Thai words
        
    Returns:
        callable: Tokenizer function
    """
    if PYTHAINLP_AVAILABLE:
        if custom_dict:
            # Create custom tokenizer with PyThaiNLP
            custom_tokenizer = Tokenizer(custom_dict=custom_dict)
            return lambda text: custom_tokenizer.word_tokenize(text)
        else:
            # Use default PyThaiNLP tokenizer
            return lambda text: pythainlp_tokenize(text)
    else:
        # Use Maximum Matching with custom dictionary
        if custom_dict:
            dictionary = {word: True for word in custom_dict}
        else:
            dictionary = _get_thai_words_dict()
            
        return lambda text: _maximum_matching_tokenize(text, dictionary)

def word_tokenize_with_custom_dict(text: str, custom_dict: Set[str]) -> List[str]:
    """
    Tokenize Thai text using a custom dictionary
    
    Args:
        text (str): Thai text to tokenize
        custom_dict (Set[str]): Custom dictionary of Thai words
        
    Returns:
        List[str]: List of tokens
    """
    tokenizer = create_custom_tokenizer(custom_dict)
    return tokenizer(text)

def sentence_tokenize(text: str) -> List[str]:
    """
    Tokenize Thai text into sentences
    
    Args:
        text (str): Thai text to tokenize
        
    Returns:
        List[str]: List of sentences
    """
    if PYTHAINLP_AVAILABLE:
        from pythainlp.tokenize import sent_tokenize
        return sent_tokenize(text)
    else:
        # Simple sentence tokenization based on punctuation
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() + '.' for s in sentences if s.strip()]

def subword_tokenize(text: str, engine: str = "tcc") -> List[str]:
    """
    Tokenize Thai text into subwords
    
    Args:
        text (str): Thai text to tokenize
        engine (str): Subword tokenization engine
               - 'tcc': Thai Character Cluster
               - 'etcc': Enhanced Thai Character Cluster
               - 'syllable': Syllable segmentation
        
    Returns:
        List[str]: List of subwords
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("PyThaiNLP is required for subword tokenization")
        
    if engine == "tcc":
        from pythainlp.tokenize import tcc
        return tcc.segment(text)
    elif engine == "etcc":
        from pythainlp.tokenize import etcc
        return etcc.segment(text)
    elif engine == "syllable":
        from pythainlp.tokenize import syllable_tokenize
        return syllable_tokenize(text)
    else:
        raise ValueError(f"Subword tokenization engine '{engine}' is not supported") 