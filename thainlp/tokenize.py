"""
Word tokenization module for Thai text.
"""
import re
from typing import List, Optional, Dict, Any
import warnings
try:
    import pythainlp
    from pythainlp.tokenize import word_tokenize as pythainlp_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    warnings.warn("PyThaiNLP not found. Using simplified dictionary-based tokenization.")

try:
    import torch
    import transformers
    DEEPCUT_AVAILABLE = False  # Set to True when model is implemented
except ImportError:
    DEEPCUT_AVAILABLE = False
    warnings.warn("Deep learning packages not found. Neural tokenization unavailable.")

# A small dictionary for demonstration
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
    "ความเป็นส่วนตัว": True,
    "ความปลอดภัย": True,
    "จริยธรรม": True,
    "พัฒนา": True,
    "ท้าทาย": True,
     "ภาพยนตร์": True,
    "เรื่องนี้": True,
    "สนุก": True,
    "มาก": True,
}

def dict_word_tokenize(text: str, custom_dict: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Tokenize Thai text using dictionary-based maximum matching.

    Args:
        text: Thai text to tokenize
        custom_dict: Optional custom dictionary

    Returns:
        List of Thai tokens
    """
    if not text:
        return []

    word_dict = custom_dict if custom_dict is not None else _THAI_WORDS
    tokens = []
    i = 0

    while i < len(text):
        longest_match = None
        longest_length = 0

        # Try to match the longest possible word
        for j in range(min(20, len(text) - i), 0, -1):  # Max word length of 20
            word = text[i:i+j]
            if word in word_dict:
                longest_match = word
                longest_length = j
                break

        if longest_match:
            tokens.append(longest_match)
            i += longest_length
        else:
            # If no match in dictionary, take the character as a token
            tokens.append(text[i])
            i += 1

    return tokens

def pythainlp_tokenize_wrapper(text: str, engine: str = "newmm") -> List[str]:
    """
    Tokenize using PyThaiNLP's tokenizers.

    Args:
        text: Thai text to tokenize
        engine: PyThaiNLP engine to use ('newmm', 'longest', 'attacut').
                See PyThaiNLP documentation for available engines.

    Returns:
        List of Thai tokens
    """
    if PYTHAINLP_AVAILABLE:
        return pythainlp_tokenize(text, engine=engine)
    else:
        return dict_word_tokenize(text)

def neural_word_tokenize(text: str, model: str = "default") -> List[str]:
    """
    Tokenize Thai text using neural network models.

    Args:
        text: Thai text to tokenize
        model: Name of the neural model to use

    Returns:
        List of Thai tokens
    """
    if not DEEPCUT_AVAILABLE:
        raise RuntimeError("Neural tokenization requires torch and transformers packages.")

    # Placeholder for neural implementation
    # In a real implementation, this would load and use a transformer model
    return dict_word_tokenize(text)  # Fallback to dictionary method for now

def word_tokenize(text: str, engine: str = "pythainlp") -> List[str]:
    """
    Tokenize Thai text into words.

    Args:
        text: Thai text to tokenize
        engine: Tokenization engine.
               - 'dict': Simple dictionary-based.
               - 'pythainlp': Use PyThaiNLP tokenizer (recommended).  You can specify a specific
                 PyThaiNLP engine by passing `engine='pythainlp:<engine_name>'`. For example,
                 `engine='pythainlp:newmm'` (default), `engine='pythainlp:longest'`,
                 `engine='pythainlp:attacut'`. See PyThaiNLP documentation for available engines.
               - 'neural': Use neural network tokenization.

    Returns:
        List of Thai tokens
    """
    if engine == "dict":
        return dict_word_tokenize(text)
    elif engine.startswith("pythainlp"):
        if ":" in engine:
            _, pythainlp_engine = engine.split(":", 1)
            return pythainlp_tokenize_wrapper(text, engine=pythainlp_engine)
        else:
            return pythainlp_tokenize_wrapper(text)  # Default to newmm
    elif engine == "neural":
        return neural_word_tokenize(text)
    else:
        raise ValueError(f"Tokenization engine '{engine}' is not supported.")