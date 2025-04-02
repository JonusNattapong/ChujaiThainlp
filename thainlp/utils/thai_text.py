"""
High-level Thai text processing interface
"""
from typing import List, Dict, Union, Optional
from thainlp.utils import thai_utils

class ThaiTextProcessor:
    """
    Easy-to-use interface for Thai text processing
    
    Example:
        processor = ThaiTextProcessor()
        result = (processor.load("สวัสดีครับ")
                 .normalize()
                 .to_roman()
                 .get_text())
    """
    
    def __init__(self):
        self.text = ""
        self._sentences: List[str] = []
        self._words: List[str] = []
        
    def load(self, text: str) -> 'ThaiTextProcessor':
        """Load text for processing"""
        self.text = text
        return self
        
    def normalize(self) -> 'ThaiTextProcessor':
        """Normalize Thai text"""
        self.text = thai_utils.normalize_text(self.text)
        return self
        
    def remove_tones(self) -> 'ThaiTextProcessor':
        """Remove tone marks"""
        self.text = thai_utils.remove_tone_marks(self.text)
        return self
        
    def remove_diacritics(self) -> 'ThaiTextProcessor':
        """Remove all diacritics"""
        self.text = thai_utils.remove_diacritics(self.text)
        return self
        
    def to_roman(self) -> 'ThaiTextProcessor':
        """Convert to roman alphabet"""
        self.text = thai_utils.thai_to_roman(self.text)
        return self
        
    def to_arabic_digits(self) -> 'ThaiTextProcessor':
        """Convert Thai digits to Arabic digits"""
        self.text = thai_utils.thai_digit_to_arabic_digit(self.text)
        return self
        
    def to_thai_digits(self) -> 'ThaiTextProcessor':
        """Convert Arabic digits to Thai digits"""
        self.text = thai_utils.arabic_digit_to_thai_digit(self.text)
        return self
        
    def extract_thai(self) -> 'ThaiTextProcessor':
        """Extract only Thai characters"""
        self.text = thai_utils.extract_thai_text(self.text)
        return self
        
    def split_sentences(self) -> 'ThaiTextProcessor':
        """Split text into sentences"""
        self._sentences = thai_utils.split_thai_sentences(self.text)
        return self
        
    def get_text(self) -> str:
        """Get processed text"""
        return self.text
        
    def get_sentences(self) -> List[str]:
        """Get split sentences"""
        if not self._sentences:
            self.split_sentences()
        return self._sentences
        
    def get_script_ratios(self) -> Dict[str, float]:
        """Get ratio of different scripts in text"""
        return thai_utils.detect_language(self.text)
        
    def get_character_counts(self) -> Dict[str, int]:
        """Get counts of different Thai character types"""
        return thai_utils.count_thai_characters(self.text)
        
    def has_thai(self) -> bool:
        """Check if text contains Thai characters"""
        return thai_utils.is_thai_word(self.text)
        
    @staticmethod
    def number_to_thai(number: Union[int, float]) -> str:
        """Convert number to Thai words"""
        return thai_utils.thai_number_to_text(number)

class ThaiValidator:
    """
    Validator for Thai text and characters
    
    Example:
        validator = ThaiValidator()
        if validator.is_valid_word("สวัสดี"):
            print("Valid Thai word")
    """
    
    @staticmethod
    def is_thai_char(char: str) -> bool:
        """Check if character is Thai"""
        return thai_utils.is_thai_char(char)
        
    @staticmethod
    def is_valid_word(word: str) -> bool:
        """Check if word follows Thai syllable patterns"""
        return thai_utils.is_valid_thai_word(word)
        
    @staticmethod
    def get_syllable_pattern() -> str:
        """Get Thai syllable pattern"""
        return thai_utils.get_thai_syllable_pattern()
        
    @staticmethod
    def get_character_types() -> Dict[str, str]:
        """Get Thai character type ranges"""
        return thai_utils.get_thai_character_types()

def process_text(text: str) -> ThaiTextProcessor:
    """
    Quick way to start processing Thai text
    
    Example:
        result = process_text("สวัสดี").normalize().to_roman().get_text()
    """
    return ThaiTextProcessor().load(text)

# Optional: Pre-built processors for common tasks
normalize_text = lambda text: process_text(text).normalize().get_text()
romanize = lambda text: process_text(text).normalize().to_roman().get_text()
extract_thai = lambda text: process_text(text).extract_thai().get_text()
split_sentences = lambda text: process_text(text).normalize().get_sentences()
