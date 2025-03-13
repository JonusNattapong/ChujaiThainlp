"""
Maximum Matching Algorithm for Thai Word Tokenization
"""

from typing import List, Set
import re

class ThaiTokenizer:
    def __init__(self, dictionary: Set[str]):
        """
        Initialize ThaiTokenizer with a dictionary
        
        Args:
            dictionary (Set[str]): Set of Thai words
        """
        self.dictionary = dictionary
        self.max_word_length = max(len(word) for word in dictionary)
        
    def _is_thai(self, char: str) -> bool:
        """Check if character is Thai"""
        return '\u0E00' <= char <= '\u0E7F'
    
    def _is_thai_word(self, word: str) -> bool:
        """Check if word contains Thai characters"""
        return any(self._is_thai(char) for char in word)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Thai text using maximum matching algorithm
        
        Args:
            text (str): Thai text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        tokens = []
        current_pos = 0
        
        while current_pos < len(text):
            # Skip non-Thai characters
            if not self._is_thai(text[current_pos]):
                current_pos += 1
                continue
                
            # Try to find longest matching word
            found = False
            for length in range(min(self.max_word_length, len(text) - current_pos), 0, -1):
                word = text[current_pos:current_pos + length]
                if word in self.dictionary:
                    tokens.append(word)
                    current_pos += length
                    found = True
                    break
            
            # If no word found, take single character
            if not found:
                tokens.append(text[current_pos])
                current_pos += 1
                
        return tokens

def create_dictionary() -> Set[str]:
    """
    Create a basic Thai dictionary
    
    Returns:
        Set[str]: Set of Thai words
    """
    # Basic dictionary - can be expanded
    words = {
        # Common words
        "ผม", "คุณ", "เขา", "เธอ", "เรา", "พวกเขา",
        "กิน", "นอน", "เดิน", "วิ่ง", "พูด", "ฟัง",
        "ดี", "ไม่ดี", "สวย", "น่าเกลียด", "ใหญ่", "เล็ก",
        
        # Numbers
        "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า", "สิบ",
        
        # Time
        "วัน", "เดือน", "ปี", "เช้า", "กลางวัน", "เย็น", "กลางคืน",
        
        # Colors
        "แดง", "เขียว", "น้ำเงิน", "เหลือง", "ขาว", "ดำ",
        
        # Emotions
        "สุข", "เศร้า", "โกรธ", "ดีใจ", "เสียใจ", "กลัว",
        
        # Directions
        "เหนือ", "ใต้", "ตะวันออก", "ตะวันตก",
        
        # Common suffixes
        "ครับ", "ค่ะ", "นะ", "ค่ะ", "ครับ", "นะคะ",
        
        # Common prefixes
        "ไม่", "จะ", "กำลัง", "เคย", "เคย",
    }
    
    return words

def tokenize(text: str) -> List[str]:
    """
    Tokenize Thai text using maximum matching algorithm
    
    Args:
        text (str): Thai text to tokenize
        
    Returns:
        List[str]: List of tokens
    """
    dictionary = create_dictionary()
    tokenizer = ThaiTokenizer(dictionary)
    return tokenizer.tokenize(text) 