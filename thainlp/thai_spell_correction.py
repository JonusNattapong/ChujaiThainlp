"""
Thai spell correction
"""
from typing import List, Tuple, Set
import re
from collections import defaultdict
from .utils.thai_utils import is_thai_word

class ThaiSpellChecker:
    def __init__(self):
        # Basic Thai words dictionary
        self.thai_words = {
            "การ", "กับ", "ก็", "ก่อน", "ขณะ", "ขึ้น", "ของ", "ครับ", "ค่ะ", 
            "ครั้ง", "ความ", "คือ", "จะ", "จัด", "จาก", "จึง", "ช่วง", "ซึ่ง",
            "ดัง", "ด้วย", "ด้าน", "ต้อง", "ถึง", "ต่างๆ", "ที่", "ทุก", "ทาง",
            "ทั้ง", "ทำ", "ที่สุด", "นี้", "นั้น", "นัก", "นั้น", "นี้", "นั้น",
            "ใน", "ให้", "หรือ", "และ", "แล้ว", "ว่า", "วัน", "ไว้", "ว่า", "เพื่อ",
            "เมื่อ", "เรา", "เริ่ม", "เลย", "เวลา", "ส่วน", "ส่ง", "ส่วน", "สามารถ",
            "สิ่ง", "หาก", "ออก", "อะไร", "อาจ", "อีก", "เขา", "เพียง", "เพราะ",
            "เปิด", "เป็น", "แบบ", "แต่", "เอง", "เอง", "เคย", "เคย", "เข้า", "เช่น",
            "เฉพาะ", "เคย", "เคย", "เคย", "เคย", "เคย"
        }
        
        # Common misspellings mapping
        self.misspellings = {
            "ครับ": ["ครัช", "ครั้บ", "ครั๊บ"],
            "ค่ะ": ["คั่", "คั๊", "คั้"],
            "ครับผม": ["ครัชผม", "ครั้บผม"],
            "สวัสดี": ["สวัสดิ์", "สวัสดีครับ", "สวัสดีค่ะ"]
        }
        
    def add_words(self, words: Set[str]):
        """Add custom words to dictionary"""
        self.thai_words.update(words)
        
    def is_correct(self, word: str) -> bool:
        """Check if word is spelled correctly"""
        return word in self.thai_words
        
    def suggest(self, word: str, max_suggestions: int = 3) -> List[Tuple[str, float]]:
        """Suggest corrections for misspelled word"""
        if self.is_correct(word):
            return []
            
        # Check common misspellings first
        for correct, wrongs in self.misspellings.items():
            if word in wrongs:
                return [(correct, 1.0)]
                
        # Generate edit distance suggestions
        suggestions = []
        for candidate in self.thai_words:
            if not is_thai_word(candidate):
                continue
                
            distance = self._edit_distance(word, candidate)
            if distance <= 2:  # Only consider close matches
                confidence = 1.0 - (distance / max(len(word), len(candidate)))
                suggestions.append((candidate, confidence))
                
        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]
        
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
