"""
Edit Distance-based Spell Checking for Thai Text
"""

from typing import List, Dict, Tuple
from collections import defaultdict

class ThaiSpellChecker:
    def __init__(self):
        """Initialize ThaiSpellChecker with dictionary"""
        self.dictionary = {
            # Common words
            'ผม', 'คุณ', 'เขา', 'เธอ', 'เรา', 'พวกเขา',
            'กิน', 'นอน', 'เดิน', 'วิ่ง', 'พูด', 'ฟัง',
            'ดี', 'ไม่ดี', 'สวย', 'น่าเกลียด', 'ใหญ่', 'เล็ก',
            
            # Numbers
            'หนึ่ง', 'สอง', 'สาม', 'สี่', 'ห้า', 'หก', 'เจ็ด', 'แปด', 'เก้า', 'สิบ',
            
            # Time
            'วัน', 'เดือน', 'ปี', 'เช้า', 'กลางวัน', 'เย็น', 'กลางคืน',
            
            # Colors
            'แดง', 'เขียว', 'น้ำเงิน', 'เหลือง', 'ขาว', 'ดำ',
            
            # Emotions
            'สุข', 'เศร้า', 'โกรธ', 'ดีใจ', 'เสียใจ', 'กลัว',
            
            # Directions
            'เหนือ', 'ใต้', 'ตะวันออก', 'ตะวันตก',
            
            # Common suffixes
            'ครับ', 'ค่ะ', 'นะ', 'ค่ะ', 'ครับ', 'นะคะ',
            
            # Common prefixes
            'ไม่', 'จะ', 'กำลัง', 'เคย', 'เคย',
        }
        
        # Thai character mapping for similar sounds
        self.sound_mapping = {
            'ก': {'ข', 'ค', 'ฆ'},
            'ข': {'ก', 'ค', 'ฆ'},
            'ค': {'ก', 'ข', 'ฆ'},
            'ฆ': {'ก', 'ข', 'ค'},
            'จ': {'ฉ', 'ช', 'ฌ'},
            'ฉ': {'จ', 'ช', 'ฌ'},
            'ช': {'จ', 'ฉ', 'ฌ'},
            'ฌ': {'จ', 'ฉ', 'ช'},
            'ด': {'ต', 'ถ', 'ท', 'ธ', 'น'},
            'ต': {'ด', 'ถ', 'ท', 'ธ', 'น'},
            'ถ': {'ด', 'ต', 'ท', 'ธ', 'น'},
            'ท': {'ด', 'ต', 'ถ', 'ธ', 'น'},
            'ธ': {'ด', 'ต', 'ถ', 'ท', 'น'},
            'น': {'ด', 'ต', 'ถ', 'ท', 'ธ'},
            'บ': {'ป', 'พ', 'ฟ', 'ภ'},
            'ป': {'บ', 'พ', 'ฟ', 'ภ'},
            'พ': {'บ', 'ป', 'ฟ', 'ภ'},
            'ฟ': {'บ', 'ป', 'พ', 'ภ'},
            'ภ': {'บ', 'ป', 'พ', 'ฟ'},
            'ม': {'ว'},
            'ว': {'ม'},
            'ร': {'ล'},
            'ล': {'ร'},
            'ย': {'ญ'},
            'ญ': {'ย'},
        }
        
    def _is_thai(self, char: str) -> bool:
        """Check if character is Thai"""
        return '\u0E00' <= char <= '\u0E7F'
        
    def _is_thai_word(self, word: str) -> bool:
        """Check if word contains Thai characters"""
        return any(self._is_thai(char) for char in word)
        
    def _get_similar_chars(self, char: str) -> set:
        """
        Get set of similar sounding characters
        
        Args:
            char (str): Input character
            
        Returns:
            set: Set of similar characters
        """
        return self.sound_mapping.get(char, {char})
        
    def _edit_distance(self, word1: str, word2: str) -> int:
        """
        Calculate edit distance between two words
        
        Args:
            word1 (str): First word
            word2 (str): Second word
            
        Returns:
            int: Edit distance
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
            
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Check for similar sounds
                    similar_chars1 = self._get_similar_chars(word1[i-1])
                    similar_chars2 = self._get_similar_chars(word2[j-1])
                    
                    if similar_chars1 & similar_chars2:  # If there's overlap
                        dp[i][j] = dp[i-1][j-1] + 0.5  # Lower cost for similar sounds
                    else:
                        dp[i][j] = min(
                            dp[i-1][j] + 1,    # Deletion
                            dp[i][j-1] + 1,    # Insertion
                            dp[i-1][j-1] + 1   # Substitution
                        )
                        
        return dp[m][n]
        
    def _get_candidates(self, word: str, max_distance: int = 2) -> List[Tuple[str, float]]:
        """
        Get candidate corrections for a word
        
        Args:
            word (str): Input word
            max_distance (int): Maximum edit distance
            
        Returns:
            List[Tuple[str, float]]: List of (word, score) pairs
        """
        candidates = []
        
        for dict_word in self.dictionary:
            distance = self._edit_distance(word, dict_word)
            if distance <= max_distance:
                # Convert distance to score (lower distance = higher score)
                score = 1.0 / (1.0 + distance)
                candidates.append((dict_word, score))
                
        return sorted(candidates, key=lambda x: x[1], reverse=True)
        
    def check(self, text: str) -> List[Tuple[str, List[Tuple[str, float]]]]:
        """
        Check spelling in Thai text
        
        Args:
            text (str): Input text
            
        Returns:
            List[Tuple[str, List[Tuple[str, float]]]]: List of (word, candidates) pairs
        """
        words = text.split()
        results = []
        
        for word in words:
            if self._is_thai_word(word) and word not in self.dictionary:
                candidates = self._get_candidates(word)
                if candidates:
                    results.append((word, candidates))
                    
        return results

def check_spelling(text: str) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """
    Check spelling in Thai text
    
    Args:
        text (str): Input text
        
    Returns:
        List[Tuple[str, List[Tuple[str, float]]]]: List of (word, candidates) pairs
    """
    checker = ThaiSpellChecker()
    return checker.check(text) 