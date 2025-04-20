"""
Error correction agent for handling malformed Thai text input
"""
from typing import Dict, List, Any, Tuple, Optional
import re
from difflib import get_close_matches

class ThaiErrorCorrectionAgent:
    """Agent for correcting various types of Thai text errors"""
    
    def __init__(self):
        # Common keyboard layout patterns (English -> Thai)
        self.en_th_map = {
            'l;ylfu': 'สวัสดี',  # สวัสดี
            'k;k': 'ค่ะ',      # ค่ะ
            'k8': 'ครับ',     # ครับ
            '8iy': 'ครับ',     # ครับ
            ';yl': 'สวย',      # สวย
            'pyh': 'ไทย',      # ไทย
            'mt': 'ใช่',       # ใช่
            'mkl': 'ไหม'      # ไหม
        }
        
        # Common Thai typos and variations
        self.common_errors = {
            # พิมพ์ผิด
            'เเ': 'แ',
            'กั': 'ก็',
            'เรือง': 'เรื่อง',
            'ปน': 'เป็น',
            'แปน': 'เป็น',
            'ทีเกิด': 'ที่เกิด',
            'เเละ': 'และ',
            'งัย': 'ไง',
            'ขึน': 'ขึ้น',
            
            # สระและวรรณยุกต์
            'ํา': 'ำ',
            'ะื': 'ื',
            'ั้': '้ั',
            
            # คำที่มักเขียนผิด
            'เช่น กัน': 'เช่นกัน',
            'โดย เฉพาะ': 'โดยเฉพาะ',
            'ถ้า หาก': 'ถ้าหาก'
        }
        
        # Social media style variations
        self.social_variations = {
            'จ้าาา': 'จ้า',
            'ค่าาาา': 'ค่ะ',
            'น้าาาา': 'นะ',
            'อ่าาา': 'อ่ะ',
            'ครับบบ': 'ครับ',
            '555+': 'ฮฮฮฮฮ',
            '55555': 'ฮฮฮฮฮ',
            '5555': 'ฮฮฮฮ',
            '555': 'ฮฮฮ'
        }
        
        # Protected patterns (should not be modified)
        self.protected_patterns = {
            'ฮฮฮ': True,  # ไม่ควรแก้ไขเพิ่มเติม
            'ฮฮฮฮ': True,
            'ฮฮฮฮฮ': True,
            'จ้า': True,
            'ค่ะ': True,
            'ครับ': True,
            'เป็น': True
        }
        
    def correct_text(self, text: str, aggressive: bool = False) -> Dict[str, Any]:
        """
        Correct errors in Thai text
        
        Args:
            text: Input text
            aggressive: Whether to apply aggressive corrections
            
        Returns:
            Dict containing corrected text and correction details
        """
        if not text or not isinstance(text, str):
            return {'corrected': '', 'corrections': [], 'confidence': 1.0}
            
        original = text
        corrections = []
        
        # Split into words for processing
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Check for keyboard layout patterns
            if word in self.en_th_map:
                thai_word = self.en_th_map[word]
                corrections.append({
                    'type': 'keyboard_layout',
                    'original': word,
                    'corrected': thai_word
                })
                corrected_words.append(thai_word)
                continue
                
            # Fix common errors
            corrected = word
            for error, correction in self.common_errors.items():
                if error in corrected:
                    corrections.append({
                        'type': 'common_error',
                        'original': error,
                        'corrected': correction
                    })
                    corrected = corrected.replace(error, correction)
                    
            # Handle social media variations if aggressive mode is on
            if aggressive:
                for variation, standard in self.social_variations.items():
                    variation_pattern = variation.replace('+', r'\+')
                    if re.search(variation_pattern, corrected):
                        corrections.append({
                            'type': 'social_variation',
                            'original': corrected,
                            'corrected': standard
                        })
                        corrected = re.sub(variation_pattern, standard, corrected)
                        
            # Fix repeated characters unless word is protected
            if corrected not in self.protected_patterns:
                orig_len = len(corrected)
                corrected = self._fix_repeated_chars(corrected)
                if len(corrected) != orig_len:
                    corrections.append({
                        'type': 'word_boundary',
                        'original': word,
                        'corrected': corrected
                    })
                    
            corrected_words.append(corrected)
            
        # Combine corrected words
        text = ' '.join(corrected_words)
        
        # Calculate confidence
        confidence = self._calculate_confidence(corrections)
        
        return {
            'corrected': text,
            'corrections': corrections,
            'confidence': confidence
        }
        
    def _fix_repeated_chars(self, word: str) -> str:
        """Fix repeated characters while preserving important patterns"""
        if word in self.protected_patterns:
            return word
            
        # ตรวจสอบรูปแบบการพิมพ์ "555" หรือ "ฮฮฮ"
        if re.match(r'^5+$', word):
            length = len(word)
            if length >= 5:
                return 'ฮฮฮฮฮ'
            elif length == 4:
                return 'ฮฮฮฮ'
            elif length == 3:
                return 'ฮฮฮ'
            return word
            
        # Fix repeated consonants (ยกเว้น ฮ)
        word = re.sub(r'([ก-ศษสห-ฮ])\1{2,}', r'\1', word)
        
        # Fix repeated vowels (ประยุกต์ตามบริบท)
        word = re.sub(r'([ะาิีึืุูเแโใไ])\1{3,}', r'\1\1', word)
        
        # Fix repeated tone marks (เก็บไว้ 1)
        word = re.sub(r'([่้๊๋])\1+', r'\1', word)
        
        return word
        
    def analyze_errors(self, text: str) -> Dict[str, Any]:
        """
        Analyze types of errors in text
        
        Args:
            text: Input text
            
        Returns:
            Dict containing error analysis
        """
        analysis = {
            'error_types': {},
            'suggestions': [],
            'severity': 'low'
        }
        
        # Check for keyboard layout issues
        if re.search(r'[a-zA-Z]', text):
            analysis['error_types']['keyboard_layout'] = True
            analysis['suggestions'].append('Text appears to be typed with wrong keyboard layout')
            analysis['severity'] = 'high'
            
        # Check for repeated characters
        repeats = len(re.findall(r'(.)\1{2,}', text))
        if repeats > 0:
            analysis['error_types']['character_repetition'] = repeats
            analysis['suggestions'].append('Contains excessive character repetition')
            
        # Check for mixed scripts
        if re.search(r'[a-zA-Z]', text) and re.search(r'[ก-๙]', text):
            analysis['error_types']['mixed_script'] = True
            analysis['suggestions'].append('Contains mixed Thai and English characters')
            
        return analysis
        
    def _calculate_confidence(self, corrections: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for corrections
        
        Args:
            corrections: List of applied corrections
            
        Returns:
            Confidence score between 0 and 1
        """
        if not corrections:
            return 1.0
            
        # Weight different correction types
        weights = {
            'keyboard_layout': 0.9,    # Direct mapping from dictionary
            'keyboard_mapping': 0.7,    # Character by character mapping
            'common_error': 0.8,       # Common spelling errors
            'social_variation': 0.85,   # Social media style
            'word_boundary': 0.75      # Word boundary issues
        }
        
        # Calculate weighted average confidence
        total_weight = 0
        total_score = 0
        
        for correction in corrections:
            weight = weights.get(correction['type'], 0.5)
            total_weight += weight
            total_score += weight
            
        # If no valid corrections, return base confidence
        if total_weight == 0:
            return 0.5
            
        return total_score / total_weight
