"""
Tests for Thai Utilities
"""

import unittest
from thainlp.utils.thai_utils import (
    is_thai_char,
    is_thai_word,
    remove_tone_marks,
    remove_diacritics,
    normalize_text,
    count_thai_words,
    extract_thai_text,
    thai_to_roman,
    detect_language
)

class TestThaiUtils(unittest.TestCase):
    def test_is_thai_char(self):
        """Test is_thai_char function"""
        self.assertTrue(is_thai_char('ก'))
        self.assertTrue(is_thai_char('ข'))
        self.assertTrue(is_thai_char('ค'))
        self.assertFalse(is_thai_char('a'))
        self.assertFalse(is_thai_char('b'))
        self.assertFalse(is_thai_char('c'))
        
    def test_is_thai_word(self):
        """Test is_thai_word function"""
        self.assertTrue(is_thai_word('สวัสดี'))
        self.assertTrue(is_thai_word('ไทย'))
        self.assertFalse(is_thai_word('hello'))
        self.assertTrue(is_thai_word('สวัสดีhello'))  # Mixed
        
    def test_remove_tone_marks(self):
        """Test remove_tone_marks function"""
        self.assertEqual(remove_tone_marks('สวัสดี'), 'สวสด')
        self.assertEqual(remove_tone_marks('ไทย'), 'ไทย')
        self.assertEqual(remove_tone_marks('ผู้'), 'ผ')
        
    def test_remove_diacritics(self):
        """Test remove_diacritics function"""
        self.assertEqual(remove_diacritics('สวัสดี'), 'สวสด')
        self.assertEqual(remove_diacritics('ไทย'), 'ไทย')
        self.assertEqual(remove_diacritics('ผู้'), 'ผ')
        
    def test_normalize_text(self):
        """Test normalize_text function"""
        self.assertEqual(normalize_text('  สวัสดี  '), 'สวัสดี')
        self.assertEqual(normalize_text('HELLO  สวัสดี'), 'hello สวัสดี')
        self.assertEqual(normalize_text('สวัสดี\n\nครับ'), 'สวัสดี ครับ')
        
    def test_count_thai_words(self):
        """Test count_thai_words function"""
        self.assertEqual(count_thai_words('สวัสดี ครับ'), 2)
        self.assertEqual(count_thai_words('สวัสดี hello ครับ'), 2)
        self.assertEqual(count_thai_words('hello world'), 0)
        
    def test_extract_thai_text(self):
        """Test extract_thai_text function"""
        self.assertEqual(extract_thai_text('สวัสดี hello ครับ'), 'สวัสดี  ครับ')
        self.assertEqual(extract_thai_text('hello world'), ' ')
        self.assertEqual(extract_thai_text('สวัสดี123ครับ'), 'สวัสดี ครับ')
        
    def test_thai_to_roman(self):
        """Test thai_to_roman function"""
        self.assertEqual(thai_to_roman('สวัสดี').lower(), 'sawasdee')
        
    def test_detect_language(self):
        """Test detect_language function"""
        self.assertEqual(detect_language('สวัสดีครับ'), 'thai')
        self.assertEqual(detect_language('hello world'), 'english')
        self.assertEqual(detect_language('สวัสดี hello'), 'mixed')
        
if __name__ == '__main__':
    unittest.main() 