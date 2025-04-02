"""
Tests for thainlp/utils/thai_utils.py
"""
import unittest
from thainlp.utils import thai_utils

class TestThaiUtils(unittest.TestCase):
    def setUp(self):
        self.thai_text = "สวัสดีครับ นี่คือการทดสอบภาษาไทย"
        self.mixed_text = "Hello สวัสดี 123"
        
    def test_is_thai_char(self):
        self.assertTrue(thai_utils.is_thai_char("ส"))
        self.assertFalse(thai_utils.is_thai_char("A"))
        self.assertFalse(thai_utils.is_thai_char(""))  # Empty string
        self.assertFalse(thai_utils.is_thai_char("สวัสดี"))  # Multiple chars
        
    def test_is_thai_word(self):
        self.assertTrue(thai_utils.is_thai_word("ภาษาไทย"))
        self.assertFalse(thai_utils.is_thai_word("English"))
        self.assertTrue(thai_utils.is_thai_word("ไทย123"))  # Mixed with numbers
        
    def test_remove_tone_marks(self):
        self.assertEqual(thai_utils.remove_tone_marks("สวัสดี"), "สวสด")
        self.assertEqual(thai_utils.remove_tone_marks("ก่อน"), "กอน")
        self.assertEqual(thai_utils.remove_tone_marks("ที่นี่"), "ทน")
        
    def test_remove_diacritics(self):
        # Test tone marks
        self.assertEqual(thai_utils.remove_diacritics("สวัสดี"), "สวสด")
        # Test other diacritics
        self.assertEqual(thai_utils.remove_diacritics("กรุ๊ป"), "กรป")
        self.assertEqual(thai_utils.remove_diacritics("กฤษณ์"), "กฤษณ")
        
    def test_normalize_text(self):
        # Test multiple spaces
        self.assertEqual(thai_utils.normalize_text("  สวัสดี   ครับ  "), "สวัสดี ครับ")
        # Test Thai numerals
        self.assertEqual(thai_utils.normalize_text("๑๒๓"), "123")
        # Test line endings
        self.assertEqual(thai_utils.normalize_text("สวัสดี\r\nครับ"), "สวัสดี\nครับ")
        # Test invisible characters
        self.assertEqual(thai_utils.normalize_text("สวัสดี​ครับ"), "สวัสดี ครับ")
        
    def test_count_thai_words(self):
        self.assertEqual(thai_utils.count_thai_words(self.thai_text), 4)
        self.assertEqual(thai_utils.count_thai_words(self.mixed_text), 1)
        self.assertEqual(thai_utils.count_thai_words(""), 0)  # Empty text
        
    def test_extract_thai_text(self):
        self.assertEqual(thai_utils.extract_thai_text(self.mixed_text), "สวัสดี ")
        self.assertEqual(thai_utils.extract_thai_text("ABC123!@#"), "")
        self.assertEqual(thai_utils.extract_thai_text(""), "")
        
    def test_thai_to_roman(self):
        self.assertEqual(thai_utils.thai_to_roman("สวัสดี"), "sawatdi")
        self.assertEqual(thai_utils.thai_to_roman("กรุงเทพ"), "krungthep")
        self.assertEqual(thai_utils.thai_to_roman("ประเทศไทย"), "prathethai")
        
    def test_detect_language(self):
        # Test empty text
        empty_ratios = thai_utils.detect_language("")
        self.assertEqual(empty_ratios["thai"], 0)
        self.assertEqual(empty_ratios["latin"], 0)
        
        # Test pure Thai text
        thai_ratios = thai_utils.detect_language(self.thai_text)
        self.assertGreater(thai_ratios["thai"], 0.8)
        self.assertEqual(thai_ratios["latin"], 0)
        
        # Test mixed text
        mixed_ratios = thai_utils.detect_language(self.mixed_text)
        self.assertGreater(mixed_ratios["thai"], 0)
        self.assertGreater(mixed_ratios["latin"], 0)
        
    def test_thai_number_to_text(self):
        # Test basic numbers
        self.assertEqual(thai_utils.thai_number_to_text(0), "ศูนย์")
        self.assertEqual(thai_utils.thai_number_to_text(5), "ห้า")
        self.assertEqual(thai_utils.thai_number_to_text(10), "สิบ")
        
        # Test teens (11-19)
        self.assertEqual(thai_utils.thai_number_to_text(11), "สิบเอ็ด")
        self.assertEqual(thai_utils.thai_number_to_text(15), "สิบห้า")
        
        # Test tens
        self.assertEqual(thai_utils.thai_number_to_text(20), "ยี่สิบ")
        self.assertEqual(thai_utils.thai_number_to_text(21), "ยี่สิบเอ็ด")
        self.assertEqual(thai_utils.thai_number_to_text(30), "สามสิบ")
        
        # Test hundreds and thousands
        self.assertEqual(thai_utils.thai_number_to_text(100), "หนึ่งร้อย")
        self.assertEqual(thai_utils.thai_number_to_text(101), "หนึ่งร้อยเอ็ด")
        self.assertEqual(thai_utils.thai_number_to_text(1000), "หนึ่งพัน")
        
        # Test negative numbers
        self.assertEqual(thai_utils.thai_number_to_text(-5), "ลบห้า")
        
        # Test decimal numbers
        self.assertEqual(thai_utils.thai_number_to_text(1.5), "หนึ่งจุดห้า")
        
    def test_digit_conversion(self):
        self.assertEqual(thai_utils.thai_digit_to_arabic_digit("๑๒๓"), "123")
        self.assertEqual(thai_utils.arabic_digit_to_thai_digit("123"), "๑๒๓")
        self.assertEqual(thai_utils.thai_digit_to_arabic_digit(""), "")
        self.assertEqual(thai_utils.arabic_digit_to_thai_digit("abc"), "abc")
        
    def test_split_thai_sentences(self):
        # Test basic sentence splitting
        text = "สวัสดีครับ! วันนี้เป็นอย่างไรบ้าง?"
        sentences = thai_utils.split_thai_sentences(text)
        self.assertEqual(len(sentences), 2)
        
        # Test Thai specific endings
        text = "สวัสดีค่ะ เป็นอย่างไรบ้างคะ ฉันสบายดีจ้า"
        sentences = thai_utils.split_thai_sentences(text)
        self.assertEqual(len(sentences), 3)
        
        # Test with abbreviations
        text = "ดร. สมศักดิ์ อยู่ที่ กทม. วันนี้"
        sentences = thai_utils.split_thai_sentences(text)
        self.assertEqual(len(sentences), 1)
        
        # Test with conjunctions
        text = "เขาไปตลาด และซื้อผลไม้มา"
        sentences = thai_utils.split_thai_sentences(text)
        self.assertEqual(len(sentences), 1)
        
    def test_count_thai_characters(self):
        counts = thai_utils.count_thai_characters("สวัสดีครับ ๑๒๓")
        self.assertGreater(counts["consonants"], 0)
        self.assertGreater(counts["vowels"], 0)
        self.assertGreater(counts["digits"], 0)
        self.assertEqual(counts.get("other", 0), 0)
        
    def test_get_thai_character_types(self):
        types = thai_utils.get_thai_character_types()
        self.assertIn("consonants", types)
        self.assertIn("vowels", types)
        self.assertIn("tonemarks", types)
        self.assertIn("digits", types)
        self.assertIn("all", types)
        
    def test_get_thai_syllable_pattern(self):
        pattern = thai_utils.get_thai_syllable_pattern()
        self.assertIsInstance(pattern, str)
        self.assertGreater(len(pattern), 0)
        
    def test_is_valid_thai_word(self):
        # Test valid Thai words
        self.assertTrue(thai_utils.is_valid_thai_word("กา"))
        self.assertTrue(thai_utils.is_valid_thai_word("ไทย"))
        self.assertTrue(thai_utils.is_valid_thai_word("สวัสดี"))
        
        # Test invalid Thai words
        self.assertFalse(thai_utils.is_valid_thai_word("a"))
        self.assertFalse(thai_utils.is_valid_thai_word("123"))
        self.assertFalse(thai_utils.is_valid_thai_word(""))
        self.assertFalse(thai_utils.is_valid_thai_word("ก า"))  # Space in middle

if __name__ == "__main__":
    unittest.main()
