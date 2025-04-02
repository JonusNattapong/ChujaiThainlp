"""
Tests for thainlp/utils/thai_text.py
"""
import unittest
from thainlp.utils.thai_text import (
    process_text,
    ThaiTextProcessor,
    ThaiValidator,
    normalize_text,
    romanize,
    extract_thai,
    split_sentences
)

class TestQuickFunctions(unittest.TestCase):
    """Test pre-built quick functions"""
    
    def setUp(self):
        self.text = "  สวัสดี   Hello  ๑๒๓  "
        
    def test_normalize_text(self):
        result = normalize_text(self.text)
        self.assertEqual(result, "สวัสดี Hello 123")
        
    def test_romanize(self):
        result = romanize(self.text)
        self.assertEqual(result.strip(), "sawatdi Hello 123")
        
    def test_extract_thai(self):
        result = extract_thai(self.text)
        self.assertEqual(result.strip(), "สวัสดี")
        
    def test_split_sentences(self):
        text = "สวัสดีครับ! วันนี้อากาศดีนะครับ"
        result = split_sentences(text)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(s, str) for s in result))

class TestThaiTextProcessor(unittest.TestCase):
    """Test ThaiTextProcessor class"""
    
    def setUp(self):
        self.processor = ThaiTextProcessor()
        
    def test_method_chaining(self):
        result = (self.processor
                 .load("สวัสดี ๑๒๓")
                 .normalize()
                 .to_arabic_digits()
                 .get_text())
        self.assertEqual(result, "สวัสดี 123")
        
    def test_multiple_operations(self):
        result = (process_text("สวัสดี Hello ๑๒๓")
                 .normalize()
                 .extract_thai()
                 .to_roman()
                 .get_text())
        self.assertEqual(result.strip(), "sawatdi")
        
    def test_script_ratios(self):
        ratios = (process_text("Hello สวัสดี")
                 .get_script_ratios())
        self.assertIsInstance(ratios, dict)
        self.assertGreater(ratios['thai'], 0)
        self.assertGreater(ratios['latin'], 0)
        
    def test_character_counts(self):
        counts = (process_text("สวัสดีครับ")
                 .get_character_counts())
        self.assertIsInstance(counts, dict)
        self.assertGreater(counts['consonants'], 0)
        
    def test_number_conversion(self):
        text = ThaiTextProcessor.number_to_thai(123)
        self.assertIsInstance(text, str)
        self.assertTrue(text)

class TestThaiValidator(unittest.TestCase):
    """Test ThaiValidator class"""
    
    def setUp(self):
        self.validator = ThaiValidator()
        
    def test_char_validation(self):
        self.assertTrue(self.validator.is_thai_char("ก"))
        self.assertFalse(self.validator.is_thai_char("a"))
        self.assertFalse(self.validator.is_thai_char(""))
        self.assertFalse(self.validator.is_thai_char("กข"))
        
    def test_word_validation(self):
        self.assertTrue(self.validator.is_valid_word("สวัสดี"))
        self.assertFalse(self.validator.is_valid_word("ก า"))
        self.assertFalse(self.validator.is_valid_word("hello"))
        self.assertFalse(self.validator.is_valid_word(""))
        
    def test_syllable_pattern(self):
        pattern = self.validator.get_syllable_pattern()
        self.assertIsInstance(pattern, str)
        self.assertGreater(len(pattern), 0)
        
    def test_character_types(self):
        types = self.validator.get_character_types()
        self.assertIsInstance(types, dict)
        self.assertIn('consonants', types)
        self.assertIn('vowels', types)
        self.assertIn('tonemarks', types)
        self.assertIn('digits', types)
        self.assertIn('all', types)

class TestProcessorFeatures(unittest.TestCase):
    """Test additional processor features"""
    
    def setUp(self):
        self.processor = ThaiTextProcessor()
        
    def test_remove_tones(self):
        result = (self.processor
                 .load("สวัสดี")
                 .remove_tones()
                 .get_text())
        self.assertEqual(result, "สวสด")
        
    def test_remove_diacritics(self):
        result = (self.processor
                 .load("กรุ๊ปกฤษณ์")
                 .remove_diacritics()
                 .get_text())
        self.assertEqual(result, "กรปกฤษณ")
        
    def test_digit_conversion(self):
        # Thai to Arabic
        result = (self.processor
                 .load("๑๒๓")
                 .to_arabic_digits()
                 .get_text())
        self.assertEqual(result, "123")
        
        # Arabic to Thai
        result = (self.processor
                 .load("123")
                 .to_thai_digits()
                 .get_text())
        self.assertEqual(result, "๑๒๓")
        
    def test_sentence_splitting(self):
        result = (self.processor
                 .load("สวัสดีครับ วันนี้อากาศดี")
                 .split_sentences()
                 .get_sentences())
        self.assertEqual(len(result), 2)
        
    def test_text_analysis(self):
        self.processor.load("สวัสดี Hello")
        
        # Test has_thai
        self.assertTrue(self.processor.has_thai())
        
        # Test character counts
        counts = self.processor.get_character_counts()
        self.assertGreater(sum(counts.values()), 0)
        
        # Test script ratios
        ratios = self.processor.get_script_ratios()
        self.assertGreater(sum(ratios.values()), 0)

if __name__ == '__main__':
    unittest.main()
