"""
Tests for Maximum Matching Tokenization
"""

import unittest
from thainlp.tokenization.maximum_matching import tokenize, ThaiTokenizer, create_dictionary

class TestMaximumMatching(unittest.TestCase):
    def test_tokenize(self):
        """Test tokenize function"""
        text = "ผมชอบกินข้าว"
        expected = ['ผม', 'ชอบ', 'กิน', 'ข้าว']
        result = tokenize(text)
        self.assertEqual(result, expected)
        
    def test_tokenize_with_spaces(self):
        """Test tokenize function with spaces"""
        text = "ผม ชอบ กิน ข้าว"
        expected = ['ผม', 'ชอบ', 'กิน', 'ข้าว']
        result = tokenize(text)
        self.assertEqual(result, expected)
        
    def test_tokenize_with_unknown_words(self):
        """Test tokenize function with unknown words"""
        text = "ผมชอบกินข้าวผัดกระเพรา"
        result = tokenize(text)
        self.assertTrue('ผม' in result)
        self.assertTrue('ชอบ' in result)
        self.assertTrue('กิน' in result)
        self.assertTrue('ข้าว' in result)
        
    def test_tokenize_with_mixed_languages(self):
        """Test tokenize function with mixed languages"""
        text = "ผมชอบกิน pizza"
        result = tokenize(text)
        self.assertTrue('ผม' in result)
        self.assertTrue('ชอบ' in result)
        self.assertTrue('กิน' in result)
        
    def test_create_dictionary(self):
        """Test create_dictionary function"""
        dictionary = create_dictionary()
        self.assertTrue(isinstance(dictionary, set))
        self.assertTrue(len(dictionary) > 0)
        self.assertTrue('ผม' in dictionary)
        self.assertTrue('กิน' in dictionary)
        
    def test_thai_tokenizer_class(self):
        """Test ThaiTokenizer class"""
        dictionary = {'ผม', 'ชอบ', 'กิน', 'ข้าว'}
        tokenizer = ThaiTokenizer(dictionary)
        text = "ผมชอบกินข้าว"
        expected = ['ผม', 'ชอบ', 'กิน', 'ข้าว']
        result = tokenizer.tokenize(text)
        self.assertEqual(result, expected)
        
if __name__ == '__main__':
    unittest.main() 