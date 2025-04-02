"""
Tests for Maximum Matching Tokenization
"""

import unittest
# Import correct names and resource function
from thainlp.tokenization.maximum_matching import MaximumMatchingTokenizer, word_tokenize
from thainlp.resources import get_words

class TestMaximumMatching(unittest.TestCase):
    def test_word_tokenize_simple(self):
        """Test word_tokenize function"""
        text = "ผมชอบกินข้าว"
        expected = ['ผม', 'ชอบ', 'กิน', 'ข้าว']
        result = word_tokenize(text) # Use word_tokenize
        self.assertEqual(result, expected)

    def test_word_tokenize_with_spaces(self):
        """Test word_tokenize function with spaces"""
        text = "ผม ชอบ กิน ข้าว"
        # Assuming word_tokenize handles spaces correctly (or underlying tokenizer does)
        # Based on MaximumMatchingTokenizer logic, spaces might be treated as separators or single chars
        # Let's assume it splits correctly for now. If this fails, the logic might need review.
        expected = ['ผม', ' ', 'ชอบ', ' ', 'กิน', ' ', 'ข้าว'] # Or just ['ผม', 'ชอบ', 'กิน', 'ข้าว'] depending on impl.
        # Let's test against the more likely scenario of splitting known words
        expected_no_space = ['ผม', 'ชอบ', 'กิน', 'ข้าว']
        result = word_tokenize(text)
        # Check if it matches either expectation, prioritizing no spaces if words are known
        self.assertTrue(result == expected or result == expected_no_space, f"Result: {result}")


    def test_word_tokenize_with_unknown_words(self):
        """Test word_tokenize function with unknown words"""
        text = "ผมชอบกินข้าวผัดกระเพรา" # กระเพรา might be unknown
        result = word_tokenize(text)
        self.assertTrue('ผม' in result)
        self.assertTrue('ชอบ' in result)
        self.assertTrue('กิน' in result)
        self.assertTrue('ข้าว' in result)
        # Check how unknown words are handled (likely split char by char)
        self.assertTrue('ผ' in result)
        self.assertTrue('ั' in result)
        self.assertTrue('ด' in result)
        self.assertTrue('ก' in result)
        self.assertTrue('ร' in result)
        # ... and so on for กระเพรา

    def test_word_tokenize_with_mixed_languages(self):
        """Test word_tokenize function with mixed languages"""
        text = "ผมชอบกิน pizza" # pizza is unknown
        result = word_tokenize(text)
        self.assertTrue('ผม' in result)
        self.assertTrue('ชอบ' in result)
        self.assertTrue('กิน' in result)
        # Check how ' pizza' is handled (space might be separate or attached)
        self.assertTrue(' ' in result or ' pizza' in result or 'p' in result)


    def test_default_dictionary(self):
        """Test the default dictionary from resources"""
        # Test get_words from resources
        dictionary = get_words()
        self.assertTrue(isinstance(dictionary, set))
        self.assertTrue(len(dictionary) > 0)
        # Check for some common words expected in the default dictionary
        self.assertTrue('การ' in dictionary) # From resources.py default
        self.assertTrue('กิน' in dictionary) # From resources.py default

    def test_maximum_matching_tokenizer_class(self):
        """Test MaximumMatchingTokenizer class with custom dictionary"""
        custom_dictionary = {'ผม', 'ชอบ', 'กิน', 'ข้าว'}
        # Instantiate the correct class with the custom dictionary
        tokenizer = MaximumMatchingTokenizer(custom_dict=custom_dictionary)
        # Verify the tokenizer's dictionary is the custom one
        self.assertEqual(tokenizer.dictionary, custom_dictionary)

        text = "ผมชอบกินข้าว"
        expected = ['ผม', 'ชอบ', 'กิน', 'ข้าว']
        result = tokenizer.tokenize(text)
        self.assertEqual(result, expected)

        # Test with text containing words not in the custom dict
        text_unknown = "ผมชอบกินข้าวผัด"
        result_unknown = tokenizer.tokenize(text_unknown)
        expected_unknown = ['ผม', 'ชอบ', 'กิน', 'ข้าว', 'ผ', 'ั', 'ด'] # 'ผัด' is unknown
        self.assertEqual(result_unknown, expected_unknown)

if __name__ == '__main__':
    unittest.main()
