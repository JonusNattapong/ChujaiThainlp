"""
Maximum matching tokenizer for Thai text
"""
from typing import List, Set # Added Set for type hint
import re
# Removed: from . import THAI_WORDS
from thainlp.resources import get_words # Import get_words function

class MaximumMatchingTokenizer:
    def __init__(self, custom_dict: Set[str] = None): # Changed type hint to Set[str]
        # Use get_words() to fetch the dictionary from resources
        self.dictionary: Set[str] = get_words() # Changed to call get_words()
        if custom_dict:
            self.dictionary.update(custom_dict)
            
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using maximum matching algorithm"""
        if not text.strip():
            return []
            
        tokens = []
        while text:
            # Find longest matching word
            for i in range(len(text), 0, -1):
                word = text[:i]
                if word in self.dictionary:
                    tokens.append(word)
                    text = text[i:]
                    break
            else:
                # No match found, split single character
                tokens.append(text[0])
                text = text[1:]
                
        return tokens

def word_tokenize(text: str) -> List[str]:
    """Convenience function for tokenizing text"""
    tokenizer = MaximumMatchingTokenizer()
    return tokenizer.tokenize(text)
