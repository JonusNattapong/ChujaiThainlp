"""
Maximum Matching tokenizer for Thai word segmentation
"""
from typing import List, Set, Optional
import os
import json
from ..utils.thai_utils import is_thai_char

class MaximumMatchingTokenizer:
    """Thai word segmentation using Maximum Matching algorithm"""
    
    def __init__(self, 
                 dict_path: Optional[str] = None,
                 max_word_len: int = 20):
        """Initialize tokenizer
        
        Args:
            dict_path: Path to custom dictionary file
            max_word_len: Maximum word length to consider
        """
        self.max_word_len = max_word_len
        self.thai_words = self._load_dictionary(dict_path)
        
    def _load_dictionary(self, dict_path: Optional[str] = None) -> Set[str]:
        """Load Thai word dictionary
        
        Args:
            dict_path: Path to custom dictionary file
            
        Returns:
            Set of Thai words
        """
        words = set()
        
        # Load default dictionary
        default_dict = os.path.join(
            os.path.dirname(__file__),
            'data',
            'thai_words.txt'
        )
        
        if os.path.exists(default_dict):
            with open(default_dict, 'r', encoding='utf-8') as f:
                words.update(line.strip() for line in f)
                
        # Load custom dictionary if provided
        if dict_path and os.path.exists(dict_path):
            with open(dict_path, 'r', encoding='utf-8') as f:
                if dict_path.endswith('.json'):
                    # JSON format
                    data = json.load(f)
                    if isinstance(data, list):
                        words.update(data)
                    elif isinstance(data, dict):
                        words.update(data.keys())
                else:
                    # Text format (one word per line)
                    words.update(line.strip() for line in f)
                    
        return words
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Thai text using Maximum Matching
        
        Args:
            text: Input Thai text
            
        Returns:
            List of Thai words
        """
        if not text:
            return []
            
        tokens = []
        idx = 0
        
        while idx < len(text):
            # Try to match longest possible word
            longest_match = ''
            
            # Look up possible words starting at current position
            for end in range(min(idx + self.max_word_len, len(text)), idx, -1):
                word = text[idx:end]
                
                # Only consider sequences containing Thai characters
                if not any(is_thai_char(c) for c in word):
                    continue
                    
                if word in self.thai_words:
                    longest_match = word
                    break
                    
            if longest_match:
                # Found a dictionary word
                tokens.append(longest_match)
                idx += len(longest_match)
            else:
                # No match found, take single character
                tokens.append(text[idx])
                idx += 1
                
        return tokens
    
    def tokenize_and_keep_delimiters(self, text: str) -> List[str]:
        """Tokenize text while preserving delimiters
        
        Args:
            text: Input text
            
        Returns:
            List of tokens with delimiters preserved
        """
        if not text:
            return []
            
        # Split text into Thai and non-Thai segments
        segments = []
        current = []
        prev_is_thai = None
        
        for char in text:
            is_thai = is_thai_char(char)
            
            # Start new segment if switching between Thai/non-Thai
            if prev_is_thai is not None and is_thai != prev_is_thai:
                if current:
                    segment = ''.join(current)
                    if prev_is_thai:
                        # Tokenize Thai segment
                        segments.extend(self.tokenize(segment))
                    else:
                        # Keep non-Thai segment as is
                        segments.append(segment)
                    current = []
                    
            current.append(char)
            prev_is_thai = is_thai
            
        # Handle final segment
        if current:
            segment = ''.join(current)
            if prev_is_thai:
                segments.extend(self.tokenize(segment))
            else:
                segments.append(segment)
                
        return segments
    
    def add_words(self, words: List[str]):
        """Add custom words to dictionary
        
        Args:
            words: List of words to add
        """
        self.thai_words.update(words)
        
    def remove_words(self, words: List[str]):
        """Remove words from dictionary
        
        Args:
            words: List of words to remove
        """
        self.thai_words.difference_update(words)
        
    def save_dictionary(self, path: str):
        """Save current dictionary to file
        
        Args:
            path: Output file path
        """
        if path.endswith('.json'):
            # Save as JSON
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(list(self.thai_words), f, ensure_ascii=False, indent=2)
        else:
            # Save as text file
            with open(path, 'w', encoding='utf-8') as f:
                for word in sorted(self.thai_words):
                    f.write(f"{word}\n")
                    
    def __len__(self) -> int:
        """Get dictionary size"""
        return len(self.thai_words)
    
    def __contains__(self, word: str) -> bool:
        """Check if word exists in dictionary"""
        return word in self.thai_words
