"""
Thai text data augmentation utilities
"""
from typing import List, Set, Dict
import random
import numpy as np
from .utils.thai_utils import get_thai_stopwords
from .tokenization import word_tokenize

class ThaiAugmenter:
    def __init__(self):
        self.stopwords = get_thai_stopwords()
        self._init_word_replacements()
        
    def _init_word_replacements(self):
        """Initialize word replacement dictionaries"""
        self.synonyms = {
            'ใหญ่': ['โต', 'มหึมา'],
            'เล็ก': ['น้อย', 'จิ๋ว'],
            'ดี': ['เยี่ยม', 'วิเศษ'],
            'เร็ว': ['ไว', 'รวดเร็ว'],
            'ช้า': ['เชื่องช้า', 'เนิบนาบ'],
            'สวย': ['งาม', 'สวยงาม'],
            'เดิน': ['ย่าง', 'ก้าว'],
            'พูด': ['กล่าว', 'บอก'],
            'กิน': ['ทาน', 'รับประทาน'],
            'นอน': ['พักผ่อน', 'หลับ']
        }
        
    def augment(self, text: str, techniques: List[str] = None,
                num_samples: int = 1) -> List[str]:
        """Augment Thai text using specified techniques
        
        Args:
            text: Input text
            techniques: List of augmentation techniques to use
                       ('synonym', 'delete', 'swap', 'back_translate')
            num_samples: Number of augmented samples to generate
            
        Returns:
            List of augmented texts
        """
        if techniques is None:
            techniques = ['synonym', 'delete', 'swap']
            
        augmented = []
        for _ in range(num_samples):
            aug_text = text
            for technique in techniques:
                if technique == 'synonym':
                    aug_text = self._synonym_replacement(aug_text)
                elif technique == 'delete':
                    aug_text = self._random_deletion(aug_text)
                elif technique == 'swap':
                    aug_text = self._random_swap(aug_text)
            augmented.append(aug_text)
            
        return augmented
        
    def _synonym_replacement(self, text: str, p: float = 0.3) -> str:
        """Replace random words with synonyms"""
        tokens = word_tokenize(text)
        n = max(1, int(len(tokens) * p))
        
        for _ in range(n):
            idx = random.randint(0, len(tokens)-1)
            word = tokens[idx]
            
            if word in self.stopwords or word not in self.synonyms:
                continue
                
            synonyms = self.synonyms[word]
            tokens[idx] = random.choice(synonyms)
            
        return ''.join(tokens)
        
    def _random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words"""
        tokens = word_tokenize(text)
        if len(tokens) == 1:
            return text
            
        new_tokens = []
        for token in tokens:
            if token in self.stopwords or random.random() > p:
                new_tokens.append(token)
                
        if not new_tokens:
            rand_idx = random.randint(0, len(tokens)-1)
            new_tokens = [tokens[rand_idx]]
            
        return ''.join(new_tokens)
        
    def _random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap words"""
        tokens = word_tokenize(text)
        length = len(tokens)
        if length < 2:
            return text
            
        for _ in range(n):
            idx1, idx2 = random.sample(range(length), 2)
            tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
            
        return ''.join(tokens)
