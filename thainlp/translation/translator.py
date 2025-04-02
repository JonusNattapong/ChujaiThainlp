"""
Thai text translation utilities
"""
from typing import List, Dict, Optional
import json
from sentence_transformers import SentenceTransformer
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize

class Translator(TransformerBase):
    def __init__(self, model_name: str = "", vocab_path: Optional[str] = None):
        super().__init__(model_name)
        self.embedding_model = None
        self.vocab = self._load_vocab(vocab_path) if vocab_path else self._get_default_vocab()
        
    def _load_vocab(self, path: str) -> Dict[str, str]:
        """Load translation vocabulary from file"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _get_default_vocab(self) -> Dict[str, str]:
        """Get basic Thai-English vocabulary"""
        return {
            # Common words
            'สวัสดี': 'hello',
            'ขอบคุณ': 'thank you',
            'ใช่': 'yes',
            'ไม่': 'no',
            
            # Numbers
            'หนึ่ง': 'one',
            'สอง': 'two',
            'สาม': 'three',
            'สี่': 'four',
            'ห้า': 'five',
            
            # Common verbs
            'กิน': 'eat',
            'ดื่ม': 'drink', 
            'นอน': 'sleep',
            'พูด': 'speak',
            'ไป': 'go',
            
            # Common nouns
            'บ้าน': 'house',
            'รถ': 'car',
            'หมา': 'dog',
            'แมว': 'cat',
            'คน': 'person',
            
            # Common adjectives
            'ดี': 'good',
            'ไม่ดี': 'bad',
            'ใหญ่': 'big',
            'เล็ก': 'small',
            'สวย': 'beautiful'
        }
        
    def translate(self, text: str, source_lang: str = 'th', target_lang: str = 'en') -> str:
        """Translate text between Thai and English
        
        Args:
            text: Input text
            source_lang: Source language code ('th' or 'en')
            target_lang: Target language code ('th' or 'en')
            
        Returns:
            Translated text
        """
        if source_lang not in ['th', 'en'] or target_lang not in ['th', 'en']:
            raise ValueError("Only Thai-English translation supported")
            
        # Tokenize input
        tokens = word_tokenize(text) if source_lang == 'th' else text.split()
        
        # Translate each token
        translations = []
        for token in tokens:
            if source_lang == 'th':
                trans = self.vocab.get(token, token)
            else:
                # Reverse vocabulary lookup
                trans = next((k for k, v in self.vocab.items() if v == token), token)
            translations.append(trans)
            
        # Join translations
        if target_lang == 'th':
            return ''.join(translations)
        return ' '.join(translations)
        
    def batch_translate(self, texts: List[str], source_lang: str = 'th', target_lang: str = 'en') -> List[str]:
        """Translate multiple texts"""
        return [self.translate(text, source_lang, target_lang) for text in texts]