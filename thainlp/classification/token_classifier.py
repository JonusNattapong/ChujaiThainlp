"""
Token classification for Thai text
"""
from typing import List, Dict, Set, Union
from transformers import AutoTokenizer
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize
from ..resources import get_stopwords, get_words
from ..tag import pos_tag
from ..utils.thai_utils import normalize_text

class TokenClassifier(TransformerBase):
    def __init__(self, model_name: str = ""):
        super().__init__(model_name)
        self.tokenizer = None
        self.stopwords = get_stopwords()
        self.thai_words = get_words()
        self.named_entities = self._load_named_entities()
        
    def _load_named_entities(self) -> Dict[str, Set[str]]:
        """Load basic named entity lists"""
        return {
            'PERSON': {
                'สมชาย', 'สมหญิง', 'วิชัย', 'สมศักดิ์', 'สุชาติ',
                'สมบัติ', 'สมพร', 'สมศรี', 'วิเชียร', 'สมคิด'
            },
            'ORGANIZATION': {
                'บริษัท', 'ธนาคาร', 'โรงเรียน', 'มหาวิทยาลัย',
                'กระทรวง', 'สถาบัน', 'องค์การ', 'สำนักงาน'
            },
            'LOCATION': {
                'กรุงเทพ', 'เชียงใหม่', 'ภูเก็ต', 'พัทยา', 'หาดใหญ่',
                'ประเทศไทย', 'ไทย', 'ลาว', 'พม่า', 'เวียดนาม'
            }
        }
        
    def classify_tokens(self, text: str) -> List[Dict[str, Union[str, dict]]]:
        """Classify tokens in text with their properties
        
        Args:
            text: Input text
            
        Returns:
            List of dicts containing token info:
            - token: The token text 
            - pos: Part of speech tag
            - is_stopword: Whether token is stopword
            - is_known_word: Whether token exists in dictionary
            - named_entity: Named entity type if applicable
            - normalized: Normalized form of token
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Get POS tags
        pos_tags = dict(pos_tag(text, tokenize=False))
        
        # Process each token
        results = []
        for token in tokens:
            # Basic token info
            token_info = {
                'token': token,
                'pos': pos_tags.get(token, ''),
                'is_stopword': token in self.stopwords,
                'is_known_word': token in self.thai_words,
                'named_entity': self._get_named_entity(token),
                'normalized': normalize_text(token)
            }
            
            results.append(token_info)
            
        return results
        
    def _get_named_entity(self, token: str) -> str:
        """Get named entity type for token"""
        for entity_type, entities in self.named_entities.items():
            if token in entities:
                return entity_type
        return ''