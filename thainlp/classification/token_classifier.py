"""
Token classification for Thai and English text
"""
from typing import List, Dict, Set, Union, Optional
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize
from ..resources import get_stopwords, get_words
from ..tag import pos_tag
from ..utils.thai_utils import normalize_text

class TokenClassifier(TransformerBase):
    """Transformer-based token classifier supporting Thai and English text"""
    
    def __init__(self, 
                 model_name: str = "xlm-roberta-base",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 32):
        """Initialize the token classifier
        
        Args:
            model_name: Name/path of pretrained model
            device: Device to run model on (cuda/cpu)
            batch_size: Batch size for processing
        """
        super().__init__(model_name)
        self.device = device
        self.batch_size = batch_size
        
        # Load model and tokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load resources
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
        
    def classify_tokens(self, 
                       texts: Union[str, List[str]], 
                       return_confidence: bool = True) -> List[List[Dict[str, Union[str, float, bool]]]]:
        """Classify tokens in one or more texts
        
        Args:
            texts: Input text or list of texts
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of lists containing token classifications:
            - token: The token text
            - pos: Part of speech tag 
            - is_stopword: Whether token is stopword
            - is_known_word: Whether token exists in dictionary
            - named_entity: Named entity type from model
            - confidence: Model confidence score (if return_confidence=True)
            - normalized: Normalized form of token
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self._process_batch(batch_texts, return_confidence)
            all_results.extend(batch_results)
            
        return all_results[0] if len(texts) == 1 else all_results
    
    def _process_batch(self,
                      texts: List[str],
                      return_confidence: bool) -> List[List[Dict[str, Union[str, float, bool]]]]:
        """Process a batch of texts"""
        batch_results = []
        
        # Tokenize all texts
        tokenized_inputs = self.tokenizer(texts, 
                                        padding=True,
                                        truncation=True,
                                        return_tensors="pt").to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
            
        # Process each text
        for text_idx, text in enumerate(texts):
            # Get basic tokenization
            tokens = word_tokenize(text)
            pos_tags = dict(pos_tag(text, tokenize=False))
            
            # Get model predictions for this text
            logits = outputs.logits[text_idx]
            predictions = torch.argmax(logits, dim=-1)
            confidences = torch.softmax(logits, dim=-1).max(dim=-1).values
            
            # Process tokens
            results = []
            for token_idx, token in enumerate(tokens):
                token_info = {
                    'token': token,
                    'pos': pos_tags.get(token, ''),
                    'is_stopword': token in self.stopwords,
                    'is_known_word': token in self.thai_words,
                    'named_entity': self.model.config.id2label[predictions[token_idx].item()],
                    'normalized': normalize_text(token)
                }
                
                if return_confidence:
                    token_info['confidence'] = confidences[token_idx].item()
                    
                results.append(token_info)
                
            batch_results.append(results)
            
        return batch_results
    
    def fine_tune(self,
                 train_texts: List[str],
                 train_labels: List[List[str]],
                 val_texts: Optional[List[str]] = None,
                 val_labels: Optional[List[List[str]]] = None,
                 epochs: int = 3,
                 learning_rate: float = 2e-5):
        """Fine-tune the model on domain data
        
        Args:
            train_texts: Training text samples
            train_labels: Token labels for training texts
            val_texts: Validation text samples
            val_labels: Token labels for validation texts
            epochs: Number of training epochs
            learning_rate: Learning rate for training
        """
        # Prepare training data
        train_encodings = self.tokenizer(train_texts, 
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt")
        train_labels = self._encode_labels(train_labels)
        
        # Prepare validation data if provided
        if val_texts is not None and val_labels is not None:
            val_encodings = self.tokenizer(val_texts,
                                         padding=True, 
                                         truncation=True,
                                         return_tensors="pt")
            val_labels = self._encode_labels(val_labels)
            
        # Set up training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            # Training
            for i in range(0, len(train_texts), self.batch_size):
                batch_texts = {k: v[i:i + self.batch_size].to(self.device) 
                             for k, v in train_encodings.items()}
                batch_labels = train_labels[i:i + self.batch_size].to(self.device)
                
                outputs = self.model(**batch_texts, labels=batch_labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            # Validation
            if val_texts is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for i in range(0, len(val_texts), self.batch_size):
                        batch_texts = {k: v[i:i + self.batch_size].to(self.device)
                                     for k, v in val_encodings.items()}
                        batch_labels = val_labels[i:i + self.batch_size].to(self.device)
                        
                        outputs = self.model(**batch_texts, labels=batch_labels)
                        val_loss += outputs.loss.item()
                        
                print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {val_loss/len(val_texts)}")
                self.model.train()
                
    def _encode_labels(self, labels: List[List[str]]) -> torch.Tensor:
        """Convert text labels to label IDs"""
        label_map = {label: idx for idx, label in self.model.config.id2label.items()}
        encoded = []
        for seq in labels:
            encoded.append([label_map[label] for label in seq])
        return torch.tensor(encoded)