"""
Token classification for Thai and English text
"""
from typing import List, Dict, Set, Union, Optional
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ...core.transformers import TransformerBase
from ...tokenization import word_tokenize
from ...utils.thai_utils import normalize_text

class TokenClassifier(TransformerBase):
    """Transformer-based token classifier supporting Thai and English text"""
    
    def __init__(self, 
                 model_name: str = "xlm-roberta-base",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 32):
        """Initialize the token classifier
        
        Args:
            model_name: Name/path of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name)
        self.device = device
        self.batch_size = batch_size
        
        # Load model and tokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load resources
        self.stopwords = set()  # Now loaded from resources module
        self.thai_words = set()  # Now loaded from resources module
        
    def classify_tokens(self, 
                       texts: Union[str, List[str]], 
                       return_confidence: bool = True) -> List[List[Dict[str, Union[str, float, bool]]]]:
        """Classify tokens in one or more texts
        
        Args:
            texts: Input text or texts
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of lists containing token classifications:
            - token: The token text
            - pos: Part of speech tag 
            - is_stopword: Whether token is stopword
            - is_known_word: Whether token exists in dictionary
            - named_entity: Named entity type if applicable
            - confidence: Model confidence score (if return_confidence=True)
            - normalized: Normalized form of token
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self._process_batch(
                batch_texts,
                return_confidence=return_confidence
            )
            all_results.extend(batch_results)
            
        return all_results[0] if len(texts) == 1 else all_results
    
    def _process_batch(self,
                      texts: List[str],
                      return_confidence: bool) -> List[List[Dict[str, Union[str, float, bool]]]]:
        """Process a batch of texts"""
        batch_results = []
        
        # Tokenize all texts
        tokenized_inputs = self.tokenizer(
            texts, 
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
            
        # Process each text
        for text_idx, text in enumerate(texts):
            # Get basic tokenization
            tokens = word_tokenize(text)
            
            # Get model predictions for this text
            logits = outputs.logits[text_idx]
            predictions = torch.argmax(logits, dim=-1)
            confidences = torch.softmax(logits, dim=-1).max(dim=-1).values
            
            # Process tokens
            results = []
            for token_idx, token in enumerate(tokens):
                token_info = {
                    'token': token,
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
                 train_data: List[Dict],
                 val_data: Optional[List[Dict]] = None,
                 epochs: int = 3,
                 learning_rate: float = 2e-5):
        """Fine-tune the model on domain data
        
        Args:
            train_data: Training examples with 'text' and 'labels' keys
            val_data: Optional validation data
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Process in batches
            for i in range(0, len(train_data), self.batch_size):
                batch_data = train_data[i:i + self.batch_size]
                
                # Prepare inputs
                inputs = self.tokenizer(
                    [d['text'] for d in batch_data],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Prepare labels
                labels = [
                    [self.model.config.label2id[l] for l in d['labels']]
                    for d in batch_data
                ]
                labels = torch.tensor(labels).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = total_loss / len(train_data)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Validation
            if val_data:
                self.model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for i in range(0, len(val_data), self.batch_size):
                        batch_data = val_data[i:i + self.batch_size]
                        
                        inputs = self.tokenizer(
                            [d['text'] for d in batch_data],
                            padding=True,
                            truncation=True,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        labels = [
                            [self.model.config.label2id[l] for l in d['labels']]
                            for d in batch_data
                        ]
                        labels = torch.tensor(labels).to(self.device)
                        
                        outputs = self.model(**inputs, labels=labels)
                        val_loss += outputs.loss.item()
                        
                avg_val_loss = val_loss / len(val_data)
                print(f"Validation Loss: {avg_val_loss:.4f}")
                
                self.model.train()