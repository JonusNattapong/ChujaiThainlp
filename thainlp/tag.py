"""
Advanced Thai part-of-speech tagging with transformer support
"""
from typing import List, Tuple, Dict, Union
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)
from .tokenization import word_tokenize
from .pos_tagging.hmm_tagger import HMMTagger
from ..model_hub import get_model_info

class ThaiPOSTagger:
    """Advanced Thai POS tagger with transformer and HMM support"""
    
    def __init__(self,
                model_name: str = "airesearch/wangchanberta-base-att-spm-pos",
                use_hmm_backup: bool = True,
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize POS tagger
        
        Args:
            model_name: Name of transformer model to use
            use_hmm_backup: Whether to use HMM as backup for unknown words
            device: Device to run model on
        """
        self.device = device
        self.use_hmm_backup = use_hmm_backup
        
        # Load transformer model
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize HMM tagger as backup
        if use_hmm_backup:
            self.hmm_tagger = HMMTagger()
            
        # Extended Thai POS tag set
        self.tag_set = {
            # Core POS tags
            'NOUN': 'Noun',
            'PROPN': 'Proper noun',
            'VERB': 'Verb',
            'ADJ': 'Adjective',
            'ADV': 'Adverb',
            'PRON': 'Pronoun',
            'DET': 'Determiner',
            'ADP': 'Adposition',
            'CONJ': 'Conjunction',
            'PART': 'Particle',
            'NUM': 'Number',
            'PUNCT': 'Punctuation',
            
            # Extended tags
            'AUX': 'Auxiliary verb',
            'INTJ': 'Interjection',
            'SYM': 'Symbol',
            'CCONJ': 'Coordinating conjunction',
            'SCONJ': 'Subordinating conjunction',
            
            # Thai-specific tags
            'CLSFR': 'Classifier',
            'NEGATE': 'Negation',
            'PRON_PERS': 'Personal pronoun',
            'PRON_POSS': 'Possessive pronoun',
            'PRON_DEM': 'Demonstrative pronoun',
            'PART_NEG': 'Negative particle',
            'PART_QUEST': 'Question particle'
        }
        
    def tag(self, text: Union[str, List[str]], return_tensors: bool = False) -> List[Tuple[str, str]]:
        """
        Tag Thai text with POS tags
        
        Args:
            text: Input text or list of tokens
            return_tensors: Whether to return raw model outputs
            
        Returns:
            List of (word, tag) tuples
        """
        # Handle input text
        if isinstance(text, str):
            tokens = word_tokenize(text)
        else:
            tokens = text
            
        # Get transformer predictions
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        # Convert predictions to tags
        tags = [self.model.config.id2label[p.item()] for p in predictions[0][:len(tokens)]]
        
        # Use HMM backup for unknown tags if enabled
        if self.use_hmm_backup:
            for i, tag in enumerate(tags):
                if tag not in self.tag_set:
                    hmm_tag = self.hmm_tagger.tag([tokens[i]])[0][1]
                    tags[i] = hmm_tag
                    
        if return_tensors:
            return list(zip(tokens, tags)), outputs
        return list(zip(tokens, tags))
        
    def fine_tune(self,
                 train_data: List[Dict[str, Union[str, List[str]]]],
                 eval_data: List[Dict[str, Union[str, List[str]]]] = None,
                 learning_rate: float = 5e-5,
                 num_epochs: int = 3,
                 batch_size: int = 16):
        """
        Fine-tune the transformer model on custom data
        
        Args:
            train_data: List of dicts with 'tokens' and 'tags' keys
            eval_data: Optional evaluation data
            learning_rate: Learning rate for fine-tuning
            num_epochs: Number of training epochs
            batch_size: Training batch size
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                
                # Prepare inputs
                tokens = [d['tokens'] for d in batch]
                tags = [d['tags'] for d in batch]
                
                inputs = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Convert tags to label ids
                label_ids = []
                for doc_tags in tags:
                    doc_labels = [self.model.config.label2id[t] for t in doc_tags]
                    label_ids.append(doc_labels)
                    
                # Pad labels
                max_len = inputs['input_ids'].size(1)
                padded_labels = torch.full((len(batch), max_len), -100)
                for i, labels in enumerate(label_ids):
                    padded_labels[i, :len(labels)] = torch.tensor(labels)
                
                labels = padded_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_data)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # Evaluation
            if eval_data:
                self.model.eval()
                eval_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for i in range(0, len(eval_data), batch_size):
                        batch = eval_data[i:i + batch_size]
                        
                        tokens = [d['tokens'] for d in batch]
                        tags = [d['tags'] for d in batch]
                        
                        inputs = self.tokenizer(
                            tokens,
                            is_split_into_words=True,
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        ).to(self.device)
                        
                        label_ids = []
                        for doc_tags in tags:
                            doc_labels = [self.model.config.label2id[t] for t in doc_tags]
                            label_ids.append(doc_labels)
                            
                        max_len = inputs['input_ids'].size(1)
                        padded_labels = torch.full((len(batch), max_len), -100)
                        for i, labels in enumerate(label_ids):
                            padded_labels[i, :len(labels)] = torch.tensor(labels)
                            
                        labels = padded_labels.to(self.device)
                        
                        outputs = self.model(**inputs, labels=labels)
                        eval_loss += outputs.loss.item()
                        
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        
                        # Calculate accuracy only on valid tokens (not padding)
                        mask = labels != -100
                        correct += (predictions[mask] == labels[mask]).sum().item()
                        total += mask.sum().item()
                        
                avg_eval_loss = eval_loss / len(eval_data)
                accuracy = correct / total
                print(f"Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")
                
                self.model.train()
                
    def save(self, path: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def load(self, path: str):
        """Load model and tokenizer"""
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

def pos_tag(text: str, use_transformer: bool = True) -> List[Tuple[str, str]]:
    """
    Tag parts of speech in Thai text
    
    Args:
        text: Input Thai text
        use_transformer: Whether to use transformer model (True) or HMM (False)
        
    Returns:
        List of (word, tag) tuples
    """
    if use_transformer:
        tagger = ThaiPOSTagger()
        return tagger.tag(text)
    else:
        tagger = HMMTagger()
        tokens = word_tokenize(text)
        return tagger.tag(tokens)