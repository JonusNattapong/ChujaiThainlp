"""
Token Classification for Thai Text using Transformer models with Advanced Features
"""

from typing import List, Dict, Union, Optional, Any, Set, Tuple
import torch
import torch.nn.functional as F
from seqeval.metrics import classification_report
from transformers import AutoTokenizer
from pythainlp.corpus import thai_words, thai_stopwords, thai_named_entities
from pythainlp.tag import pos_tag
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize
from ..core.transformers import TransformerBase

class ThaiTokenClassifier(TransformerBase):
    """Advanced token classifier for Thai text using transformer models"""
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        task_type: str = "ner",  # ner or pos
        labels: Optional[List[str]] = None,
        use_dict_features: bool = True,
        use_context_features: bool = True,
        use_pattern_features: bool = True,
        custom_dict_paths: Optional[List[str]] = None,
        window_size: int = 3,
        **kwargs
    ):
        """Initialize advanced token classifier
        
        Args:
            model_name_or_path: Name or path of the model
            task_type: Type of token classification task (ner or pos)
            labels: List of label names
            use_dict_features: Whether to use dictionary features
            use_context_features: Whether to use contextual features
            use_pattern_features: Whether to use pattern-based features
            custom_dict_paths: List of paths to custom dictionaries
            window_size: Size of context window for feature extraction
            **kwargs: Additional arguments for model initialization
        """
        if model_name_or_path is None:
            model_name_or_path = self.get_default_model(task_type)
            
        self.task_type = task_type
        self.labels = labels
        self.use_dict_features = use_dict_features
        self.use_context_features = use_context_features
        self.use_pattern_features = use_pattern_features
        self.window_size = window_size
        
        # Load dictionaries and patterns
        self.dictionaries = self._load_dictionaries(custom_dict_paths)
        self.patterns = self._load_patterns()
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            task_type="token-classification",
            **kwargs
        )
        
        if not self.labels:
            self.labels = self.model.config.id2label

    def _load_dictionaries(self, custom_dict_paths: Optional[List[str]] = None) -> Dict[str, Set[str]]:
        """Load built-in and custom dictionaries
        
        Args:
            custom_dict_paths: List of paths to custom dictionaries
            
        Returns:
            Dictionary of word sets by category
        """
        dicts = {
            'words': set(thai_words()),
            'stopwords': set(thai_stopwords()),
            'named_entities': set(thai_named_entities()),
            'person': set(),
            'location': set(),
            'organization': set()
        }
        
        # Load named entities by type
        for word in thai_named_entities():
            pos = pos_tag(word, engine='perceptron')[0][1]
            if pos == 'PROPN':
                if any(prefix in word for prefix in ['นาย', 'นาง', 'นางสาว', 'ดร.', 'อาจารย์']):
                    dicts['person'].add(word)
                elif any(word.startswith(prefix) for prefix in ['จังหวัด', 'อำเภอ', 'ตำบล']):
                    dicts['location'].add(word)
                elif any(word.startswith(prefix) for prefix in ['บริษัท', 'ธนาคาร', 'โรงเรียน', 'มหาวิทยาลัย']):
                    dicts['organization'].add(word)
        
        # Load custom dictionaries if provided
        if custom_dict_paths:
            for path in custom_dict_paths:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        category = path.split('/')[-1].replace('.txt', '')
                        words = set(line.strip() for line in f)
                        dicts[category] = words
                except Exception as e:
                    print(f"Warning: Could not load dictionary from {path}: {str(e)}")
                    
        return dicts
    
    def _get_dict_features(self, word: str) -> Dict[str, bool]:
        """Get dictionary features for a word
        
        Args:
            word: Input word
            
        Returns:
            Dictionary of boolean features
        """
        return {
            f"in_{dict_name}": word in word_set
            for dict_name, word_set in self.dictionaries.items()
        }
    
    def _align_tokens_and_labels(
        self,
        text: str,
        predictions: torch.Tensor,
        input_ids: torch.Tensor,
        include_dict_features: bool = True
    ) -> List[Dict[str, Any]]:
        """Align tokens and labels with dictionary features
        
        Args:
            text: Input text
            predictions: Model predictions
            input_ids: Input token IDs
            include_dict_features: Whether to include dictionary features
            
        Returns:
            List of aligned tokens and labels with features
        """
        # Get word ids from tokenizer
        word_ids = self.tokenizer.get_word_ids(text)
        
        # Convert predictions to label indices
        pred_labels = predictions.argmax(dim=-1).squeeze()
        
        # Initialize results
        results = []
        current_word = ""
        current_label = None
        
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:  # Special tokens
                continue
                
            # Get token and label
            token = self.tokenizer.convert_ids_to_tokens([input_ids[token_idx].item()])[0]
            label_idx = pred_labels[token_idx].item()
            label = self.model.config.id2label[label_idx]
            
            # Handle subwords
            if token.startswith("##"):
                current_word += token[2:]
            else:
                if current_word:
                    result = {
                        "word": current_word,
                        "label": current_label,
                        "score": float(F.softmax(predictions[0, token_idx-1], dim=-1).max())
                    }
                    
                    # Add dictionary features if requested
                    if include_dict_features and self.use_dict_features:
                        result["dict_features"] = self._get_dict_features(current_word)
                        
                    results.append(result)
                    
                current_word = token
                current_label = label
                
        # Add last word
        if current_word:
            result = {
                "word": current_word,
                "label": current_label,
                "score": float(F.softmax(predictions[0, -1], dim=-1).max())
            }
            
            if include_dict_features and self.use_dict_features:
                result["dict_features"] = self._get_dict_features(current_word)
                
            results.append(result)
            
        return results
    
    def _load_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Thai language patterns for feature extraction"""
        return {
            'name_patterns': [
                {'prefix': ['นาย', 'นาง', 'นางสาว', 'ดร.', 'ศ.', 'รศ.', 'ผศ.', 'อ.'], 'type': 'PERSON'},
                {'suffix': ['ณ อยุธยา', 'จ เจริญ'], 'type': 'PERSON'}
            ],
            'location_patterns': [
                {'prefix': ['จังหวัด', 'อำเภอ', 'ตำบล', 'หมู่', 'ซอย', 'ถนน'], 'type': 'LOCATION'},
                {'suffix': ['บุรี', 'ธานี', 'นคร'], 'type': 'LOCATION'}
            ],
            'org_patterns': [
                {'prefix': ['บริษัท', 'ธนาคาร', 'โรงเรียน', 'มหาวิทยาลัย', 'สถาบัน', 'องค์การ'], 'type': 'ORGANIZATION'},
                {'suffix': ['จำกัด', 'มหาชน', 'คอร์ปอเรชั่น'], 'type': 'ORGANIZATION'}
            ],
            'time_patterns': [
                {'regex': r'\d{1,2}:\d{2}', 'type': 'TIME'},
                {'suffix': ['นาฬิกา', 'น.', 'โมง'], 'type': 'TIME'}
            ],
            'date_patterns': [
                {'regex': r'\d{1,2}/\d{1,2}/\d{2,4}', 'type': 'DATE'},
                {'thai_month': ['มกราคม', 'กุมภาพันธ์', 'มีนาคม'], 'type': 'DATE'}
            ]
        }

    def _get_context_features(self, tokens: List[str], index: int) -> Dict[str, Any]:
        """Extract contextual features for a token
        
        Args:
            tokens: List of tokens
            index: Current token index
            
        Returns:
            Dictionary of contextual features
        """
        features = {}
        
        # Get surrounding tokens within window
        start = max(0, index - self.window_size)
        end = min(len(tokens), index + self.window_size + 1)
        context = tokens[start:end]
        
        # Previous and next tokens
        features['prev_tokens'] = tokens[start:index] if index > 0 else []
        features['next_tokens'] = tokens[index+1:end] if index < len(tokens)-1 else []
        
        # POS tags of context
        pos_tags = pos_tag(context)
        features['context_pos'] = [tag for _, tag in pos_tags]
        
        # Check for special patterns in context
        features['has_number'] = any(t.isdigit() for t in context)
        features['has_english'] = any(any(c.isascii() and c.isalpha() for c in t) for t in context)
        
        return features

    def _get_pattern_features(self, token: str, context: List[str]) -> Dict[str, bool]:
        """Extract pattern-based features for a token
        
        Args:
            token: Current token
            context: Context tokens
            
        Returns:
            Dictionary of pattern features
        """
        features = {}
        
        # Check each pattern category
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                # Check prefixes
                if 'prefix' in pattern:
                    features[f'has_{pattern["type"]}_prefix'] = any(
                        token.startswith(prefix) for prefix in pattern['prefix']
                    )
                
                # Check suffixes
                if 'suffix' in pattern:
                    features[f'has_{pattern["type"]}_suffix'] = any(
                        token.endswith(suffix) for suffix in pattern['suffix']
                    )
                
                # Check regex patterns
                if 'regex' in pattern:
                    import re
                    features[f'matches_{pattern["type"]}_pattern'] = bool(
                        re.match(pattern['regex'], token)
                    )
                
                # Check Thai month names
                if 'thai_month' in pattern:
                    features[f'is_{pattern["type"]}_term'] = token in pattern['thai_month']
        
        return features

    def _preprocess_text(self, text: str) -> str:
        """Preprocess Thai text
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Normalize text
        text = normalize(text)
        
        # Remove redundant spaces
        text = ' '.join(text.split())
        
        return text

    def classify(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        include_dict_features: bool = True,
        include_context_features: bool = True,
        include_pattern_features: bool = True,
        **kwargs
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Classify tokens in text with advanced features
        
        Args:
            texts: Input text or list of texts
            batch_size: Batch size for processing
            include_dict_features: Whether to include dictionary features
            include_context_features: Whether to include contextual features
            include_pattern_features: Whether to include pattern features
            **kwargs: Additional arguments for encoding
            
        Returns:
            List of dictionaries containing token classifications and features
        """
        # Handle single text input
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]
            
        # Preprocess texts
        texts = [self._preprocess_text(text) for text in texts]
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize texts
            batch_tokens = [word_tokenize(text) for text in batch_texts]
            
            # Encode texts
            inputs = self.encode(batch_texts, **kwargs)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Process each text in batch
            for j, (text, tokens) in enumerate(zip(batch_texts, batch_tokens)):
                results = []
                
                # Get base predictions
                base_results = self._align_tokens_and_labels(
                    text,
                    outputs.logits[j:j+1],
                    inputs["input_ids"][j]
                )
                
                # Add advanced features
                for idx, result in enumerate(base_results):
                    if include_dict_features and self.use_dict_features:
                        result["dict_features"] = self._get_dict_features(result["word"])
                        
                    if include_context_features and self.use_context_features:
                        result["context_features"] = self._get_context_features(tokens, idx)
                        
                    if include_pattern_features and self.use_pattern_features:
                        result["pattern_features"] = self._get_pattern_features(
                            result["word"],
                            tokens[max(0, idx-self.window_size):idx+self.window_size+1]
                        )
                        
                    results.append(result)
                    
                all_results.append(results)
                
        return all_results[0] if single_text else all_results
    
    def add_dictionary(self, category: str, words: Set[str]):
        """Add new dictionary
        
        Args:
            category: Dictionary category name
            words: Set of words to add
        """
        self.dictionaries[category] = words
        
    def update_dictionary(self, category: str, words: Set[str]):
        """Update existing dictionary
        
        Args:
            category: Dictionary category name
            words: Set of words to add
        """
        if category in self.dictionaries:
            self.dictionaries[category].update(words)
        else:
            self.add_dictionary(category, words)
            
    def remove_dictionary(self, category: str):
        """Remove dictionary
        
        Args:
            category: Dictionary category name to remove
        """
        if category in self.dictionaries:
            del self.dictionaries[category]
            
    def train(
        self,
        train_texts: List[str],
        train_labels: List[List[str]],
        validation_texts: Optional[List[str]] = None,
        validation_labels: Optional[List[List[str]]] = None,
        batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        max_length: int = 512,
        **kwargs
    ):
        """Fine-tune the model on custom data
        
        Args:
            train_texts: Training texts
            train_labels: Training labels (list of label sequences)
            validation_texts: Validation texts
            validation_labels: Validation labels
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            max_length: Maximum sequence length
            **kwargs: Additional training arguments
        """
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim import AdamW
        from tqdm.auto import tqdm
        
        # Convert labels to IDs
        label2id = {label: i for i, label in enumerate(self.labels)}
        
        # Prepare training data
        train_encodings = self.encode(train_texts, max_length=max_length)
        train_label_ids = [
            [label2id[label] for label in seq]
            for seq in train_labels
        ]
        
        # Create attention masks for labels
        label_attention_masks = [
            [1] * len(seq) + [0] * (max_length - len(seq))
            for seq in train_label_ids
        ]
        
        # Pad label sequences
        train_label_ids = [
            seq + [0] * (max_length - len(seq))
            for seq in train_label_ids
        ]
        
        # Convert to tensors
        train_label_ids = torch.tensor(train_label_ids)
        label_attention_masks = torch.tensor(label_attention_masks)
        
        # Create dataset
        train_dataset = TensorDataset(
            train_encodings["input_ids"],
            train_encodings["attention_mask"],
            train_label_ids,
            label_attention_masks
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation data if provided
        if validation_texts and validation_labels:
            val_encodings = self.encode(validation_texts, max_length=max_length)
            val_label_ids = [
                [label2id[label] for label in seq]
                for seq in validation_labels
            ]
            
            val_label_attention_masks = [
                [1] * len(seq) + [0] * (max_length - len(seq))
                for seq in val_label_ids
            ]
            
            val_label_ids = [
                seq + [0] * (max_length - len(seq))
                for seq in val_label_ids
            ]
            
            val_label_ids = torch.tensor(val_label_ids)
            val_label_attention_masks = torch.tensor(val_label_attention_masks)
            
            val_dataset = TensorDataset(
                val_encodings["input_ids"],
                val_encodings["attention_mask"],
                val_label_ids,
                val_label_attention_masks
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Prepare optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids, attention_mask, labels, label_attention = [
                    b.to(self.device) for b in batch
                ]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Calculate loss only for actual tokens
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": total_loss / len(train_loader)})
            
            # Validation
            if validation_texts and validation_labels:
                self.model.eval()
                val_loss = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids, attention_mask, labels, label_attention = [
                            b.to(self.device) for b in batch
                        ]
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        val_loss += outputs.loss.item()
                        
                        # Convert predictions and labels to label names
                        preds = outputs.logits.argmax(dim=-1)
                        
                        for i in range(len(preds)):
                            pred_seq = [
                                self.labels[p.item()]
                                for p, m in zip(preds[i], label_attention[i])
                                if m.item() == 1
                            ]
                            label_seq = [
                                self.labels[l.item()]
                                for l, m in zip(labels[i], label_attention[i])
                                if m.item() == 1
                            ]
                            
                            all_preds.append(pred_seq)
                            all_labels.append(label_seq)
                
                # Calculate metrics
                val_loss = val_loss / len(val_loader)
                report = classification_report(all_labels, all_preds)
                print(f"Validation Loss: {val_loss:.4f}")
                print("\nClassification Report:")
                print(report)
                
                self.model.train()
    
    def zero_shot_classify(
        self,
        texts: Union[str, List[str]],
        candidate_labels: List[str],
        **kwargs
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Zero-shot token classification
        
        Args:
            texts: Input text or list of texts
            candidate_labels: List of possible labels
            **kwargs: Additional arguments for classification
        
    Returns:
            Token classifications with probabilities for each label
        """
        from transformers import pipeline
        
        classifier = pipeline(
            "token-classification",
            model=self.model_name,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            aggregation_strategy="simple"
        )
        
        return classifier(texts, candidate_labels=candidate_labels, **kwargs) 