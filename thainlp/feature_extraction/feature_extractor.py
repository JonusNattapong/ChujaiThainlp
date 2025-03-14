"""
Advanced Feature Extraction for Thai Language
"""

from typing import List, Dict, Union, Optional, Any
import torch
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from sentence_transformers import SentenceTransformer
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag
from pythainlp.util import thai_characters, thai_digits
from sklearn.feature_extraction.text import TfidfVectorizer
from ..core.transformers import TransformerBase

class ThaiFeatureExtractor(TransformerBase):
    """Advanced feature extraction for Thai language"""
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        embedding_type: str = "contextual",
        use_cuda: bool = True,
        max_length: int = 512,
        pooling_strategy: str = "mean",
        domain_features: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize feature extractor
        
        Args:
            model_name_or_path: Name or path of the model
            embedding_type: Type of embeddings (contextual, static, or both)
            use_cuda: Whether to use GPU if available
            max_length: Maximum sequence length
            pooling_strategy: Strategy for pooling token embeddings
            domain_features: List of domain-specific features to extract
            **kwargs: Additional arguments for model initialization
        """
        if model_name_or_path is None:
            model_name_or_path = "wangchanberta-base-att-spm-uncased"
            
        self.embedding_type = embedding_type
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.domain_features = domain_features or []
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            task_type="feature_extraction",
            **kwargs
        )
        
        # Initialize sentence transformer for static embeddings
        if embedding_type in ["static", "both"]:
            self.static_model = SentenceTransformer(
                'paraphrase-multilingual-mpnet-base-v2',
                device="cuda" if use_cuda and torch.cuda.is_available() else "cpu"
            )
            
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(tokenizer=word_tokenize)
        
    def _get_contextual_embeddings(
        self,
        texts: Union[str, List[str]],
        layers: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Get contextual embeddings from transformer model
        
        Args:
            texts: Input text or list of texts
            layers: List of layers to extract embeddings from
            
        Returns:
            Tensor of embeddings
        """
        # Prepare input
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}
            
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(
                **encoded,
                output_hidden_states=True
            )
            
        # Get hidden states
        if layers is None:
            # Use last layer by default
            hidden_states = outputs.last_hidden_state
        else:
            # Combine specified layers
            hidden_states = torch.stack([
                outputs.hidden_states[layer]
                for layer in layers
            ]).mean(dim=0)
            
        # Apply pooling
        if self.pooling_strategy == "mean":
            # Mean pooling
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
        elif self.pooling_strategy == "cls":
            # Use [CLS] token
            embeddings = hidden_states[:, 0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            
        return embeddings
        
    def _get_static_embeddings(
        self,
        texts: Union[str, List[str]]
    ) -> torch.Tensor:
        """Get static embeddings from sentence transformer
        
        Args:
            texts: Input text or list of texts
            
        Returns:
            Tensor of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        return self.static_model.encode(
            texts,
            convert_to_tensor=True
        )
        
    def _get_linguistic_features(
        self,
        text: str
    ) -> Dict[str, float]:
        """Extract linguistic features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of linguistic features
        """
        # Tokenize
        tokens = word_tokenize(text)
        pos_tags = pos_tag(text)
        
        # Basic statistics
        features = {
            'num_tokens': len(tokens),
            'avg_token_length': np.mean([len(t) for t in tokens]),
            'num_sentences': len(text.split('.')),
            'num_thai_chars': sum(1 for c in text if c in thai_characters),
            'num_thai_digits': sum(1 for c in text if c in thai_digits),
            'num_spaces': text.count(' '),
        }
        
        # POS tag distribution
        pos_counts = {}
        for _, pos in pos_tags:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
        for pos, count in pos_counts.items():
            features[f'pos_{pos}'] = count / len(pos_tags)
            
        return features
        
    def _get_domain_features(
        self,
        text: str
    ) -> Dict[str, Any]:
        """Extract domain-specific features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of domain features
        """
        features = {}
        
        for feature in self.domain_features:
            if feature == "sentiment":
                # Add sentiment analysis
                from pythainlp.sentiment import sentiment
                features['sentiment_score'] = sentiment(text)
                
            elif feature == "readability":
                # Add readability metrics
                tokens = word_tokenize(text)
                features.update({
                    'readability_score': len(set(tokens)) / len(tokens),
                    'avg_word_length': np.mean([len(t) for t in tokens])
                })
                
            elif feature == "formality":
                # Add formality detection
                formal_markers = ['ครับ', 'ค่ะ', 'ท่าน', 'กระผม']
                informal_markers = ['จ้า', 'ครัช', 'เด้อ', 'จ้ะ']
                
                formal_count = sum(text.count(m) for m in formal_markers)
                informal_count = sum(text.count(m) for m in informal_markers)
                
                total = formal_count + informal_count
                if total > 0:
                    features['formality_score'] = formal_count / total
                else:
                    features['formality_score'] = 0.5
                    
        return features
        
    def extract_features(
        self,
        texts: Union[str, List[str]],
        include_embeddings: bool = True,
        include_linguistic: bool = True,
        include_domain: bool = True,
        layers: Optional[List[int]] = None,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Extract features from text
        
        Args:
            texts: Input text or list of texts
            include_embeddings: Whether to include embeddings
            include_linguistic: Whether to include linguistic features
            include_domain: Whether to include domain features
            layers: List of layers to extract embeddings from
            **kwargs: Additional arguments for feature extraction
            
        Returns:
            Dictionary or list of dictionaries with extracted features
        """
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]
            
        results = []
        
        for text in texts:
            features = {'text': text}
            
            # Get embeddings
            if include_embeddings:
                if self.embedding_type in ["contextual", "both"]:
                    features['contextual_embedding'] = self._get_contextual_embeddings(
                        text,
                        layers=layers
                    )
                    
                if self.embedding_type in ["static", "both"]:
                    features['static_embedding'] = self._get_static_embeddings(text)
                    
            # Get linguistic features
            if include_linguistic:
                features['linguistic'] = self._get_linguistic_features(text)
                
            # Get domain features
            if include_domain and self.domain_features:
                features['domain'] = self._get_domain_features(text)
                
            results.append(features)
            
        return results[0] if single_text else results
        
    def get_similarity(
        self,
        text1: str,
        text2: str,
        method: str = "cosine",
        embedding_type: Optional[str] = None
    ) -> float:
        """Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method (cosine, euclidean, or dot)
            embedding_type: Type of embeddings to use (contextual or static)
            
        Returns:
            Similarity score
        """
        # Get embeddings
        if embedding_type is None:
            embedding_type = self.embedding_type
            
        if embedding_type == "contextual":
            emb1 = self._get_contextual_embeddings(text1)
            emb2 = self._get_contextual_embeddings(text2)
        else:
            emb1 = self._get_static_embeddings(text1)
            emb2 = self._get_static_embeddings(text2)
            
        # Calculate similarity
        if method == "cosine":
            return torch.nn.functional.cosine_similarity(emb1, emb2).item()
        elif method == "euclidean":
            return -torch.norm(emb1 - emb2).item()
        elif method == "dot":
            return torch.dot(emb1.flatten(), emb2.flatten()).item()
        else:
            raise ValueError(f"Unknown similarity method: {method}")
            
    def get_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Find most similar texts from candidates
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return
            **kwargs: Additional arguments for similarity calculation
            
        Returns:
            List of dictionaries with text and similarity score
        """
        similarities = []
        
        for text in candidates:
            score = self.get_similarity(query, text, **kwargs)
            similarities.append({
                'text': text,
                'similarity': score
            })
            
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
        
    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        use_tfidf: bool = True
    ) -> List[Dict[str, float]]:
        """Extract keywords from text
        
        Args:
            text: Input text
            top_k: Number of keywords to extract
            use_tfidf: Whether to use TF-IDF scores
            
        Returns:
            List of dictionaries with keyword and score
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        if use_tfidf:
            # Fit TF-IDF
            self.tfidf.fit([text])
            
            # Get feature names and scores
            feature_names = self.tfidf.get_feature_names_out()
            scores = self.tfidf.transform([text]).toarray()[0]
            
            # Create keyword-score pairs
            keywords = [
                {'keyword': feature_names[i], 'score': scores[i]}
                for i in range(len(feature_names))
            ]
            
        else:
            # Use frequency
            from collections import Counter
            counts = Counter(tokens)
            total = sum(counts.values())
            
            keywords = [
                {'keyword': word, 'score': count / total}
                for word, count in counts.items()
            ]
            
        # Sort by score
        keywords.sort(key=lambda x: x['score'], reverse=True)
        
        return keywords[:top_k] 