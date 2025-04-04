"""
Advanced sentence similarity and ranking for Thai and English
"""
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
import torch
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
    util
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from ..tokenization import word_tokenize
from ..core.transformers import TransformerBase
from ..extensions.monitoring import ProgressTracker

class SentenceSimilarity(TransformerBase):
    """Sentence similarity with support for bi-encoders and cross-encoders"""
    
    def __init__(self,
                 model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 cross_encoder_name: Optional[str] = "cross-encoder/stsb-roberta-large",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 32,
                 cache_embeddings: bool = True):
        """Initialize similarity models
        
        Args:
            model_name: Bi-encoder model name
            cross_encoder_name: Cross-encoder model name (or None to disable)
            device: Device to run models on
            batch_size: Batch size for processing
            cache_embeddings: Whether to cache computed embeddings
        """
        super().__init__(model_name)
        self.device = device
        self.batch_size = batch_size
        self.cache_embeddings = cache_embeddings
        
        # Load bi-encoder model
        self.model = SentenceTransformer(model_name).to(device)
        
        # Load cross-encoder if specified
        self.cross_encoder = None
        if cross_encoder_name:
            self.cross_encoder = CrossEncoder(cross_encoder_name).to(device)
            
        # Initialize embedding cache
        self._embedding_cache = {}
        
        # Set up progress tracking
        self.progress = ProgressTracker()
        
        # TF-IDF vectorizer for traditional similarity
        self.vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
        
    def get_similarity(self,
                      texts1: Union[str, List[str]],
                      texts2: Union[str, List[str]],
                      method: str = 'bi-encoder',
                      use_cross_encoder: bool = False) -> Union[float, List[float]]:
        """Calculate similarity between text pairs
        
        Args:
            texts1: First text or list of texts
            texts2: Second text or list of texts
            method: Similarity method ('bi-encoder', 'cross-encoder', 'tfidf', 'token')
            use_cross_encoder: Whether to use cross-encoder for refinement
            
        Returns:
            Similarity score(s) between 0 and 1
        """
        # Handle single string inputs
        if isinstance(texts1, str):
            texts1 = [texts1]
            texts2 = [texts2]
            single_pair = True
        else:
            single_pair = False
            
        # Validate input lengths
        if len(texts1) != len(texts2):
            raise ValueError("Input text lists must have same length")
            
        all_scores = []
        self.progress.start_task(len(texts1))
        
        # Process in batches
        for i in range(0, len(texts1), self.batch_size):
            batch_texts1 = texts1[i:i + self.batch_size]
            batch_texts2 = texts2[i:i + self.batch_size]
            
            if method == 'bi-encoder':
                scores = self._bi_encoder_similarity(batch_texts1, batch_texts2)
                
                # Refine with cross-encoder if requested
                if use_cross_encoder and self.cross_encoder:
                    cross_scores = self._cross_encoder_similarity(
                        batch_texts1,
                        batch_texts2
                    )
                    scores = [0.7 * s1 + 0.3 * s2 for s1, s2 in zip(scores, cross_scores)]
                    
            elif method == 'cross-encoder':
                if not self.cross_encoder:
                    raise ValueError("Cross-encoder not initialized")
                scores = self._cross_encoder_similarity(batch_texts1, batch_texts2)
            elif method == 'tfidf':
                scores = self._tfidf_similarity(batch_texts1, batch_texts2)
            elif method == 'token':
                scores = self._token_similarity(batch_texts1, batch_texts2)
            else:
                raise ValueError(f"Unknown similarity method: {method}")
                
            all_scores.extend(scores)
            self.progress.update(len(batch_texts1))
            
        self.progress.end_task()
        
        return all_scores[0] if single_pair else all_scores
    
    def _bi_encoder_similarity(self,
                             texts1: List[str],
                             texts2: List[str]) -> List[float]:
        """Calculate similarity using bi-encoder"""
        # Get or compute embeddings
        embeddings1 = self._get_embeddings(texts1)
        embeddings2 = self._get_embeddings(texts2)
        
        # Calculate cosine similarities
        similarities = util.cos_sim(embeddings1, embeddings2)
        return [similarities[i][i].item() for i in range(len(texts1))]
    
    def _cross_encoder_similarity(self,
                                texts1: List[str],
                                texts2: List[str]) -> List[float]:
        """Calculate similarity using cross-encoder"""
        scores = self.cross_encoder.predict(
            list(zip(texts1, texts2)),
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        return [float(score) for score in scores]
    
    def _tfidf_similarity(self,
                         texts1: List[str],
                         texts2: List[str]) -> List[float]:
        """Calculate TF-IDF based similarity"""
        similarities = []
        
        for text1, text2 in zip(texts1, texts2):
            tfidf = self.vectorizer.fit_transform([text1, text2])
            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            similarities.append(float(score))
            
        return similarities
    
    def _token_similarity(self,
                         texts1: List[str],
                         texts2: List[str]) -> List[float]:
        """Calculate token overlap based similarity"""
        similarities = []
        
        for text1, text2 in zip(texts1, texts2):
            tokens1 = set(word_tokenize(text1))
            tokens2 = set(word_tokenize(text2))
            
            if not tokens1 or not tokens2:
                similarities.append(0.0)
                continue
                
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            score = len(intersection) / len(union)
            similarities.append(score)
            
        return similarities
    
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get or compute embeddings for texts"""
        if not self.cache_embeddings:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=self.device
            )
            
        # Check cache and compute missing embeddings
        uncached_texts = []
        uncached_indices = []
        embeddings = []
        
        for i, text in enumerate(texts):
            if text in self._embedding_cache:
                embeddings.append(self._embedding_cache[text])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=self.device
            )
            
            # Update cache
            for text, embedding in zip(uncached_texts, new_embeddings):
                self._embedding_cache[text] = embedding
                
            # Insert new embeddings
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings.insert(idx, embedding)
                
        return torch.stack(embeddings)
    
    def find_most_similar(self,
                         query: Union[str, List[str]],
                         candidates: List[str],
                         method: str = 'bi-encoder',
                         top_k: int = 1,
                         threshold: float = 0.0) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Find most similar candidates for query text(s)
        
        Args:
            query: Query text or texts
            candidates: List of candidate texts
            method: Similarity method to use
            top_k: Number of results to return per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of (text, score) tuples or list of lists for multiple queries
        """
        # Handle single query
        if isinstance(query, str):
            query = [query]
            single_query = True
        else:
            single_query = False
            
        all_results = []
        self.progress.start_task(len(query))
        
        # Process in batches
        for i in range(0, len(query), self.batch_size):
            batch_queries = query[i:i + self.batch_size]
            
            # Calculate similarities
            if method == 'bi-encoder':
                # Get embeddings
                query_embeddings = self._get_embeddings(batch_queries)
                candidate_embeddings = self._get_embeddings(candidates)
                
                # Calculate similarities
                similarities = util.cos_sim(query_embeddings, candidate_embeddings)
                
                # Get top k for each query
                batch_results = []
                for q_idx in range(len(batch_queries)):
                    q_scores = similarities[q_idx]
                    top_indices = q_scores.topk(min(top_k, len(candidates)))
                    
                    results = []
                    for idx, score in zip(top_indices.indices, top_indices.values):
                        if score >= threshold:
                            results.append((candidates[idx], score.item()))
                    batch_results.append(results)
                    
            else:
                # Use pairwise comparison for other methods
                batch_results = []
                for q in batch_queries:
                    similarities = []
                    for c in candidates:
                        score = self.get_similarity(q, c, method=method)
                        if score >= threshold:
                            similarities.append((c, score))
                            
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    batch_results.append(similarities[:top_k])
                    
            all_results.extend(batch_results)
            self.progress.update(len(batch_queries))
            
        self.progress.end_task()
        
        return all_results[0] if single_query else all_results
    
    def rank_texts(self,
                  query: str,
                  texts: List[str],
                  method: str = 'bi-encoder') -> List[Tuple[str, float]]:
        """Rank texts by relevance to query
        
        Args:
            query: Query text
            texts: Texts to rank
            method: Ranking method to use
            
        Returns:
            List of (text, score) tuples sorted by relevance
        """
        # Calculate similarities
        scores = self.get_similarity([query] * len(texts), texts, method=method)
        
        # Sort by score
        ranked = list(zip(texts, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def fine_tune(self,
                 train_pairs: List[Tuple[str, str, float]],
                 val_pairs: Optional[List[Tuple[str, str, float]]] = None,
                 epochs: int = 3,
                 learning_rate: float = 2e-5):
        """Fine-tune the similarity model
        
        Args:
            train_pairs: List of (text1, text2, score) tuples
            val_pairs: Optional validation pairs
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        from sentence_transformers import InputExample
        from torch.utils.data import DataLoader
        
        # Prepare training examples
        train_examples = [
            InputExample(texts=[t1, t2], label=score)
            for t1, t2, score in train_pairs
        ]
        
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=self.batch_size
        )
        
        # Prepare validation if provided
        if val_pairs:
            val_examples = [
                InputExample(texts=[t1, t2], label=score)
                for t1, t2, score in val_pairs
            ]
            evaluator = None  # TODO: Add proper evaluator
        else:
            evaluator = None
            
        # Train the model
        warmup_steps = int(len(train_dataloader) * epochs * 0.1)
        
        self.model.fit(
            train_objectives=[(train_dataloader, self.model.get_loss_fn())],
            epochs=epochs,
            evaluator=evaluator,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': learning_rate},
            show_progress_bar=True
        )
        
        # Clear embedding cache after fine-tuning
        self._embedding_cache.clear()