"""
Advanced Sentence Similarity Module for Thai Language
"""

from typing import List, Dict, Union, Optional, Any
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from pythainlp.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from ..core.transformers import TransformerBase

class ThaiSentenceSimilarity(TransformerBase):
    """Advanced sentence similarity for Thai language"""
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        embedding_type: str = "transformer",
        pooling_strategy: str = "mean",
        use_cuda: bool = True,
        cache_size: int = 10000,
        **kwargs
    ):
        """Initialize sentence similarity model
        
        Args:
            model_name_or_path: Name or path of the model
            embedding_type: Type of embeddings to use (transformer or sentence-transformer)
            pooling_strategy: Strategy for pooling token embeddings
            use_cuda: Whether to use GPU if available
            cache_size: Size of embedding cache
            **kwargs: Additional arguments for model initialization
        """
        if model_name_or_path is None:
            model_name_or_path = "airesearch/wangchanberta-base-att-spm-uncased"
            
        self.embedding_type = embedding_type
        self.pooling_strategy = pooling_strategy
        self.cache_size = cache_size
        self.embedding_cache = {}
        
        if embedding_type == "sentence-transformer":
            self.model = SentenceTransformer(model_name_or_path)
            if use_cuda and torch.cuda.is_available():
                self.model = self.model.to("cuda")
        else:
            super().__init__(
                model_name_or_path=model_name_or_path,
                task_type="feature-extraction",
                **kwargs
            )
            
    def _get_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> np.ndarray:
        """Get embedding for text
        
        Args:
            text: Input text
            use_cache: Whether to use embedding cache
            
        Returns:
            Text embedding
        """
        if use_cache and text in self.embedding_cache:
            return self.embedding_cache[text]
            
        if self.embedding_type == "sentence-transformer":
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        else:
            # Tokenize and get model outputs
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Pool token embeddings
            if self.pooling_strategy == "cls":
                embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
            else:  # mean pooling
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = embedding.cpu().numpy()
                
        # Update cache
        if use_cache:
            if len(self.embedding_cache) >= self.cache_size:
                # Remove oldest item
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
            self.embedding_cache[text] = embedding
            
        return embedding
        
    def compute_similarity(
        self,
        text1: str,
        text2: str,
        metric: str = "cosine",
        **kwargs
    ) -> float:
        """Compute similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric to use
            **kwargs: Additional arguments for embedding computation
            
        Returns:
            Similarity score
        """
        # Get embeddings
        emb1 = self._get_embedding(text1, **kwargs)
        emb2 = self._get_embedding(text2, **kwargs)
        
        # Reshape embeddings
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        
        # Compute similarity
        if metric == "cosine":
            return float(cosine_similarity(emb1, emb2)[0, 0])
        elif metric == "euclidean":
            return float(1 / (1 + np.linalg.norm(emb1 - emb2)))
        elif metric == "dot":
            return float(np.dot(emb1, emb2.T)[0, 0])
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
            
    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        metric: str = "cosine",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Find most similar texts from candidates
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return
            metric: Similarity metric to use
            **kwargs: Additional arguments for similarity computation
            
        Returns:
            List of dictionaries with text and similarity score
        """
        # Get query embedding
        query_emb = self._get_embedding(query, **kwargs)
        
        # Get candidate embeddings
        candidate_embs = []
        for text in candidates:
            emb = self._get_embedding(text, **kwargs)
            candidate_embs.append(emb)
            
        # Stack embeddings
        candidate_embs = np.stack(candidate_embs)
        
        # Compute similarities
        if metric == "cosine":
            similarities = cosine_similarity(
                query_emb.reshape(1, -1),
                candidate_embs
            )[0]
        elif metric == "euclidean":
            distances = np.linalg.norm(candidate_embs - query_emb, axis=1)
            similarities = 1 / (1 + distances)
        elif metric == "dot":
            similarities = np.dot(candidate_embs, query_emb)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
            
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            {
                'text': candidates[idx],
                'similarity': float(similarities[idx])
            }
            for idx in top_indices
        ]
        
    def compute_similarity_matrix(
        self,
        texts: List[str],
        metric: str = "cosine",
        **kwargs
    ) -> np.ndarray:
        """Compute similarity matrix for list of texts
        
        Args:
            texts: List of texts
            metric: Similarity metric to use
            **kwargs: Additional arguments for similarity computation
            
        Returns:
            Similarity matrix
        """
        # Get embeddings for all texts
        embeddings = []
        for text in texts:
            emb = self._get_embedding(text, **kwargs)
            embeddings.append(emb)
            
        # Stack embeddings
        embeddings = np.stack(embeddings)
        
        # Compute similarity matrix
        if metric == "cosine":
            return cosine_similarity(embeddings)
        elif metric == "euclidean":
            distances = np.linalg.norm(
                embeddings[:, np.newaxis] - embeddings,
                axis=2
            )
            return 1 / (1 + distances)
        elif metric == "dot":
            return np.dot(embeddings, embeddings.T)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
            
    def cluster_texts(
        self,
        texts: List[str],
        n_clusters: int,
        metric: str = "cosine",
        **kwargs
    ) -> Dict[str, Any]:
        """Cluster texts based on similarity
        
        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters
            metric: Similarity metric to use
            **kwargs: Additional arguments for clustering
            
        Returns:
            Dictionary with clustering results
        """
        from sklearn.cluster import KMeans
        
        # Get embeddings for all texts
        embeddings = []
        for text in texts:
            emb = self._get_embedding(text, **kwargs)
            embeddings.append(emb)
            
        # Stack embeddings
        embeddings = np.stack(embeddings)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            **kwargs
        )
        labels = kmeans.fit_predict(embeddings)
        
        # Group texts by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[i])
            
        return {
            'labels': labels.tolist(),
            'clusters': clusters,
            'centroids': kmeans.cluster_centers_.tolist()
        }
        
    def semantic_search(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 5,
        threshold: float = 0.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for semantically similar texts in corpus
        
        Args:
            query: Query text
            corpus: List of texts to search in
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            **kwargs: Additional arguments for similarity computation
            
        Returns:
            List of search results with text and score
        """
        results = self.find_most_similar(
            query,
            corpus,
            top_k=len(corpus),  # Get all results first
            **kwargs
        )
        
        # Filter by threshold and limit to top k
        filtered = [
            result for result in results
            if result['similarity'] >= threshold
        ][:top_k]
        
        return filtered 