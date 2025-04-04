"""
Visual document retrieval for finding and ranking documents based on content and layout
"""
from typing import Dict, List, Union, Optional, Any, Tuple
import os
import torch
import numpy as np
import json
import pickle
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModel,
    LayoutLMv3Processor,
    LayoutLMv3Model
)
from pathlib import Path
from .base import MultimodalBase
from .document_qa import DocumentProcessor

class VisualDocumentRetriever(MultimodalBase):
    """Retrieve documents based on visual and textual content"""
    
    def __init__(
        self,
        text_model: str = "sentence-transformers/all-mpnet-base-v2",
        layout_model: str = "microsoft/layoutlmv3-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8
    ):
        """Initialize document retriever
        
        Args:
            text_model: Name of pretrained text embedding model
            layout_model: Name of pretrained layout embedding model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(text_model, device, batch_size)
        
        # Load text embedding model
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_model = AutoModel.from_pretrained(text_model).to(device)
        
        # Store layout model name for lazy loading
        self.layout_model_name = layout_model
        self.layout_processor = None
        self.layout_model = None
        
        # Initialize document processor
        self.doc_processor = DocumentProcessor(device=device)
        
    def _load_layout_model(self):
        """Load layout model on demand"""
        if self.layout_model is None:
            try:
                self.layout_processor = LayoutLMv3Processor.from_pretrained(self.layout_model_name)
                self.layout_model = LayoutLMv3Model.from_pretrained(self.layout_model_name).to(self.device)
            except (ImportError, OSError):
                print(f"Warning: Could not load {self.layout_model_name}. Using text-only retrieval.")
    
    def retrieve(
        self,
        query: Union[str, Dict[str, Any]],
        index_path: Optional[str] = None,
        document_paths: Optional[List[str]] = None,
        top_k: int = 5,
        use_layout: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve documents similar to query
        
        Args:
            query: Text query or document dictionary
            index_path: Path to index file
            document_paths: List of document paths (used if no index)
            top_k: Number of top results to return
            use_layout: Whether to use layout information
            
        Returns:
            List of retrieved documents with similarity scores
        """
        # Ensure we have either index or document paths
        if index_path is None and document_paths is None:
            raise ValueError("Either index_path or document_paths must be provided")
            
        # Load index if path is provided
        if index_path is not None:
            index_data = self._load_index(index_path)
            document_embeddings = index_data["embeddings"]
            document_info = index_data["documents"]
        else:
            # Process documents and create embeddings
            processed_docs = []
            for doc_path in document_paths:
                doc_data = self.doc_processor.process(doc_path)
                processed_docs.append(doc_data)
                
            # Generate embeddings
            document_embeddings = self._encode_documents(processed_docs, use_layout)
            document_info = [{"path": path, "metadata": self._extract_metadata(doc)} 
                           for path, doc in zip(document_paths, processed_docs)]
            
        # Process query
        if isinstance(query, str):
            query_embedding = self._encode_text(query)
        else:
            # Process query as a document
            query_embedding = self._encode_documents([query], use_layout)[0]
            
        # Calculate similarities
        similarities = self._calculate_similarities(query_embedding, document_embeddings)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        # Create result list
        results = []
        for idx, score in zip(top_indices, top_scores):
            results.append({
                "document": document_info[idx],
                "score": float(score),
                "similarity": float(score)
            })
            
        return results
    
    def _load_index(self, index_path: str) -> Dict[str, Any]:
        """Load document index from file"""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
            
        # Load index based on file extension
        ext = os.path.splitext(index_path)[1].lower()
        
        if ext == '.pkl':
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
        elif ext == '.json':
            with open(index_path, 'r') as f:
                index_data = json.load(f)
                # Convert string arrays to numpy arrays
                if "embeddings" in index_data:
                    index_data["embeddings"] = np.array([
                        np.array(emb) for emb in index_data["embeddings"]
                    ])
        else:
            raise ValueError(f"Unsupported index format: {ext}")
            
        return index_data
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding"""
        # Tokenize
        inputs = self.text_tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Normalize
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        
        return embeddings[0].cpu().numpy()
    
    def _encode_documents(
        self,
        documents: List[Dict[str, Any]],
        use_layout: bool = True
    ) -> np.ndarray:
        """Encode documents to embeddings"""
        embeddings = []
        
        for doc in documents:
            # Extract text from document
            if "text" in doc:
                if isinstance(doc["text"], list):
                    text = " ".join(doc["text"])
                else:
                    text = doc["text"]
            else:
                text = ""
                
            # Get text embedding
            text_embedding = self._encode_text(text)
            
            # Get layout embedding if requested
            if use_layout and "images" in doc and doc["images"] and self.layout_model is not None:
                # Load layout model if needed
                self._load_layout_model()
                
                # Process first image with layout
                img = doc["images"][0]["image"] if isinstance(doc["images"][0], dict) else doc["images"][0]
                
                # Process with LayoutLM
                inputs = self.layout_processor(
                    img,
                    text,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.layout_model(**inputs)
                    
                # Mean pooling
                layout_embedding = outputs.last_hidden_state.mean(dim=1)
                
                # Normalize
                layout_embedding = layout_embedding / layout_embedding.norm(dim=1, keepdim=True)
                layout_embedding = layout_embedding[0].cpu().numpy()
                
                # Combine embeddings (simple average)
                embedding = (text_embedding + layout_embedding) / 2
            else:
                # Use text-only embedding
                embedding = text_embedding
                
            embeddings.append(embedding)
            
        return np.array(embeddings)
    
    def _calculate_similarities(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarities between query and documents"""
        # Ensure query embedding is 1D
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding.flatten()
            
        # Calculate dot products
        dot_products = np.dot(document_embeddings, query_embedding)
        
        # Normalize if not already normalized
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(document_embeddings, axis=1)
        
        # Avoid division by zero
        doc_norms = np.maximum(doc_norms, 1e-10)
        query_norm = max(query_norm, 1e-10)
        
        # Calculate cosine similarities
        similarities = dot_products / (doc_norms * query_norm)
        
        return similarities
    
    def _extract_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from document"""
        metadata = {}
        
        # Extract title
        if "metadata" in document and "title" in document["metadata"]:
            metadata["title"] = document["metadata"]["title"]
        else:
            metadata["title"] = "Untitled Document"
            
        # Extract page count
        if "metadata" in document and "pages" in document["metadata"]:
            metadata["pages"] = document["metadata"]["pages"]
        elif "images" in document:
            metadata["pages"] = len(document["images"])
        else:
            metadata["pages"] = 1
            
        # Extract document type
        if "type" in document:
            metadata["type"] = document["type"]
            
        return metadata

class DocumentIndexer(VisualDocumentRetriever):
    """Index documents for efficient retrieval"""
    
    def __init__(
        self,
        text_model: str = "sentence-transformers/all-mpnet-base-v2",
        layout_model: str = "microsoft/layoutlmv3-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8
    ):
        super().__init__(text_model, layout_model, device, batch_size)
    
    def index(
        self,
        document_paths: List[str],
        index_path: Optional[str] = None,
        use_layout: bool = True
    ) -> Dict[str, Any]:
        """Index documents for future retrieval
        
        Args:
            document_paths: List of paths to documents
            index_path: Path to save index (if None, index is returned)
            use_layout: Whether to use layout information
            
        Returns:
            Dictionary with index data
        """
        # Process all documents
        processed_docs = []
        for doc_path in self.progress.track(document_paths, description="Processing documents"):
            try:
                doc_data = self.doc_processor.process(doc_path)
                processed_docs.append(doc_data)
            except Exception as e:
                print(f"Error processing {doc_path}: {e}")
                # Add empty document as placeholder
                processed_docs.append({"text": "", "path": doc_path})
                
        # Generate embeddings
        self.progress.start_task(len(processed_docs), description="Generating embeddings")
        document_embeddings = self._encode_documents(processed_docs, use_layout)
        document_info = []
        
        # Extract metadata
        for path, doc in zip(document_paths, processed_docs):
            metadata = self._extract_metadata(doc)
            document_info.append({
                "path": path,
                "metadata": metadata
            })
            self.progress.update(1)
            
        self.progress.end_task()
        
        # Create index data
        index_data = {
            "embeddings": document_embeddings,
            "documents": document_info,
            "metadata": {
                "count": len(document_paths),
                "use_layout": use_layout,
                "text_model": self.config.model_name,
                "layout_model": self.layout_model_name if use_layout else None
            }
        }
        
        # Save index if path is provided
        if index_path is not None:
            self._save_index(index_data, index_path)
            
        return index_data
    
    def _save_index(self, index_data: Dict[str, Any], index_path: str) -> None:
        """Save document index to file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        
        # Save based on file extension
        ext = os.path.splitext(index_path)[1].lower()
        
        if ext == '.pkl':
            with open(index_path, 'wb') as f:
                pickle.dump(index_data, f)
        elif ext == '.json':
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = index_data.copy()
            serializable_data["embeddings"] = [emb.tolist() for emb in index_data["embeddings"]]
            
            with open(index_path, 'w') as f:
                json.dump(serializable_data, f)
        else:
            # Default to pickle
            with open(index_path + '.pkl', 'wb') as f:
                pickle.dump(index_data, f)
