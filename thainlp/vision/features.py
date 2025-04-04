"""
Feature extraction from images
"""
from typing import Dict, List, Union, Optional, Any, Tuple
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    CLIPProcessor,
    CLIPModel
)
from .base import VisionBase

class FeatureExtractor(VisionBase):
    """Extract features from images using vision models"""
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
        pooling_strategy: str = "mean"
    ):
        """Initialize feature extractor
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
            pooling_strategy: Strategy for pooling features (mean, cls, etc.)
        """
        super().__init__(model_name, device, batch_size)
        
        # Load model and processor
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.pooling_strategy = pooling_strategy
        
    def extract(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        return_tensors: bool = False,
        normalize: bool = True
    ) -> Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]]:
        """Extract features from one or more images
        
        Args:
            images: Image or list of images (as paths or PIL Images)
            return_tensors: Whether to return PyTorch tensors instead of numpy arrays
            normalize: Whether to L2-normalize feature vectors
            
        Returns:
            Extracted features as numpy arrays or PyTorch tensors
        """
        # Handle single image input
        if isinstance(images, (str, Image.Image, np.ndarray)):
            images = [images]
            single_input = True
        else:
            single_input = False
            
        def process_batch(batch_images):
            # Preprocess images
            inputs = self.processor(
                batch_images, 
                return_tensors="pt"
            ).to(self.device)
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get feature representations based on pooling strategy
            if self.pooling_strategy == "cls":
                # Use [CLS] token representation
                features = outputs.last_hidden_state[:, 0]
            elif self.pooling_strategy == "mean":
                # Mean pooling over tokens
                features = outputs.last_hidden_state.mean(dim=1)
            else:
                # Default to mean pooling
                features = outputs.last_hidden_state.mean(dim=1)
                
            # Normalize if requested
            if normalize:
                features = features / features.norm(dim=1, keepdim=True)
                
            # Convert to numpy or keep as tensor
            if return_tensors:
                batch_results = [feat.cpu() for feat in features]
            else:
                batch_results = [feat.cpu().numpy() for feat in features]
                
            return batch_results
            
        results = self.batch_process(
            images,
            process_batch,
            preprocess_fn=lambda img: img if isinstance(img, Image.Image) else self.load_image(img)
        )
        
        return results[0] if single_input else results

class EmbeddingExtractor(VisionBase):
    """Extract embeddings for multimodal alignment using CLIP"""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16
    ):
        """Initialize embedding extractor
        
        Args:
            model_name: Name of pretrained CLIP model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name, device, batch_size)
        
        # Load model and processor
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        
    def extract_image_embeddings(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        return_tensors: bool = False,
        normalize: bool = True
    ) -> Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]]:
        """Extract image embeddings
        
        Args:
            images: Image or list of images (as paths or PIL Images)
            return_tensors: Whether to return PyTorch tensors instead of numpy arrays
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Image embeddings
        """
        # Handle single image input
        if isinstance(images, (str, Image.Image, np.ndarray)):
            images = [images]
            single_input = True
        else:
            single_input = False
            
        def process_batch(batch_images):
            # Preprocess images
            inputs = self.processor(
                images=batch_images, 
                return_tensors="pt"
            ).to(self.device)
            
            # Extract features
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                
            # Normalize if requested
            if normalize:
                outputs = outputs / outputs.norm(dim=1, keepdim=True)
                
            # Convert to numpy or keep as tensor
            if return_tensors:
                batch_results = [emb.cpu() for emb in outputs]
            else:
                batch_results = [emb.cpu().numpy() for emb in outputs]
                
            return batch_results
            
        results = self.batch_process(
            images,
            process_batch,
            preprocess_fn=lambda img: img if isinstance(img, Image.Image) else self.load_image(img)
        )
        
        return results[0] if single_input else results
    
    def extract_text_embeddings(
        self,
        texts: Union[str, List[str]],
        return_tensors: bool = False,
        normalize: bool = True
    ) -> Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]]:
        """Extract text embeddings
        
        Args:
            texts: Text or list of texts
            return_tensors: Whether to return PyTorch tensors instead of numpy arrays
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Text embeddings
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
            
        def process_batch(batch_texts):
            # Preprocess texts
            inputs = self.processor(
                text=batch_texts, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Extract features
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                
            # Normalize if requested
            if normalize:
                outputs = outputs / outputs.norm(dim=1, keepdim=True)
                
            # Convert to numpy or keep as tensor
            if return_tensors:
                batch_results = [emb.cpu() for emb in outputs]
            else:
                batch_results = [emb.cpu().numpy() for emb in outputs]
                
            return batch_results
            
        results = self.batch_process(
            texts,
            process_batch
        )
        
        return results[0] if single_input else results
    
    def compute_similarity(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        texts: Union[str, List[str]]
    ) -> Union[float, np.ndarray]:
        """Compute similarity between images and texts
        
        Args:
            images: Image or list of images
            texts: Text or list of texts
            
        Returns:
            Similarity scores
        """
        # Extract embeddings
        image_embeddings = self.extract_image_embeddings(images, return_tensors=True)
        text_embeddings = self.extract_text_embeddings(texts, return_tensors=True)
        
        # Handle different input combinations
        if not isinstance(image_embeddings, list):
            image_embeddings = [image_embeddings]
        if not isinstance(text_embeddings, list):
            text_embeddings = [text_embeddings]
            
        # Stack embeddings
        image_embeddings = torch.stack(image_embeddings)
        text_embeddings = torch.stack(text_embeddings)
        
        # Compute similarity
        similarity = torch.matmul(image_embeddings, text_embeddings.t()).cpu().numpy()
        
        # Return single score or matrix
        if similarity.shape == (1, 1):
            return float(similarity[0, 0])
        return similarity
