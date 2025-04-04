"""
Base classes for computer vision tasks
"""
from typing import Any, Dict, List, Optional, Union, Tuple
import os
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from ..core.transformers import TransformerBase
from ..extensions.monitoring import ProgressTracker

@dataclass
class VisionConfig:
    """Configuration for vision models"""
    model_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    image_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    max_length: int = 77  # For text inputs with vision models

class VisionBase:
    """Base class for vision tasks"""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8
    ):
        """Initialize vision model
        
        Args:
            model_name: Name of vision model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        self.config = VisionConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size
        )
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.processor = None
        self.progress = ProgressTracker()
        
    def load_image(self, image_path: str) -> Image.Image:
        """Load an image from a file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        return Image.open(image_path).convert("RGB")
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for model input
        
        Args:
            image: Image as path, PIL Image, or numpy array
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, str):
            image = self.load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        if self.processor is not None:
            return self.processor(image, return_tensors="pt").to(self.device)
        
        # Default preprocessing if no processor available
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ])
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def batch_process(self, 
                     items: List[Any], 
                     process_fn: callable, 
                     preprocess_fn: Optional[callable] = None,
                     postprocess_fn: Optional[callable] = None) -> List[Any]:
        """Process items in batches
        
        Args:
            items: List of items to process
            process_fn: Function to process each batch
            preprocess_fn: Optional function to preprocess each item
            postprocess_fn: Optional function to postprocess results
            
        Returns:
            List of processed results
        """
        all_results = []
        self.progress.start_task(len(items))
        
        for i in range(0, len(items), self.batch_size):
            batch_items = items[i:i + self.batch_size]
            
            # Preprocess if needed
            if preprocess_fn:
                batch_items = [preprocess_fn(item) for item in batch_items]
                
            # Process batch
            batch_results = process_fn(batch_items)
            
            # Postprocess if needed
            if postprocess_fn:
                batch_results = [postprocess_fn(res) for res in batch_results]
                
            all_results.extend(batch_results)
            self.progress.update(len(batch_items))
            
        self.progress.end_task()
        return all_results
