"""
Image and video classification models
"""
from typing import Dict, List, Union, Optional, Any, Tuple
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPProcessor,
    CLIPModel
)
from .base import VisionBase

class ImageClassifier(VisionBase):
    """Image classification using transformer models"""
    
    def __init__(
        self,
        model_name: str = "microsoft/resnet-50",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16
    ):
        """Initialize image classifier
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name, device, batch_size)
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
        
    def classify(
        self, 
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        top_k: int = 5
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Classify one or more images
        
        Args:
            images: Image or list of images (as paths or PIL Images)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary or list of dictionaries with class labels and scores
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
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Process each prediction
            batch_results = []
            for logits in outputs.logits:
                probs = torch.softmax(logits, dim=0)
                top_probs, top_idxs = probs.topk(top_k)
                
                results = {}
                for i, (prob, idx) in enumerate(zip(top_probs, top_idxs)):
                    label = self.model.config.id2label[idx.item()]
                    results[label] = prob.item()
                    
                batch_results.append(results)
                
            return batch_results
            
        results = self.batch_process(
            images,
            process_batch,
            preprocess_fn=lambda img: img if isinstance(img, Image.Image) else self.load_image(img)
        )
        
        return results[0] if single_input else results

class ZeroShotImageClassifier(VisionBase):
    """Zero-shot image classification using CLIP"""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8
    ):
        """Initialize zero-shot classifier
        
        Args:
            model_name: Name of pretrained CLIP model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name, device, batch_size)
        
        # Load model and processor
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        
    def classify(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        categories: List[str],
        template: str = "a photo of {}"
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Classify images using zero-shot classification
        
        Args:
            images: Image or list of images (as paths or PIL Images)
            categories: List of category names for classification
            template: Template string for text prompts
            
        Returns:
            Dictionary or list of dictionaries with class labels and scores
        """
        # Handle single image input
        if isinstance(images, (str, Image.Image, np.ndarray)):
            images = [images]
            single_input = True
        else:
            single_input = False
            
        # Create text prompts from categories
        texts = [template.format(category) for category in categories]
        
        def process_batch(batch_images):
            # Preprocess inputs
            inputs = self.processor(
                text=texts,
                images=batch_images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Process each prediction
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            similarity = image_features @ text_features.T
            
            batch_results = []
            for sim_scores in similarity:
                # Convert to probabilities
                probs = torch.softmax(sim_scores, dim=0)
                
                # Create results dictionary
                results = {}
                for category, prob in zip(categories, probs):
                    results[category] = prob.item()
                    
                batch_results.append(results)
                
            return batch_results
            
        results = self.batch_process(
            images,
            process_batch,
            preprocess_fn=lambda img: img if isinstance(img, Image.Image) else self.load_image(img)
        )
        
        return results[0] if single_input else results

class VideoClassifier(VisionBase):
    """Video classification using transformer models"""
    
    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4,
        num_frames: int = 16
    ):
        """Initialize video classifier
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
            num_frames: Number of frames to sample from video
        """
        super().__init__(model_name, device, batch_size)
        
        # Load model and processor
        try:
            from transformers import AutoProcessor, AutoModelForVideoClassification
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVideoClassification.from_pretrained(model_name).to(device)
        except ImportError:
            raise ImportError("Video classification requires transformers >= 4.21.0")
            
        self.num_frames = num_frames
        
    def load_video(self, video_path: str) -> List[np.ndarray]:
        """Load video frames from a file
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of numpy arrays representing frames
        """
        try:
            import decord
        except ImportError:
            raise ImportError("Video loading requires decord library")
            
        # Load video and extract frames
        video = decord.VideoReader(video_path)
        total_frames = len(video)
        
        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = video.get_batch(indices).asnumpy()
        
        return frames
    
    def classify(
        self,
        videos: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
        top_k: int = 5
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Classify one or more videos
        
        Args:
            videos: Video path, frames, or list of videos
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary or list of dictionaries with class labels and scores
        """
        # Handle single video input
        if isinstance(videos, (str, np.ndarray)):
            videos = [videos]
            single_input = True
        else:
            single_input = False
            
        def process_batch(batch_videos):
            # Preprocess videos
            inputs = self.processor(
                batch_videos, 
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Process each prediction
            batch_results = []
            for logits in outputs.logits:
                probs = torch.softmax(logits, dim=0)
                top_probs, top_idxs = probs.topk(top_k)
                
                results = {}
                for i, (prob, idx) in enumerate(zip(top_probs, top_idxs)):
                    label = self.model.config.id2label[idx.item()]
                    results[label] = prob.item()
                    
                batch_results.append(results)
                
            return batch_results
            
        results = self.batch_process(
            videos,
            process_batch,
            preprocess_fn=lambda video: (
                video if isinstance(video, np.ndarray) else self.load_video(video)
            )
        )
        
        return results[0] if single_input else results
