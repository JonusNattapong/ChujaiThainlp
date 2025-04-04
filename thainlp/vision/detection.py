"""
Object detection and keypoint detection models
"""
from typing import Dict, List, Union, Optional, Any, Tuple
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoImageProcessor, 
    AutoModelForObjectDetection,
    DetrForObjectDetection,
    OwlViTProcessor,
    OwlViTForObjectDetection
)
from .base import VisionBase

class ObjectDetector(VisionBase):
    """Object detection using transformer models"""
    
    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8,
        threshold: float = 0.5
    ):
        """Initialize object detector
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
            threshold: Confidence threshold for detections
        """
        super().__init__(model_name, device, batch_size)
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name).to(device)
        self.threshold = threshold
        
    def detect(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        threshold: Optional[float] = None
    ) -> Union[List[Dict], List[List[Dict]]]:
        """Detect objects in one or more images
        
        Args:
            images: Image or list of images (as paths or PIL Images)
            threshold: Optional confidence threshold override
            
        Returns:
            List of detections or list of lists of detections
        """
        # Use instance threshold or override
        threshold = threshold if threshold is not None else self.threshold
        
        # Handle single image input
        if isinstance(images, (str, Image.Image, np.ndarray)):
            images = [images]
            single_input = True
        else:
            single_input = False
            
        def process_batch(batch_images):
            # Get image sizes for scaling bounding boxes
            sizes = [(img.width, img.height) for img in batch_images]
            
            # Preprocess images
            inputs = self.processor(
                batch_images, 
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Process predictions for each image
            batch_results = []
            for i, (preds, size) in enumerate(zip(outputs.pred_boxes, sizes)):
                # Get scores and labels
                scores = outputs.pred_scores[i]
                labels = outputs.pred_labels[i]
                boxes = outputs.pred_boxes[i]
                
                # Filter by threshold
                mask = scores >= threshold
                filtered_scores = scores[mask]
                filtered_labels = labels[mask]
                filtered_boxes = boxes[mask]
                
                # Convert normalized boxes to pixel coordinates
                scaled_boxes = []
                for box in filtered_boxes:
                    x0, y0, x1, y1 = box.tolist()
                    scaled_boxes.append([
                        x0 * size[0],  # x_min
                        y0 * size[1],  # y_min
                        x1 * size[0],  # x_max
                        y1 * size[1]   # y_max
                    ])
                
                # Create detection results
                detections = []
                for score, label_id, box in zip(filtered_scores, filtered_labels, scaled_boxes):
                    label = self.model.config.id2label[label_id.item()]
                    detections.append({
                        "label": label,
                        "score": score.item(),
                        "box": box
                    })
                    
                batch_results.append(detections)
                
            return batch_results
        
        results = self.batch_process(
            images,
            process_batch,
            preprocess_fn=lambda img: img if isinstance(img, Image.Image) else self.load_image(img)
        )
        
        return results[0] if single_input else results

class ZeroShotObjectDetector(VisionBase):
    """Zero-shot object detection using open-vocabulary models"""
    
    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8,
        threshold: float = 0.1
    ):
        """Initialize zero-shot object detector
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
            threshold: Confidence threshold for detections
        """
        super().__init__(model_name, device, batch_size)
        
        # Load model and processor
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
        self.threshold = threshold
        
    def detect(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        categories: List[str],
        threshold: Optional[float] = None
    ) -> Union[List[Dict], List[List[Dict]]]:
        """Detect objects in one or more images
        
        Args:
            images: Image or list of images (as paths or PIL Images)
            categories: List of category names to detect
            threshold: Optional confidence threshold override
            
        Returns:
            List of detections or list of lists of detections
        """
        # Use instance threshold or override
        threshold = threshold if threshold is not None else self.threshold
        
        # Handle single image input
        if isinstance(images, (str, Image.Image, np.ndarray)):
            images = [images]
            single_input = True
        else:
            single_input = False
            
        def process_batch(batch_images):
            # Preprocess images and text
            inputs = self.processor(
                text=categories,
                images=batch_images, 
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Process each prediction
            batch_results = []
            for i, image in enumerate(batch_images):
                # Get image size for scaling
                size = (image.width, image.height)
                
                # Get target sizes for post-processing
                target_sizes = torch.Tensor([size]).to(self.device)
                
                # Convert outputs
                results = self.processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=threshold
                )[i]
                
                # Create detection list
                detections = []
                for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                    detections.append({
                        "label": categories[label],
                        "score": score.item(),
                        "box": box.tolist()
                    })
                    
                batch_results.append(detections)
                
            return batch_results
            
        results = self.batch_process(
            images,
            process_batch,
            preprocess_fn=lambda img: img if isinstance(img, Image.Image) else self.load_image(img)
        )
        
        return results[0] if single_input else results

class KeypointDetector(VisionBase):
    """Keypoint detection for human pose estimation"""
    
    def __init__(
        self,
        model_name: str = "google/deplot",  # Using DeepLot for keypoints
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8
    ):
        """Initialize keypoint detector
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name, device, batch_size)
        
        try:
            # Load model and processor
            from transformers import AutoProcessor, AutoModelForKeypointDetection
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForKeypointDetection.from_pretrained(model_name).to(device)
        except (ImportError, ValueError):
            # Fallback to more specialized models if needed
            from transformers import DetrImageProcessor, DetrForObjectDetection
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
            print("Warning: Using fallback model for keypoint detection")
        
    def detect(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]]
    ) -> Union[List[Dict], List[List[Dict]]]:
        """Detect keypoints in one or more images
        
        Args:
            images: Image or list of images (as paths or PIL Images)
            
        Returns:
            List of keypoints or list of lists of keypoints
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
            for i, image in enumerate(batch_images):
                # Get image size for scaling
                size = (image.width, image.height)
                
                # Process keypoints (model-specific implementation)
                if hasattr(outputs, 'pred_keypoints'):
                    # For models with keypoint outputs
                    keypoints = outputs.pred_keypoints[i].cpu().numpy()
                    scores = outputs.pred_keypoint_scores[i].cpu().numpy()
                    
                    # Scale keypoints to image size
                    scaled_keypoints = keypoints * np.array([size[0], size[1]])
                    
                    # Create keypoint results
                    keypoint_results = []
                    for person_idx in range(len(keypoints)):
                        person_keypoints = []
                        for kp_idx, (kp, score) in enumerate(zip(scaled_keypoints[person_idx], scores[person_idx])):
                            keypoint_name = self.model.config.id2label.get(kp_idx, f"keypoint_{kp_idx}")
                            person_keypoints.append({
                                "name": keypoint_name,
                                "position": kp.tolist(),
                                "score": score.item()
                            })
                        keypoint_results.append({
                            "person_id": person_idx,
                            "keypoints": person_keypoints
                        })
                else:
                    # Fallback for models without explicit keypoint outputs
                    # This is a simplified representation for demo purposes
                    keypoint_results = [{
                        "message": "Using fallback detection model - keypoint-specific model required",
                        "detected_objects": len(outputs.pred_boxes[i])
                    }]
                    
                batch_results.append(keypoint_results)
                
            return batch_results
        
        results = self.batch_process(
            images,
            process_batch,
            preprocess_fn=lambda img: img if isinstance(img, Image.Image) else self.load_image(img)
        )
        
        return results[0] if single_input else results
