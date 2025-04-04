"""
Image segmentation models (semantic, instance, and panoptic)
"""
from typing import Dict, List, Union, Optional, Any, Tuple
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    AutoModelForInstanceSegmentation,
    MaskFormerForInstanceSegmentation,
    MaskFormerImageProcessor,
    SamProcessor,
    SamModel
)
from .base import VisionBase

class ImageSegmenter(VisionBase):
    """Semantic segmentation using transformer models"""
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        """Initialize semantic segmenter
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name, device, batch_size)
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)
        
    def segment(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        return_tensors: bool = False
    ) -> Union[np.ndarray, List[np.ndarray], Dict, List[Dict]]:
        """Segment one or more images
        
        Args:
            images: Image or list of images (as paths or PIL Images)
            return_tensors: Whether to return raw tensor outputs
            
        Returns:
            Segmentation maps or dictionaries with segmentation information
        """
        # Handle single image input
        if isinstance(images, (str, Image.Image, np.ndarray)):
            images = [images]
            single_input = True
        else:
            single_input = False
            
        def process_batch(batch_images):
            # Original sizes for rescaling
            sizes = [(img.width, img.height) for img in batch_images]
            
            # Preprocess images
            inputs = self.processor(
                batch_images, 
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Post-process predictions
            batch_results = []
            for i, size in enumerate(sizes):
                # Get segmentation logits
                logits = outputs.logits[i]
                
                if return_tensors:
                    # Return raw tensor
                    batch_results.append({
                        "logits": logits.cpu(),
                        "id2label": self.model.config.id2label
                    })
                else:
                    # Convert to segmentation map
                    seg_map = logits.argmax(dim=0).cpu().numpy()
                    
                    # Resize to original image size
                    from PIL import Image
                    seg_map_img = Image.fromarray(seg_map.astype(np.uint8))
                    seg_map_img = seg_map_img.resize(size, Image.NEAREST)
                    seg_map = np.array(seg_map_img)
                    
                    # Create color-coded segmentation map
                    colored_map = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
                    
                    # Add class information
                    classes = {}
                    for class_idx in np.unique(seg_map):
                        if class_idx in self.model.config.id2label:
                            class_name = self.model.config.id2label[int(class_idx)]
                            # Generate a color based on class idx
                            color = [(class_idx * 50) % 256, (class_idx * 100) % 256, (class_idx * 150) % 256]
                            classes[int(class_idx)] = {
                                "name": class_name,
                                "color": color
                            }
                            # Apply color to segmentation map
                            colored_map[seg_map == class_idx] = color
                    
                    batch_results.append({
                        "segmentation_map": seg_map,
                        "colored_map": colored_map,
                        "classes": classes
                    })
                    
            return batch_results
        
        results = self.batch_process(
            images,
            process_batch,
            preprocess_fn=lambda img: img if isinstance(img, Image.Image) else self.load_image(img)
        )
        
        return results[0] if single_input else results

class InstanceSegmenter(VisionBase):
    """Instance segmentation using transformer models"""
    
    def __init__(
        self,
        model_name: str = "facebook/maskformer-swin-base-ade",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4,
        threshold: float = 0.5
    ):
        """Initialize instance segmenter
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
            threshold: Confidence threshold for instance masks
        """
        super().__init__(model_name, device, batch_size)
        
        # Load model and processor
        self.processor = MaskFormerImageProcessor.from_pretrained(model_name)
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(model_name).to(device)
        self.threshold = threshold
        
    def segment(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        threshold: Optional[float] = None
    ) -> Union[Dict, List[Dict]]:
        """Segment instances in one or more images
        
        Args:
            images: Image or list of images (as paths or PIL Images)
            threshold: Optional confidence threshold override
            
        Returns:
            Dictionary or list of dictionaries with instance masks
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
            # Original sizes for rescaling
            sizes = [(img.width, img.height) for img in batch_images]
            
            # Preprocess images
            inputs = self.processor(
                batch_images, 
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Post-process predictions
            batch_results = []
            for i, size in enumerate(sizes):
                # Process instance predictions
                result = self.processor.post_process_instance_segmentation(
                    outputs,
                    target_sizes=[size],
                    threshold=threshold
                )[i]
                
                instances = []
                for mask, label, score in zip(
                    result["masks"],
                    result["labels"],
                    result["scores"]
                ):
                    class_name = self.model.config.id2label[label.item()]
                    instances.append({
                        "mask": mask.cpu().numpy(),
                        "label": class_name,
                        "score": score.item()
                    })
                
                batch_results.append({
                    "instances": instances,
                    "size": size
                })
                    
            return batch_results
        
        results = self.batch_process(
            images,
            process_batch,
            preprocess_fn=lambda img: img if isinstance(img, Image.Image) else self.load_image(img)
        )
        
        return results[0] if single_input else results

class PanopticSegmenter(VisionBase):
    """Panoptic segmentation using transformer models"""
    
    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50-panoptic",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        """Initialize panoptic segmenter
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name, device, batch_size)
        
        try:
            from transformers import DetrImageProcessor, DetrForPanopticSegmentation
            # Load model and processor
            self.processor = DetrImageProcessor.from_pretrained(model_name)
            self.model = DetrForPanopticSegmentation.from_pretrained(model_name).to(device)
        except (ImportError, ValueError):
            # Fallback to maskformer if detr not available
            self.processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
            self.model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco").to(device)
            print("Warning: Using fallback model for panoptic segmentation")
        
    def segment(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]]
    ) -> Union[Dict, List[Dict]]:
        """Generate panoptic segmentation for one or more images
        
        Args:
            images: Image or list of images (as paths or PIL Images)
            
        Returns:
            Dictionary or list of dictionaries with panoptic segmentation
        """
        # Handle single image input
        if isinstance(images, (str, Image.Image, np.ndarray)):
            images = [images]
            single_input = True
        else:
            single_input = False
            
        def process_batch(batch_images):
            # Original sizes for rescaling
            sizes = [(img.width, img.height) for img in batch_images]
            
            # Preprocess images
            inputs = self.processor(
                batch_images, 
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Post-process predictions
            batch_results = []
            for i, size in enumerate(sizes):
                # Process panoptic segmentation
                # The exact post-processing depends on the model
                if hasattr(self.processor, "post_process_panoptic_segmentation"):
                    result = self.processor.post_process_panoptic_segmentation(
                        outputs,
                        target_sizes=[size]
                    )[i]
                    
                    batch_results.append({
                        "segmentation": result["segmentation"].cpu().numpy(),
                        "segments_info": result["segments_info"]
                    })
                else:
                    # Fallback for models without panoptic post-processing
                    result = self.processor.post_process_instance_segmentation(
                        outputs,
                        target_sizes=[size]
                    )[i]
                    
                    instances = []
                    for mask, label, score in zip(
                        result["masks"],
                        result["labels"],
                        result["scores"]
                    ):
                        class_name = self.model.config.id2label[label.item()]
                        instances.append({
                            "mask": mask.cpu().numpy(),
                            "label": class_name,
                            "score": score.item(),
                            "is_thing": True  # Assume all are "things" in fallback
                        })
                    
                    batch_results.append({
                        "instances": instances,
                        "size": size
                    })
                    
            return batch_results
        
        results = self.batch_process(
            images,
            process_batch,
            preprocess_fn=lambda img: img if isinstance(img, Image.Image) else self.load_image(img)
        )
        
        return results[0] if single_input else results

class MaskGenerator(VisionBase):
    """Generate masks for images using SAM (Segment Anything)"""
    
    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        """Initialize mask generator
        
        Args:
            model_name: Name of pretrained SAM model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name, device, batch_size)
        
        # Load model and processor
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name).to(device)
        
    def generate(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        points: Optional[List[Tuple[int, int]]] = None,
        boxes: Optional[List[List[int]]] = None,
        masks: Optional[np.ndarray] = None
    ) -> Union[Dict, List[Dict]]:
        """Generate masks based on input prompts
        
        Args:
            images: Image or list of images (as paths or PIL Images)
            points: Optional list of point prompts
            boxes: Optional list of bounding box prompts
            masks: Optional mask prompt
            
        Returns:
            Dictionary or list of dictionaries with generated masks
        """
        # Handle single image input
        if isinstance(images, (str, Image.Image, np.ndarray)):
            images = [images]
            single_input = True
            # Make sure prompts are in list format for a single image
            if points is not None and not isinstance(points[0][0], list):
                points = [points]
            if boxes is not None and not isinstance(boxes[0][0], list):
                boxes = [boxes]
            if masks is not None and not isinstance(masks, list):
                masks = [masks]
        else:
            single_input = False
            
        def process_batch(batch_images):
            # Extract image sizes for reference
            sizes = [(img.width, img.height) for img in batch_images]
            
            batch_results = []
            # Process each image individually due to different prompts
            for i, image in enumerate(batch_images):
                image_points = points[i] if points is not None and i < len(points) else None
                image_boxes = boxes[i] if boxes is not None and i < len(boxes) else None
                image_masks = masks[i] if masks is not None and i < len(masks) else None
                
                # Prepare processor inputs
                inputs = self.processor(
                    image,
                    return_tensors="pt"
                ).to(self.device)
                
                # Add prompts if provided
                if image_points is not None:
                    point_coords = torch.tensor([p[:2] for p in image_points]).to(self.device)
                    point_labels = torch.tensor([p[2] if len(p) > 2 else 1 for p in image_points]).to(self.device)
                    inputs["point_coords"] = point_coords.unsqueeze(0)
                    inputs["point_labels"] = point_labels.unsqueeze(0)
                
                if image_boxes is not None:
                    boxes_tensor = torch.tensor(image_boxes).to(self.device)
                    inputs["boxes"] = boxes_tensor.unsqueeze(0)
                
                if image_masks is not None:
                    masks_tensor = torch.tensor(image_masks).to(self.device)
                    inputs["mask_inputs"] = masks_tensor.unsqueeze(0)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Process masks
                masks = self.processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )
                
                # Get scores
                scores = outputs.iou_scores.cpu().numpy()
                
                # Create results
                masks_list = []
                for mask_idx, (mask, score) in enumerate(zip(masks[0], scores[0])):
                    masks_list.append({
                        "mask": mask.numpy(),
                        "score": score.item(),
                        "id": mask_idx
                    })
                
                batch_results.append({
                    "masks": masks_list,
                    "size": sizes[i]
                })
                
            return batch_results
        
        results = self.batch_process(
            images,
            process_batch,
            preprocess_fn=lambda img: img if isinstance(img, Image.Image) else self.load_image(img)
        )
        
        return results[0] if single_input else results
