"""
Image-Text processing for OCR, captioning, and analysis
"""
from typing import Dict, List, Union, Optional, Any, Tuple
import torch
import numpy as np
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoModelForImageClassification
)
from .base import MultimodalBase

class ImageTextProcessor(MultimodalBase):
    """Process images for text extraction, captioning, and analysis"""
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        ocr_model: str = "microsoft/trocr-large-printed",
        analysis_model: str = "google/vit-large-patch16-224",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        """Initialize image text processor
        
        Args:
            model_name: Name of pretrained captioning model
            ocr_model: Name of pretrained OCR model
            analysis_model: Name of pretrained analysis model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name, device, batch_size)
        
        # Initialize core model for captioning
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        
        # Store model names for OCR and analysis
        self.ocr_model_name = ocr_model
        self.analysis_model_name = analysis_model
        
        # Initialize OCR components (loaded on demand)
        self.ocr_processor = None
        self.ocr_model = None
        
        # Initialize analysis components (loaded on demand)
        self.analysis_processor = None
        self.analysis_model = None
    
    def _load_ocr_model(self):
        """Load OCR model on demand"""
        if self.ocr_processor is None:
            self.ocr_processor = TrOCRProcessor.from_pretrained(self.ocr_model_name)
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained(self.ocr_model_name).to(self.device)
    
    def _load_analysis_model(self):
        """Load analysis model on demand"""
        if self.analysis_processor is None:
            try:
                from transformers import AutoFeatureExtractor
                self.analysis_processor = AutoFeatureExtractor.from_pretrained(self.analysis_model_name)
                self.analysis_model = AutoModelForImageClassification.from_pretrained(self.analysis_model_name).to(self.device)
            except:
                # Fallback to ViT feature extractor
                self.analysis_processor = ViTFeatureExtractor.from_pretrained(self.analysis_model_name)
                self.analysis_model = AutoModelForImageClassification.from_pretrained(self.analysis_model_name).to(self.device)
    
    def caption(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        prompt: str = "A photo of",
        max_length: int = 30,
        num_captions: int = 1
    ) -> Union[str, List[str], List[List[str]]]:
        """Generate captions for images
        
        Args:
            images: Image or list of images (as paths or PIL Images)
            prompt: Text prompt to condition the generation
            max_length: Maximum length of generated text
            num_captions: Number of captions to generate per image
            
        Returns:
            Caption or list of captions
        """
        # Handle single image input
        if isinstance(images, (str, Image.Image)):
            images = [images]
            single_input = True
        else:
            single_input = False
            
        def process_batch(batch_images):
            batch_results = []
            
            for image in batch_images:
                # Preprocess image if it's a path
                if isinstance(image, str):
                    image = self.load_image(image)
                
                # Prepare inputs
                inputs = self.processor(
                    image, 
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate captions
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_return_sequences=num_captions,
                        do_sample=num_captions > 1,
                        top_k=50,
                        top_p=0.95
                    )
                
                # Decode captions
                captions = self.processor.batch_decode(
                    outputs, 
                    skip_special_tokens=True
                )
                
                if num_captions == 1:
                    batch_results.append(captions[0])
                else:
                    batch_results.append(captions)
            
            return batch_results
            
        results = self.batch_process(
            images,
            process_batch
        )
        
        return results[0] if single_input else results
    
    def extract_text(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        return_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract text from images using OCR.

        Args:
            images: Image or list of images (as paths or PIL Images).
            return_confidence: Whether to return confidence scores.

        Returns:
            A list of dictionaries. Each dictionary corresponds to an input image
            and contains at least the key 'text' with the extracted string.
            If return_confidence is True, it also includes the key 'confidence'.
        """
        # Load OCR model if needed
        self._load_ocr_model()
        
        # Ensure images is a list
        if isinstance(images, (str, Image.Image)):
            images = [images]
            
        def process_batch(batch_images):
            batch_results = []
            
            for image in batch_images:
                # Preprocess image if it's a path
                if isinstance(image, str):
                    image = self.load_image(image)
                
                # Prepare inputs
                pixel_values = self.ocr_processor(image, return_tensors="pt").pixel_values.to(self.device)
                
                # Extract text
                with torch.no_grad():
                    generated_ids = self.ocr_model.generate(pixel_values)
                    
                # Decode text
                generated_text = self.ocr_processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]
                
                result_dict = {"text": generated_text}
                if return_confidence:
                    # Compute rough confidence score based on token probabilities
                    # This is a simplified approach since TrOCR doesn't provide direct confidence
                    try:
                        with torch.no_grad():
                            # Ensure model has necessary config attributes
                            decoder_start_token_id = getattr(self.ocr_model.config, 'decoder_start_token_id', self.ocr_processor.tokenizer.bos_token_id)
                            if decoder_start_token_id is None:
                                # Fallback if BOS token ID is also missing (unlikely)
                                decoder_start_token_id = 0
                                
                            decoder_input_ids = torch.tensor([[decoder_start_token_id]]).to(self.device)

                            outputs = self.ocr_model(
                                pixel_values=pixel_values,
                                decoder_input_ids=decoder_input_ids
                            )
                            logits = outputs.logits
                            # Get probabilities for the generated sequence tokens
                            # We need the generated_ids to align probabilities
                            probs = torch.softmax(logits, dim=-1)
                            # Ensure generated_ids is on the same device and has the correct shape
                            generated_ids_for_gather = generated_ids.to(probs.device)
                            if len(generated_ids_for_gather.shape) == 1:
                                generated_ids_for_gather = generated_ids_for_gather.unsqueeze(0) # Add batch dim if missing
                            
                            # Ensure logits and generated_ids have compatible batch sizes
                            if probs.shape[0] != generated_ids_for_gather.shape[0]:
                                # This case might happen if batching logic changes upstream
                                # For now, assume batch size 1 if shapes mismatch unexpectedly
                                if probs.shape[0] == 1 and generated_ids_for_gather.shape[0] > 1:
                                     probs = probs.expand(generated_ids_for_gather.shape[0], -1, -1)
                                elif generated_ids_for_gather.shape[0] == 1 and probs.shape[0] > 1:
                                     generated_ids_for_gather = generated_ids_for_gather.expand(probs.shape[0], -1)
                                else: # If mismatch is not easily fixable, skip confidence
                                     raise ValueError("Logits and generated_ids batch sizes mismatch")

                            # Adjust generated_ids shape for gather if needed (B, S, 1)
                            if len(generated_ids_for_gather.shape) == 2:
                                generated_ids_for_gather = generated_ids_for_gather.unsqueeze(-1)

                            # Ensure generated_ids indices are within the vocab size
                            vocab_size = probs.shape[-1]
                            generated_ids_for_gather = generated_ids_for_gather.clamp(0, vocab_size - 1)

                            generated_probs = probs.gather(2, generated_ids_for_gather).squeeze(-1)
                            
                            # Mask out padding, BOS, and EOS tokens
                            pad_token_id = self.ocr_processor.tokenizer.pad_token_id
                            eos_token_id = self.ocr_processor.tokenizer.eos_token_id
                            bos_token_id = self.ocr_processor.tokenizer.bos_token_id # Use the actual BOS ID used

                            mask = (generated_ids != pad_token_id) & \
                                   (generated_ids != eos_token_id) & \
                                   (generated_ids != bos_token_id) # Use the correct BOS ID here
                            
                            # Ensure mask has the same shape as generated_probs for element-wise selection
                            if mask.shape != generated_probs.shape:
                                if len(mask.shape) == 1 and len(generated_probs.shape) == 2:
                                    mask = mask.unsqueeze(0).expand_as(generated_probs)
                                else:
                                     raise ValueError("Mask and generated_probs shape mismatch")

                            if mask.sum() > 0:
                                confidence = generated_probs[mask].mean().item()
                            else: # Handle cases with empty or only special tokens
                                confidence = 0.0
                                
                    except Exception as e:
                        # Fallback in case of unexpected errors during confidence calculation
                        print(f"Warning: Could not compute confidence score. Error: {e}")
                        confidence = None # Or 0.0, depending on desired behavior

                    result_dict["confidence"] = confidence

                batch_results.append(result_dict)
            
            return batch_results
            
        results = self.batch_process(
            images,
            process_batch
        )
        
        # Always return the list of dictionaries
        return results
    
    def analyze(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        analysis_type: str = "general",
        top_k: int = 5
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Analyze image content
        
        Args:
            images: Image or list of images (as paths or PIL Images)
            analysis_type: Type of analysis ("general", "scene", "object", etc.)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary or list of dictionaries with analysis results
        """
        # Load analysis model if needed
        self._load_analysis_model()
        
        # Handle single image input
        if isinstance(images, (str, Image.Image)):
            images = [images]
            single_input = True
        else:
            single_input = False
            
        def process_batch(batch_images):
            batch_results = []
            
            for image in batch_images:
                # Preprocess image if it's a path
                if isinstance(image, str):
                    image = self.load_image(image)
                
                # Prepare inputs
                inputs = self.analysis_processor(
                    image, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.analysis_model(**inputs)
                
                # Process predictions
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[0]
                top_probs, top_indices = probs.topk(top_k)
                
                # Create results dictionary
                results = {}
                for prob, idx in zip(top_probs, top_indices):
                    label = self.analysis_model.config.id2label[idx.item()]
                    results[label] = prob.item()
                
                batch_results.append(results)
            
            return batch_results
            
        results = self.batch_process(
            images,
            process_batch
        )
        
        return results[0] if single_input else results

class OCRProcessor(ImageTextProcessor):
    """Specialized class for OCR processing"""
    
    def __init__(
        self,
        model_name: str = "microsoft/trocr-large-printed",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        super().__init__(None, model_name, None, device, batch_size)
        # Directly load OCR model
        self._load_ocr_model()
    
    def extract_text(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        return_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract text from images using OCR.

        Args:
            images: Image or list of images (as paths or PIL Images).
            return_confidence: Whether to return confidence scores.

        Returns:
            A list of dictionaries. Each dictionary corresponds to an input image
            and contains at least the key 'text'. If return_confidence is True,
            it also includes the key 'confidence'.
        """
        return super().extract_text(images, return_confidence)
    
    def __call__(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        return_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """Extract text from images using OCR"""
        return self.extract_text(images, return_confidence)

class ImageCaptioner(ImageTextProcessor):
    """Specialized class for image captioning"""
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        super().__init__(model_name, None, None, device, batch_size)
    
    def caption(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        prompt: str = "A photo of",
        max_length: int = 30,
        num_captions: int = 1
    ) -> Union[str, List[str], List[List[str]]]:
        """Generate captions for images"""
        return super().caption(images, prompt, max_length, num_captions)
    
    def __call__(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        prompt: str = "A photo of",
        max_length: int = 30,
        num_captions: int = 1
    ) -> Union[str, List[str], List[List[str]]]:
        """Generate captions for images"""
        return self.caption(images, prompt, max_length, num_captions)

class ImageAnalyzer(ImageTextProcessor):
    """Specialized class for image analysis"""
    
    def __init__(
        self,
        model_name: str = "google/vit-large-patch16-224",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        super().__init__(None, None, model_name, device, batch_size)
        # Directly load analysis model
        self._load_analysis_model()
    
    def analyze(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        analysis_type: str = "general",
        top_k: int = 5
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Analyze image content"""
        return super().analyze(images, analysis_type, top_k)
    
    def __call__(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        analysis_type: str = "general",
        top_k: int = 5
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Analyze image content"""
        return self.analyze(images, analysis_type, top_k)
