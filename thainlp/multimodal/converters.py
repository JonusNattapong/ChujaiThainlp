"""
Modality conversion for transforming between different data modalities
"""
from typing import Dict, List, Union, Optional, Any, Tuple
import os
import torch
import numpy as np
from PIL import Image
import tempfile
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    CLIPTextModel,
    CLIPTokenizer
)
from .base import MultimodalBase
from .audio_text import AudioTextProcessor
from .image_text import ImageTextProcessor
from ..vision.generation import Text2Image
from ..speech.tts import ThaiTTS

class ModalityConverter(MultimodalBase):
    """Convert between different data modalities"""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        """Initialize modality converter
        
        Args:
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(None, device, batch_size)
        
        # Initialize component converters
        self.converters = {}
        
    def _get_converter(self, source_type: str, target_type: str):
        """Get or load converter for specific modality types"""
        converter_key = f"{source_type}_to_{target_type}"
        
        if converter_key not in self.converters:
            # Initialize converter based on source and target types
            if source_type == "text" and target_type == "text":
                self.converters[converter_key] = Text2TextConverter(device=self.device)
            elif source_type == "text" and target_type == "image":
                self.converters[converter_key] = Text2ImageConverter(device=self.device)
            elif source_type == "text" and target_type == "audio":
                self.converters[converter_key] = TextToAudioConverter(device=self.device)
            elif source_type == "audio" and target_type == "text":
                self.converters[converter_key] = AudioToTextConverter(device=self.device)
            elif source_type == "image" and target_type == "text":
                self.converters[converter_key] = ImageToTextConverter(device=self.device)
            else:
                raise ValueError(f"Unsupported conversion: {source_type} to {target_type}")
                
        return self.converters[converter_key]
    
    def convert(
        self,
        source: Union[str, Dict[str, Any]],
        source_type: str,
        target_type: str,
        **kwargs
    ) -> Any:
        """Convert content from one modality to another
        
        Args:
            source: Source content (path or data)
            source_type: Source modality type ('text', 'image', 'audio', 'video')
            target_type: Target modality type ('text', 'image', 'audio', 'video')
            **kwargs: Additional conversion parameters
            
        Returns:
            Converted content in target modality
        """
        # Get appropriate converter
        converter = self._get_converter(source_type, target_type)
        
        # Perform conversion
        return converter.convert(source, **kwargs)

class Text2TextConverter(ModalityConverter):
    """Convert text from one format to another (translation, summarization, etc.)"""
    
    def __init__(
        self,
        model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8
    ):
        super().__init__(device, batch_size)
        
        # Store model name for lazy loading
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        """Load model on demand"""
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
    
    def convert(
        self,
        source: Union[str, Dict[str, Any]],
        conversion_type: str = "translate",
        source_lang: str = "th",
        target_lang: str = "en",
        max_length: int = 512,
        min_length: Optional[int] = None,
        **kwargs
    ) -> str:
        """Convert text from one format to another
        
        Args:
            source: Source text or text object
            conversion_type: Type of conversion ('translate', 'summarize', etc.)
            source_lang: Source language code
            target_lang: Target language code
            max_length: Maximum length of output text
            min_length: Minimum length of output text
            **kwargs: Additional conversion parameters
            
        Returns:
            Converted text
        """
        # Extract text from source
        if isinstance(source, dict) and "text" in source:
            text = source["text"]
        else:
            text = source
            
        # Load model
        self._load_model()
        
        # Process based on conversion type
        if conversion_type == "translate":
            # For mBART-50 multilingual translation
            if hasattr(self.tokenizer, "src_lang"):
                self.tokenizer.src_lang = source_lang
                
            # Encode text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            ).to(self.device)
            
            # Set forced decoder IDs for target language
            forced_bos_token_id = None
            if hasattr(self.tokenizer, "lang_code_to_id"):
                forced_bos_token_id = self.tokenizer.lang_code_to_id.get(target_lang)
                
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    forced_bos_token_id=forced_bos_token_id,
                    num_beams=kwargs.get("num_beams", 4),
                    early_stopping=True
                )
                
            # Decode output
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated_text
            
        elif conversion_type == "summarize":
            # Encode text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=kwargs.get("summary_max_length", 150),
                    min_length=kwargs.get("summary_min_length", 50),
                    num_beams=kwargs.get("num_beams", 4),
                    early_stopping=True
                )
                
            # Decode output
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
            
        else:
            raise ValueError(f"Unsupported conversion type: {conversion_type}")

class Text2ImageConverter(ModalityConverter):
    """Convert text to images"""
    
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-2-1",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1
    ):
        super().__init__(device, batch_size)
        
        # Store model name for lazy loading
        self.model_name = model_name
        self.generator = None
        
    def _load_model(self):
        """Load model on demand"""
        if self.generator is None:
            self.generator = Text2Image(
                model_name=self.model_name,
                device=self.device
            )
    
    def convert(
        self,
        source: str,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Image.Image:
        """Convert text to image
        
        Args:
            source: Source text prompt
            width: Width of generated image
            height: Height of generated image
            num_inference_steps: Number of inference steps
            negative_prompt: Optional negative prompt
            **kwargs: Additional conversion parameters
            
        Returns:
            Generated image
        """
        # Extract text from source
        if isinstance(source, dict) and "text" in source:
            text = source["text"]
        else:
            text = source
            
        # Load generator
        self._load_model()
        
        # Generate image
        image = self.generator.generate(
            text,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            **kwargs
        )
        
        return image

class TextToAudioConverter(ModalityConverter):
    """Convert text to audio"""
    
    def __init__(
        self,
        model_name: str = "tts-thai/ltl-th",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1
    ):
        super().__init__(device, batch_size)
        
        # Store model name for lazy loading
        self.model_name = model_name
        self.tts = None
        
    def _load_model(self):
        """Load model on demand"""
        if self.tts is None:
            self.tts = ThaiTTS(
                model_name=self.model_name,
                device=self.device
            )
    
    def convert(
        self,
        source: Union[str, Dict[str, Any]],
        voice_id: int = 0,
        speed: float = 1.0,
        save_path: Optional[str] = None,
        return_audio: bool = True,
        **kwargs
    ) -> Union[np.ndarray, str]:
        """Convert text to audio
        
        Args:
            source: Source text
            voice_id: Voice ID to use for synthesis
            speed: Speaking speed factor
            save_path: Optional path to save audio file
            return_audio: Whether to return audio array
            **kwargs: Additional conversion parameters
            
        Returns:
            Audio array or path to saved audio file
        """
        # Extract text from source
        if isinstance(source, dict) and "text" in source:
            text = source["text"]
        else:
            text = source
            
        # Load TTS model
        self._load_model()
        
        # Generate speech
        audio, sample_rate = self.tts.synthesize(
            text,
            voice_id=voice_id,
            speed=speed,
            **kwargs
        )
        
        # Save if path is provided
        if save_path:
            from ..speech.audio_utils import AudioUtils
            AudioUtils.save_audio(audio, sample_rate, save_path)
            
            if not return_audio:
                return save_path
        
        return audio if return_audio else None

class AudioToTextConverter(ModalityConverter):
    """Convert audio to text"""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1
    ):
        super().__init__(device, batch_size)
        
        # Store model name for lazy loading
        self.model_name = model_name
        self.processor = None
        
    def _load_processor(self):
        """Load processor on demand"""
        if self.processor is None:
            self.processor = AudioTextProcessor(
                model_name=self.model_name,
                device=self.device
            )
    
    def convert(
        self,
        source: Union[str, np.ndarray, Dict[str, Any]],
        language: Optional[str] = None,
        return_timestamps: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Convert audio to text
        
        Args:
            source: Source audio file path or numpy array
            language: Language code for transcription
            return_timestamps: Whether to return timestamps
            **kwargs: Additional conversion parameters
            
        Returns:
            Transcribed text or dictionary with text and metadata
        """
        # Extract audio from source
        if isinstance(source, dict) and "audio" in source:
            audio = source["audio"]
        else:
            audio = source
            
        # Load processor
        self._load_processor()
        
        # Transcribe audio
        result = self.processor.transcribe(
            audio,
            language=language,
            return_timestamps=return_timestamps,
            **kwargs
        )
        
        return result

class ImageToTextConverter(ModalityConverter):
    """Convert image to text"""
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        super().__init__(device, batch_size)
        
        # Store model name for lazy loading
        self.model_name = model_name
        self.processor = None
        
    def _load_processor(self):
        """Load processor on demand"""
        if self.processor is None:
            self.processor = ImageTextProcessor(
                model_name=self.model_name,
                device=self.device
            )
    
    def convert(
        self,
        source: Union[str, Image.Image, Dict[str, Any]],
        mode: str = "caption",
        prompt: str = "A photo of",
        max_length: int = 30,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Convert image to text
        
        Args:
            source: Source image file path or PIL Image
            mode: Conversion mode ('caption', 'ocr', 'analyze')
            prompt: Text prompt for captioning
            max_length: Maximum length of generated text
            **kwargs: Additional conversion parameters
            
        Returns:
            Generated text or dictionary with text and metadata
        """
        # Extract image from source
        if isinstance(source, dict) and "image" in source:
            image = source["image"]
        else:
            image = source
            
        # Load processor
        self._load_processor()
        
        # Process based on mode
        if mode == "caption":
            return self.processor.caption(
                image,
                prompt=prompt,
                max_length=max_length,
                **kwargs
            )
        elif mode == "ocr":
            return self.processor.extract_text(
                image,
                **kwargs
            )
        elif mode == "analyze":
            return self.processor.analyze(
                image,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported conversion mode: {mode}")
