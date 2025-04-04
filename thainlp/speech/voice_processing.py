"""
Voice Processing for Conversion and Style Transfer
"""
from typing import Optional, Union
import numpy as np
import torch
from transformers import AutoModel, AutoFeatureExtractor
from .audio_utils import AudioUtils

class VoiceProcessor:
    """Voice conversion and style transfer"""
    
    def __init__(self,
                model_name: str = "thainer-voice",
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize voice processing model
        
        Args:
            model_name: Name of voice model
            device: Device to run model on
        """
        self.device = device
        self.model_name = model_name
        self._load_model()
        
    def _load_model(self):
        """Load voice model and feature extractor"""
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.sample_rate = self.feature_extractor.sampling_rate
        
    def convert_voice(self,
                    source_audio: Union[np.ndarray, str, bytes],
                    target_voice_id: Optional[int] = None,
                    target_style: Optional[str] = None,
                    **kwargs) -> np.ndarray:
        """
        Convert voice characteristics
        
        Args:
            source_audio: Input audio
            target_voice_id: Target speaker ID
            target_style: Target style (e.g. "happy", "sad")
            
        Returns:
            Converted audio
        """
        # Load and preprocess audio
        if isinstance(source_audio, str) or isinstance(source_audio, bytes):
            audio, _ = AudioUtils.load_audio(source_audio, target_sr=self.sample_rate)
        else:
            audio = source_audio
            
        # Extract features
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)
        
        # Add style/voice parameters
        if target_voice_id:
            inputs['voice_id'] = torch.tensor([target_voice_id])
        if target_style:
            inputs['style'] = target_style
            
        # Convert voice
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process
        converted = outputs['audio'].cpu().numpy().squeeze()
        converted = AudioUtils.normalize_audio(converted)
        
        return converted
        
    def transfer_style(self,
                      source_audio: Union[np.ndarray, str, bytes],
                      style_audio: Union[np.ndarray, str, bytes],
                      **kwargs) -> np.ndarray:
        """
        Transfer style from one audio to another
        
        Args:
            source_audio: Audio to convert
            style_audio: Audio with target style
            
        Returns:
            Style-transferred audio
        """
        # Load and preprocess audios
        source, _ = AudioUtils.load_audio(source_audio, target_sr=self.sample_rate)
        style, _ = AudioUtils.load_audio(style_audio, target_sr=self.sample_rate)
        
        # Extract features
        source_inputs = self.feature_extractor(
            source,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)
            
        style_inputs = self.feature_extractor(
            style,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)
        
        # Transfer style
        with torch.no_grad():
            outputs = self.model.transfer_style(
                source_inputs,
                style_inputs
            )
            
        # Post-process
        transferred = outputs['audio'].cpu().numpy().squeeze()
        transferred = AudioUtils.normalize_audio(transferred)
        
        return transferred
        
    def blend_voices(self,
                    audio1: Union[np.ndarray, str, bytes],
                    audio2: Union[np.ndarray, str, bytes],
                    ratio: float = 0.5,
                    **kwargs) -> np.ndarray:
        """
        Blend characteristics of two voices
        
        Args:
            audio1: First voice
            audio2: Second voice
            ratio: Blend ratio (0-1)
            
        Returns:
            Blended voice audio
        """
        # Load and preprocess audios
        audio1, _ = AudioUtils.load_audio(audio1, target_sr=self.sample_rate)
        audio2, _ = AudioUtils.load_audio(audio2, target_sr=self.sample_rate)
        
        # Extract features
        inputs1 = self.feature_extractor(
            audio1,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)
            
        inputs2 = self.feature_extractor(
            audio2,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)
        
        # Blend voices
        with torch.no_grad():
            outputs = self.model.blend_voices(
                inputs1,
                inputs2,
                ratio=ratio
            )
            
        # Post-process
        blended = outputs['audio'].cpu().numpy().squeeze()
        blended = AudioUtils.normalize_audio(blended)
        
        return blended