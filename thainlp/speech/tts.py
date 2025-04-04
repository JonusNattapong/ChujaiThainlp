"""
Thai Text-to-Speech Synthesis Module
"""
from typing import Union, Optional
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from ..model_hub import ModelManager

class ThaiTTS:
    """Thai text-to-speech synthesis"""
    
    def __init__(self, 
                model_name: str = "thainer-speech",
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize TTS model
        
        Args:
            model_name: Name of TTS model 
            device: Device to run model on
        """
        self.device = device
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load TTS model"""
        manager = ModelManager()
        model_info = manager.get_model_info(self.model_name)
        
        if not model_info or model_info['task'] != 'tts':
            raise ValueError(f"Model {self.model_name} not found or invalid for TTS")

        self.model = AutoModel.from_pretrained(model_info['hf_id']).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_info['hf_id'])
        self.sample_rate = model_info.get('sample_rate', 22050)
        self.vocoder = None  # Will be lazy loaded if needed
        
    def _load_vocoder(self):
        """Lazy load vocoder"""
        if self.vocoder is None:
            from transformers import AutoModel
            # Load default vocoder
            self.vocoder = AutoModel.from_pretrained("facebook/hifigan").to(self.device)
        
    def synthesize(self, 
                  text: str,
                  speaker_id: Optional[int] = None,
                  speed: float = 1.0,
                  pitch: float = 1.0,
                  **kwargs) -> np.ndarray:
        """
        Synthesize speech from text
        
        Args:
            text: Input text
            speaker_id: Optional speaker ID for multi-speaker models
            speed: Speech speed (0.5-2.0)
            pitch: Pitch adjustment (0.5-2.0)
            
        Returns:
            audio: Generated audio waveform as numpy array
        """
        # Preprocess and normalize text
        text = self._preprocess_text(text)
        
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt"
        ).to(self.device)
        
        # Add optional parameters
        if speaker_id:
            inputs['speaker_id'] = torch.tensor([speaker_id])
        inputs['speed'] = torch.tensor([speed])
        inputs['pitch'] = torch.tensor([pitch])
        
        # Generate mel spectrogram
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert mel to waveform
        self._load_vocoder()
        with torch.no_grad():
            audio = self.vocoder(outputs['mel_spectrogram'])
            
        return audio.cpu().numpy().squeeze()

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TTS"""
        # TODO: Add Thai-specific text normalization
        import re
        text = re.sub(r'([!?,.:])', r' \1 ', text)  # Add space around punctuation
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def save_to_file(self, 
                   audio: np.ndarray,
                   filename: str,
                   format: str = "wav"):
        """
        Save generated audio to file
        
        Args:
            audio: Audio waveform (numpy array)
            filename: Output filename
            format: Audio format ('wav', 'mp3', etc)
        """
        import soundfile as sf
        sf.write(filename, audio, samplerate=self.sample_rate, format=format)

    def synthesize_to_file(self,
                         text: str,
                         filename: str,
                         format: str = "wav",
                         **kwargs):
        """
        Synthesize speech and save directly to file
        
        Args:
            text: Input text
            filename: Output filename
            format: Audio format
            **kwargs: Additional arguments for synthesize()
        """
        audio = self.synthesize(text, **kwargs)
        self.save_to_file(audio, filename, format)