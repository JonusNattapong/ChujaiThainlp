"""
Thai Text-to-Speech Synthesis Module
"""
from typing import Union, Optional
import numpy as np
import torch
from transformers import VitsModel, AutoTokenizer
from ..model_hub import ModelManager

class ThaiTTS:
    """Thai text-to-speech synthesis"""
    
    def __init__(self,
                model_name: str = "facebook/mms-tts-tha",  # Use Facebook's MMS TTS model for Thai
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

        self.model = VitsModel.from_pretrained(model_info['hf_id']).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_info['hf_id'])
        self.sample_rate = model_info.get('sample_rate', 16000)  # MMS-TTS uses 16kHz
        
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
        
        try:
            # Tokenize input text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Convert speaker_id if provided
            if speaker_id is not None:
                inputs['speaker_id'] = torch.tensor([speaker_id], device=self.device)
            
            # Generate audio with MMS-TTS
            with torch.no_grad():
                outputs = self.model(**inputs)
                waveform = outputs.waveform[0].cpu().numpy()
        except Exception as e:
            raise RuntimeError(f"Error generating speech: {str(e)}")
            
        # Apply post-processing effects
        if speed != 1.0 or pitch != 1.0:
            try:
                import librosa
                # Speed change
                if speed != 1.0:
                    waveform = librosa.effects.time_stretch(waveform, rate=1/speed)
                # Pitch adjustment
                if pitch != 1.0:
                    waveform = librosa.effects.pitch_shift(y=waveform, sr=self.sample_rate, n_steps=12 * np.log2(pitch))
            except ImportError:
                print("Warning: librosa not found. Speed and pitch adjustments skipped.")
            
        # Ensure output shape is correct
        waveform = waveform.squeeze()
        return waveform

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TTS"""
        import re
        # Thai-specific text normalization
        # Remove repeating spaces
        text = re.sub(r'\s+', ' ', text)
        # Add spaces around punctuation while preserving Thai characters
        text = re.sub(r'([!?,.:]|[-])', r' \1 ', text)
        # Clean up multiple spaces again and strip
        text = re.sub(r'\s+', ' ', text).strip()
        # Ensure proper spacing around Thai characters
        text = re.sub(r'([^-\s])([^-\s])', r'\1 \2', text)
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