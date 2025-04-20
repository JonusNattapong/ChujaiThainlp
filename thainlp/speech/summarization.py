"""
Speech Summarization Module
"""
from typing import Optional, Union
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from ..model_hub import ModelManager

class SpeechSummarizer:
    """Speech summarization using multimodal models"""
    
    def __init__(self,
                model_name: str = "openai/whisper-large-v3",
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize speech summarizer
        
        Args:
            model_name: Name of summarization model
            device: Device to run model on
        """
        self.device = device
        self.model_name = model_name
        self._load_model()
        
    def _load_model(self):
        """Load summarization model"""
        manager = ModelManager()
        model_info = manager.get_model_info(self.model_name)
        
        if not model_info or model_info['task'] != 'speech-summarization':
            raise ValueError(f"Model {self.model_name} not found or invalid for speech summarization")
            
        self.processor = WhisperProcessor.from_pretrained(model_info['hf_id'])
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_info['hf_id']
        ).to(self.device)
        self.sample_rate = model_info.get('sample_rate', 16000)
        
    def summarize(self,
                audio: Union[np.ndarray, str, bytes],
                max_length: int = 150,
                min_length: int = 30,
                **kwargs) -> str:
        """
        Summarize speech content
        
        Args:
            audio: Input audio as:
                  - numpy array (waveform)
                  - file path (str) 
                  - bytes (raw audio data)
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Generated summary text
        """
        # Load audio if needed
        if isinstance(audio, str):
            audio = self._load_audio_file(audio)
        elif isinstance(audio, bytes):
            audio = self._bytes_to_array(audio)
            
        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            # Create attention mask
            attention_mask = torch.ones_like(inputs.input_features)
            
            generated_ids = self.model.generate(
                inputs.input_features,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                language="th",
                task="transcribe",
                forced_decoder_ids=None
            )
            
        # Decode output
        summary = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return summary.strip()
    
    def _load_audio_file(self, path: str) -> np.ndarray:
        """Load audio file into numpy array"""
        import soundfile as sf
        audio, sr = sf.read(path)
        
        # Resample if needed
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
        return audio
        
    def _bytes_to_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert raw bytes to numpy array"""
        import soundfile as sf
        import io
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f)
            
        # Resample if needed
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
        return audio
