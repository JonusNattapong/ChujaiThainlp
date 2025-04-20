"""
Voice Activity Detection Module
"""
from typing import Union, List, Tuple
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class VoiceActivityDetector:
    """Detect speech segments in audio"""
    
    def __init__(self,
                model_name: str = "facebook/wav2vec2-base",
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize VAD model
        
        Args:
            model_name: Name of VAD model
            device: Device to run model on
        """
        self.device = device
        self.model_name = model_name
        self._load_model()
        
    def _load_model(self):
        """Load Wav2Vec2 model for VAD"""
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)
        self.sample_rate = 16000
        self.frame_duration = 0.02  # 20ms frames
        self.min_speech_duration = 0.3  # Minimum speech segment duration
        
    def detect(self,
              audio: Union[np.ndarray, str, bytes],
              threshold: float = 0.5) -> List[Tuple[float, float]]:
        """
        Detect speech segments in audio
        
        Args:
            audio: Input audio as:
                  - numpy array
                  - file path (str)
                  - bytes (raw audio data)
            threshold: Confidence threshold (0-1)
            
        Returns:
            List of (start, end) timestamps in seconds
        """
        # Load audio if needed
        if isinstance(audio, str):
            audio = self._load_audio_file(audio)
        elif isinstance(audio, bytes):
            audio = self._bytes_to_array(audio)
            
        # Process audio in chunks
        frame_size = int(self.sample_rate * self.frame_duration)
        num_frames = len(audio) // frame_size
        speech_segments = []
        current_segment = None
        
        # Process audio with Wav2Vec2
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use feature activations to detect speech
            features = outputs.last_hidden_state.squeeze()
            energy = torch.norm(features, dim=-1)
            # Normalize energy
            energy = (energy - energy.mean()) / energy.std()
                
        # Detect speech segments based on energy
        for i in range(len(energy)):
            frame_energy = energy[i].item()
            if frame_energy > threshold:
                time = i * self.frame_duration
                if current_segment is None:
                    current_segment = [time, time + self.frame_duration]
                else:
                    current_segment[1] = time + self.frame_duration
            elif current_segment is not None:
                if current_segment[1] - current_segment[0] >= self.min_speech_duration:
                    speech_segments.append(tuple(current_segment))
                current_segment = None
        
        # Handle last segment
        if current_segment is not None:
            if current_segment[1] - current_segment[0] >= self.min_speech_duration:
                speech_segments.append(tuple(current_segment))
        
        return speech_segments
        
    def _load_audio_file(self, path: str) -> np.ndarray:
        """Load audio file into numpy array"""
        import soundfile as sf
        audio, sr = sf.read(path)
        
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
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
            
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Resample if needed
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
        return audio