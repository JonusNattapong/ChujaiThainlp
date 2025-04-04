"""
Voice Activity Detection Module
"""
from typing import Union, List, Tuple
import numpy as np
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

class VoiceActivityDetector:
    """Detect speech segments in audio"""
    
    def __init__(self,
                model_name: str = "thainer-vad",
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
        """Load VAD model and feature extractor"""
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_name).to(self.device)
        self.sample_rate = self.feature_extractor.sampling_rate
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
        
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio[start:end]
            
            # Extract features
            inputs = self.feature_extractor(
                frame,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                speech_prob = probs[0, 1].item()  # Assuming class 1 is speech
                
            # Detect speech
            if speech_prob >= threshold:
                start_time = i * self.frame_duration
                if current_segment is None:
                    current_segment = [start_time, start_time + self.frame_duration]
                else:
                    current_segment[1] = start_time + self.frame_duration
            else:
                if current_segment is not None:
                    # Only keep segments longer than minimum duration
                    duration = current_segment[1] - current_segment[0]
                    if duration >= self.min_speech_duration:
                        speech_segments.append(tuple(current_segment))
                    current_segment = None
                    
        # Add final segment if needed
        if current_segment is not None:
            duration = current_segment[1] - current_segment[0]
            if duration >= self.min_speech_duration:
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