"""
Audio Utilities for Processing and Format Handling
"""
from typing import Union, Tuple
import numpy as np
import soundfile as sf
import io

class AudioUtils:
    """Audio processing utilities"""
    
    @staticmethod
    def load_audio(file: Union[str, bytes], 
                  target_sr: int = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file from path or bytes
        
        Args:
            file: Audio file path or bytes
            target_sr: Optional target sample rate for resampling
            
        Returns:
            (audio_array, sample_rate)
        """
        if isinstance(file, str):
            audio, sr = sf.read(file)
        else:  # bytes
            with io.BytesIO(file) as f:
                audio, sr = sf.read(f)
                
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Resample if needed
        if target_sr and sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            
        return audio, sr
        
    @staticmethod
    def save_audio(audio: np.ndarray,
                  sr: int,
                  filename: str,
                  format: str = "wav"):
        """
        Save audio to file
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            filename: Output file path
            format: Audio format ('wav', 'mp3', etc)
        """
        sf.write(filename, audio, samplerate=sr, format=format)
        
    @staticmethod
    def convert_audio(input_audio: Union[str, bytes],
                     output_format: str = "wav",
                     target_sr: int = None) -> bytes:
        """
        Convert audio between formats
        
        Args:
            input_audio: Input file path or bytes
            output_format: Target format ('wav', 'mp3', etc)
            target_sr: Target sample rate
            
        Returns:
            bytes containing converted audio
        """
        audio, sr = AudioUtils.load_audio(input_audio, target_sr=target_sr)
        
        with io.BytesIO() as f:
            sf.write(f, audio, samplerate=sr or 16000, format=output_format)
            f.seek(0)
            return f.read()
            
    @staticmethod
    def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """
        Normalize audio to target dB level
        
        Args:
            audio: Input audio
            target_db: Target dB level
            
        Returns:
            Normalized audio
        """
        import librosa
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 10 ** (target_db / 20)
        factor = target_rms / (rms + 1e-8)
        return audio * factor
        
    @staticmethod
    def trim_silence(audio: np.ndarray,
                    sr: int,
                    top_db: int = 30,
                    frame_length: int = 2048,
                    hop_length: int = 512) -> np.ndarray:
        """
        Trim leading/trailing silence from audio
        
        Args:
            audio: Input audio
            sr: Sample rate
            top_db: Threshold in dB for silence
            frame_length: FFT window size
            hop_length: Hop length between frames
            
        Returns:
            Trimmed audio
        """
        import librosa
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        return trimmed
        
    @staticmethod
    def mix_audios(audio1: np.ndarray,
                  audio2: np.ndarray,
                  ratio: float = 0.5) -> np.ndarray:
        """
        Mix two audio signals
        
        Args:
            audio1: First audio
            audio2: Second audio
            ratio: Mix ratio (0-1)
            
        Returns:
            Mixed audio
        """
        min_len = min(len(audio1), len(audio2))
        mixed = ratio * audio1[:min_len] + (1 - ratio) * audio2[:min_len]
        return mixed / np.max(np.abs(mixed))